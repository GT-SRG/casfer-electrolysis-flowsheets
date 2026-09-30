#------------------------------------------------------------------------------
# function:    flowsheet2.py                                                  #
# Description: Flowsheet 2                                                    #
#              Feed -> Prep Tank -> Electrolyzer -> GO membrane dewatering    #
#              GO retentate -> Acid (receive) tank -> E-GROW line             #
#              GO permeate  -> NF (bivalent removal) -> GPM                   #
#              GPM ammonium sulfate product + acid tank outlet -> combined    #
#              E-GROW product -> Storage                                      #
#              NF retentate and GPM raffinate are combined, neutralized to    #
#              pH 7 with H2SO4 (neutralization tank), and returned to the     #
#              WWTP.                                                          #
#                                                                             #
#              Every unit carries the full component stream (streamTools):   #
#              this script only connects outlets to inlets, sets targets,    #
#              and assembles the cost objective.                              #
#------------------------------------------------------------------------------

import os
import sys

repoRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repoRoot not in sys.path:
    sys.path.insert(0, repoRoot)

import pyomo.environ as pyo
import src.singleUnitModels.getParams as getParams
import src.singleUnitModels.getStandardParams as getStandardParams
import src.singleUnitModels.streamTools as streamTools
import src.singleUnitModels.flowsheetTools as flowsheetTools
from src.singleUnitModels.mixer import mixer
import src.singleUnitModels.prepTank as prepTank
import src.singleUnitModels.electrolyzer as electrolyzer
import src.singleUnitModels.goMembraneDewatering as goMembraneDewatering
import src.singleUnitModels.nanofiltration as nanofiltration
import src.singleUnitModels.receiveTank as receiveTank
import src.singleUnitModels.gpm as gpm
import src.singleUnitModels.storageTank as storageTank


def buildModel():
    m = pyo.ConcreteModel()
    try:
        getParams.getParams(m)
    except Exception:
        pass
    getStandardParams.getStandardParams(m)

    # ------------------------------ Feeds ------------------------------
    flowsheetTools.addStandardFeeds(m, feedPH=7.5)
    m.targetProductTss = pyo.Param(initialize=0.25, mutable=True)   # combined product TSS

    # ------------------------------ Units ------------------------------
    prepTank.prepTank(m)
    electrolyzer.electrolyzer(m)
    goMembraneDewatering.goMembraneDewatering(m)
    receiveTank.receiveTank(m)          # acid tank on the E-GROW (retentate) line
    nanofiltration.nf(m)                # bivalent removal on GO permeate
    gpm.gpm(m)
    mixer(m, 'mxProduct', ['egrow', 'gpmProduct'], pHOut=7.0)   # combined E-GROW product
    storageTank.storageTank(m)
    mixer(m, 'mxReject', ['nfRetentate', 'gpmRaffinate'], pHOut=13.0)   # both rejects leave at ~pH 13
    receiveTank.receiveTank(m, 'nt')    # reject neutralization to pH 7 before return to the WWTP
    m.nt.ohFromInletPH.set_value(1.0)

    m.pt.sludgeTSSOutTarget.set_value(0.08)
    m.pt.targetTSSConstr.deactivate()
    m.pt.waterMassFlowIn.fix(0.0)
    # GO retentate TSS is free: the combined-product TSS target below sets how
    # far GO must dewater to compensate for the GPM product liquid.
    m.go.targetSolidsConstraint.deactivate()
    m.mxProduct.pHOutSpec.set_value(pyo.value(m.rt.targetpH))

    # --------------------------- Connections ---------------------------
    streamTools.connectStreams(m, 'sludgeToPt', m.sludgeFeed.outlet, m.pt.sludgeIn)
    streamTools.connectStreams(m, 'centrateToPt', m.centrateFeed.outlet, m.pt.centrateIn)
    streamTools.connectStreams(m, 'ptToEl', m.pt.outlet, m.el.inlet)
    streamTools.connectStreams(m, 'elToGo', m.el.outlet, m.go.inlet)
    streamTools.connectStreams(m, 'goRetentateToRt', m.go.retentate, m.rt.inlet)
    streamTools.connectStreams(m, 'goPermeateToNf', m.go.permeate, m.nf.inlet)
    streamTools.connectStreams(m, 'nfPermeateToGpm', m.nf.permeate, m.gpm.inlet)
    streamTools.connectStreams(m, 'rtToProduct', m.rt.outlet, m.mxProduct.egrow)
    streamTools.connectStreams(m, 'gpmToProduct', m.gpm.product, m.mxProduct.gpmProduct)
    streamTools.connectStreams(m, 'productToStorage', m.mxProduct.outlet, m.st.inlet)
    streamTools.connectStreams(m, 'nfRetentateToReject', m.nf.retentate, m.mxReject.nfRetentate)
    streamTools.connectStreams(m, 'gpmRaffinateToReject', m.gpm.raffinate, m.mxReject.gpmRaffinate)
    streamTools.connectStreams(m, 'rejectToNeutralization', m.mxReject.outlet, m.nt.inlet)

    # At least 90% TAN capture across the GPM (fresh-feed basis)
    m.gpmCaptureTarget = pyo.Constraint(expr=m.gpm.concOut <= 0.10 * m.gpm.concInFresh)

    # Combined product TSS target (linear form)
    m.combinedProductTSSTarget = pyo.Constraint(
        expr=m.mxProduct.outlet.solidsMass == m.targetProductTss * m.mxProduct.outlet.totalMass
    )

    # --- GPM excess-acid credit to the receive tank ---
    # H+ equivalents dosed to the GPM draw as H2SO4 minus those consumed by NH3 capture
    m.gpmTotalAcidEqAdded = pyo.Expression(expr=m.gpm.acidFlowIn * m.gpm.acidDensity * m.gpm.acidWtFracIn / 0.098 * 2.0)
    m.gpmAcidConsumedByNH3 = pyo.Expression(expr=m.gpm.nRemoved / 0.014)
    m.gpmExcessAcidEq = pyo.Expression(expr=m.gpmTotalAcidEqAdded - m.gpmAcidConsumedByNH3)
    m.rt.acidEqCredit.unfix()
    m.rtAcidCreditLink = pyo.Constraint(expr=m.rt.acidEqCredit == m.gpmExcessAcidEq)

    # ------------------------------ Product ------------------------------
    flowsheetTools.addProductQuality(m, m.st.outlet)

    # ------------------------------ Costs ------------------------------
    units = (m.pt, m.el, m.go, m.nf, m.rt, m.gpm, m.st, m.nt)
    m.totalCapex = pyo.Expression(expr=1.35 * sum(u.capex for u in units))
    m.totalOpex = pyo.Expression(expr=sum(u.opex for u in units))
    m.totalCost = pyo.Objective(expr=(m.totalCapex + m.totalOpex) / 1e6, sense=pyo.minimize)

    # Reported cost basis
    m.reportedOpex = pyo.Expression(expr=m.totalOpex + m.el.opex)
    m.reportedCost = pyo.Expression(expr=m.totalCapex + m.reportedOpex)
    m.wetTonnesTotal = pyo.Expression(expr=m.st.outlet.totalMass * m.daysOperation * 1e-3)
    m.nTonnesTotal = pyo.Expression(expr=m.st.outlet.totalN * m.daysOperation * 1e-3)
    m.costPerWetTonne = pyo.Expression(expr=m.reportedCost / (m.wetTonnesTotal + 1e-9))
    m.costPerTonneN = pyo.Expression(expr=m.reportedCost / (m.nTonnesTotal + 1e-9))
    return m


def printResults(m):
    v = flowsheetTools.safeValue
    print('================ FEED / TARGETS ==================')
    print('Feed mass flow (kg/day):', v(m.feedMassFlow) * 86400, '  target combined TSS:', v(m.targetProductTss))
    print('================ PREP TANK / ELECTROLYZER ========')
    print('CaO (kg/day):', v(m.pt.totalCaO) * 86400, ' prep tank volume (m3):', v(m.pt.tankVolume))
    print('Electrolyzer area (m2):', v(m.el.area), ' power (kW):', v(m.el.power),
          ' liberated N (kg-N/day):', v(m.el.liberatedN) * 86400, ' TAN lost (kg-N/day):', v(m.el.nitrogenLost) * 86400)
    print('================ GO MEMBRANE DEWATERING ==========')
    flowsheetTools.printStream('GO retentate', m.go.retentate)
    flowsheetTools.printStream('GO permeate', m.go.permeate)
    print('Area (m2):', v(m.go.area), ' deltaP (bar):', v(m.go.deltaP), ' delta-pi (bar):', v(m.go.deltaPi),
          ' liquid retention:', v(m.go.liquidRetentionFrac), ' orgN retention:', v(m.go.orgNRetentionFrac))
    print('================ NANOFILTRATION ==================')
    flowsheetTools.printStream('NF permeate', m.nf.permeate)
    flowsheetTools.printStream('NF retentate (reject)', m.nf.retentate)
    print('Recovery:', v(m.nf.recovery), ' area (m2):', v(m.nf.area), ' deltaP (bar):', v(m.nf.deltaP),
          ' TAN to permeate:', v(m.nf.nRecoveryToPermeate), ' Ca/Mg to retentate:', v(m.nf.caRejectionToRetentate), v(m.nf.mgRejectionToRetentate))
    print('================ GPM ============================')
    print('Fresh feed TAN (ppm):', v(m.gpm.concInFresh) * 1000, ' out (ppm):', v(m.gpm.concOut) * 1000)
    print('Area (m2):', v(m.gpm.area), ' recirculation ratio:', v(m.gpm.recircRatio), ' outlet pH:', v(m.gpm.pHOut))
    print('N captured (kg-N/day):', v(m.gpm.nRemoved) * 86400, ' product (kg/day):', v(m.gpm.productMassFlow) * 86400,
          ' product N wt%:', v(m.gpm.nWtPercent), ' product pH:', v(m.gpm.productPH), ' (NH4+-only estimate:', v(m.gpm.productPHEst), ')')
    print('Acid (m3/day):', v(m.gpm.acidFlowIn) * 86400, ' acid wt%:', v(m.gpm.acidWtFracIn) * 100,
          ' extra dilution water (kg/day):', v(m.gpm.extraDilutionWaterNeeded) * 86400)
    print('Water flux into draw (kg/day):', v(m.gpm.jWaterVaporMassFlow) * 86400, ' (kg/m2/h):', v(m.gpm.waterFluxKgM2h),
          ' draw aw in/out:', v(m.gpm.awDrawIn), v(m.gpm.awDrawOut), ' product AS wt%:', v(m.gpm.productAsWtPercent))
    flowsheetTools.printStream('GPM raffinate (effluent)', m.gpm.raffinate)
    print('================ RECEIVE (ACID) TANK =============')
    print('H2SO4 (kg/day, pure, net of GPM credit):', v(m.rt.h2so4RequiredKgPerS) * 86400,
          ' GPM credit (mol H+/day):', v(m.rt.acidEqCredit) * 86400)
    print('================ REJECT NEUTRALIZATION ===========')
    print('H2SO4 (kg/day, pure):', v(m.nt.h2so4RequiredKgPerS) * 86400, ' acid eq (mol/day): OH-', v(m.nt.ohNeutralizationDemand) * 86400,
          ' TAN', v(m.nt.tanProtonationDemand) * 86400, ' buffer', v(m.nt.bufferingAcidDemand) * 86400)
    flowsheetTools.printStream('Neutralized reject to WWTP', m.nt.outlet)
    flowsheetTools.printStream('Combined E-GROW product', m.st.outlet)
    flowsheetTools.printProductQuality('COMBINED E-GROW PRODUCT (FS2)', m.quality)
    flowsheetTools.printBalance(
        m, [m.st.outlet],
        nLosses=[('Electrolyzer TAN loss', m.el.nitrogenLost)],
        otherOutletStreams=[('Neutralized reject to WWTP (NF retentate + GPM raffinate)', m.nt.outlet)],
    )
    print('================ COSTS ===========================')
    wetT = v(m.wetTonnesTotal)
    for label, blk, opexFactor in (('Prep tank', m.pt, 1), ('Electrolyzer', m.el, 2), ('GO membrane', m.go, 1),
                                   ('Nanofiltration', m.nf, 1), ('Receive tank', m.rt, 1), ('GPM', m.gpm, 1), ('Storage tank', m.st, 1), ('Reject neutralization', m.nt, 1)):
        print(f'{label} capex / opex ($/wet t): {1.35 * v(blk.capex) / wetT:.4f} / {opexFactor * v(blk.opex) / wetT:.4f}')
    print('Total capex / opex / cost ($ lifetime):', v(m.totalCapex), v(m.reportedOpex), v(m.reportedCost))
    print('Total cost ($/wet t combined product):', v(m.costPerWetTonne))
    print('Total cost ($/dry t combined product):', v(m.reportedCost) / (v(m.st.outlet.solidsMass) * v(m.daysOperation) * 1e-3))
    print('Total cost ($/t-N):', v(m.costPerTonneN))
    print('================ ELECTRICITY / CHEMICALS =========')
    print('Electrolyzer (kWh/day):', v(m.el.power) * 24.0, ' NF pump (kWh/day):', v(m.nf.pumpPower) * 24.0,
          ' GO pump (kWh/day):', v(m.go.pumpPower) * 24.0, ' GPM pump (kWh/day):', v(m.gpm.pumpPower) * 24.0)
    print('CaO (kg/day):', v(m.pt.totalCaO) * 86400, ' receive-tank acid solution (kg/day):', v(m.rt.acidMassFlowIn) * 86400,
          ' GPM acid solution (kg/day):', v(m.gpm.acidMassFlow) * 86400,
          ' reject-neutralization acid solution (kg/day):', v(m.nt.acidMassFlowIn) * 86400)


if __name__ == '__main__':
    m = buildModel()
    tol = 1e-3
    solver = pyo.SolverFactory('baron', options={'EpsA': 1 * tol, 'AbsConFeasTol': 1 * tol, 'TDo': 0, 'MDo': 0, 'OBTTDo': 1},
                               executable='C:/baron/baron.exe')
    sol = solver.solve(m, tee=True)
    printResults(m)