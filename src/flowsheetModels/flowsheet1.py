#------------------------------------------------------------------------------
# function:    flowsheet1.py                                                  #
# Description: Optimization model for wastewater treatment process            #
#              Process flow: Feed -> Prep Tank -> Electrolyzer ->             #
#              Receive Tank -> [optional Centrifuge] -> Dryer -> Storage      #
#                                                                             #
#              The optional centrifuge is a splitter whose split fraction is  #
#              the binary y_cf (1: all flow to the centrifuge, 0: bypass).    #
#              Cake and bypass recombine in a mixer ahead of the dryer.       #
#              Every unit carries the full component stream (streamTools).   #
#                                                                             #
# Output:      - m with all parameters, variables, constraints, and objective #
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
from src.singleUnitModels.splitter import splitter
import src.singleUnitModels.prepTank as prepTank
import src.singleUnitModels.electrolyzer as electrolyzer
import src.singleUnitModels.receiveTank as receiveTank
import src.singleUnitModels.centrifuge as centrifuge
import src.singleUnitModels.dryer as dryer
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
    m.targetProductTSS = pyo.Param(initialize=0.25, mutable=True)   # 25 wt% solids

    # Centrifuge on/off binary and big-M (mass basis)
    m.y_cf = pyo.Var(within=pyo.Binary, initialize=1)
    mMass = float(pyo.value(m.totalFlowIn) * 1200.0 * 100.0)  # kg/s

    # ------------------------------ Units ------------------------------
    prepTank.prepTank(m)
    electrolyzer.electrolyzer(m)
    receiveTank.receiveTank(m)
    splitter(m, 'sp')                                # outlet1 -> centrifuge, outlet2 -> bypass
    centrifuge.centrifuge(m)
    mixer(m, 'mxDr', ['cake', 'bypass'], pHOut=7.0)  # both branches leave the receive tank at its target pH
    dryer.dryer(m)
    storageTank.storageTank(m)

    m.mxDr.pHOutSpec.set_value(pyo.value(m.rt.targetpH))
    m.pt.targetTSSConstr.deactivate()
    m.pt.waterMassFlowIn.fix(0.0)       # no dilution water
    m.cf.mGeo.set_value(10.0)            # tighter geometry big-M

    # --------------------------- Connections ---------------------------
    streamTools.connectStreams(m, 'sludgeToPt', m.sludgeFeed.outlet, m.pt.sludgeIn)
    streamTools.connectStreams(m, 'centrateToPt', m.centrateFeed.outlet, m.pt.centrateIn)
    streamTools.connectStreams(m, 'ptToEl', m.pt.outlet, m.el.inlet)
    streamTools.connectStreams(m, 'elToRt', m.el.outlet, m.rt.inlet)
    streamTools.connectStreams(m, 'rtToSplitter', m.rt.outlet, m.sp.inlet)
    streamTools.connectStreams(m, 'splitterToCf', m.sp.outlet1, m.cf.inlet)
    streamTools.connectStreams(m, 'cakeToMixer', m.cf.cake, m.mxDr.cake)
    streamTools.connectStreams(m, 'bypassToMixer', m.sp.outlet2, m.mxDr.bypass)
    streamTools.connectStreams(m, 'mixerToDryer', m.mxDr.outlet, m.dr.inlet)
    streamTools.connectStreams(m, 'dryerToStorage', m.dr.product, m.st.inlet)

    # --------------------- Optional centrifuge logic ---------------------
    m.splitIsBinary = pyo.Constraint(expr=m.sp.splitFrac == m.y_cf)

    m.vMinCf = pyo.Param(initialize=0.1, mutable=True)    # m3, minimum when ON
    m.nMinCf = pyo.Param(initialize=20.0, mutable=True)   # rps, minimum when ON
    m.vMaxCf = pyo.Param(initialize=2.0, mutable=True)    # m3
    m.dMaxCf = pyo.Param(initialize=1.0, mutable=True)    # m
    epsOff = 1e-6
    m.cfMinVolumeOn = pyo.Constraint(expr=m.cf.tankVolume >= m.vMinCf * m.y_cf)
    m.cfMinRpmOn = pyo.Constraint(expr=m.cf.agitRotation >= m.nMinCf * m.y_cf)
    m.cfCakeGate = pyo.Constraint(expr=m.cf.cake.totalMass <= mMass * m.y_cf)
    m.cfCentrateGate = pyo.Constraint(expr=m.cf.centrate.totalMass <= mMass * m.y_cf)
    m.cfMaxVolumeGate = pyo.Constraint(expr=m.cf.tankVolume <= m.vMaxCf * m.y_cf + epsOff * (1 - m.y_cf))
    m.cfMaxDiameterGate = pyo.Constraint(expr=m.cf.tankDiameter <= m.dMaxCf * m.y_cf + epsOff * (1 - m.y_cf))
    m.cfMaxRpsGate = pyo.Constraint(expr=m.cf.agitRotation <= 250.0 * m.y_cf + epsOff * (1 - m.y_cf))

    m.productTssTarget = pyo.Constraint(expr=m.dr.finalSolidsFrac == m.targetProductTSS)

    # ------------------------------ Product ------------------------------
    flowsheetTools.addProductQuality(m, m.st.outlet)

    # ------------------------------ Costs ------------------------------
    m.tonnesProductTotal = pyo.Expression(expr=m.dr.productMassFlow * 1e-3 * m.daysOperation)  # wet tonnes
    for blk in (m.pt, m.el, m.rt, m.dr, m.st):
        blk.capexPerTonne = pyo.Expression(expr=blk.capex / (m.tonnesProductTotal + 1e-6))
        blk.opexPerTonne = pyo.Expression(expr=blk.opex / (m.tonnesProductTotal + 1e-6))
    m.cf.capexPerTonne = pyo.Expression(expr=m.y_cf * m.cf.capex / (m.tonnesProductTotal + 1e-6))
    m.cf.opexPerTonne = pyo.Expression(expr=m.y_cf * m.cf.opex / (m.tonnesProductTotal + 1e-6))

    # 1.32 covers installation, engineering, and contingency
    m.capex = pyo.Expression(expr=(m.pt.capexPerTonne + m.el.capexPerTonne + m.rt.capexPerTonne
                                   + m.cf.capexPerTonne + m.dr.capexPerTonne + m.st.capexPerTonne) * 1.32)
    m.opex = pyo.Expression(expr=m.pt.opexPerTonne + m.el.opexPerTonne + m.rt.opexPerTonne
                            + m.cf.opexPerTonne + m.dr.opexPerTonne + m.st.opexPerTonne)
    m.totalCost = pyo.Objective(expr=m.capex + m.opex, sense=pyo.minimize)

    m.cf.agitRotation.set_value(60.0)
    m.cf.tankDiameter.set_value(0.8)
    m.cf.tankVolume.set_value(0.6)
    return m


def printResults(m):
    v = flowsheetTools.safeValue
    print('================ FEED / TARGETS ==================')
    print('Feed mass flow (kg/day):', v(m.feedMassFlow) * 86400)
    print('Target product TSS (fraction):', v(m.targetProductTSS))
    print('Centrifuge active (y_cf):', v(m.y_cf))
    print('================ PREP TANK =======================')
    print('CaO (kg/day):', v(m.pt.totalCaO) * 86400, ' tank / lime tank volume (m3):', v(m.pt.tankVolume), v(m.pt.limeTankVolume))
    flowsheetTools.printStream('Prep tank outlet', m.pt.outlet)
    print('================ ELECTROLYZER ====================')
    print('Area (m2):', v(m.el.area), ' power (kW):', v(m.el.power))
    print('Solid-bound N in (kg-N/day):', v(m.el.solidsNIn) * 86400, ' liberated (kg-N/day):', v(m.el.liberatedN) * 86400,
          ' of which TAN:', v(m.el.liberatedTan) * 86400)
    print('TAN volatilized/oxidized (kg-N/day):', v(m.el.nitrogenLost) * 86400)
    print('================ RECEIVE TANK ====================')
    print('H2SO4 (kg/day, pure):', v(m.rt.h2so4RequiredKgPerS) * 86400, ' acid solution (kg/day):', v(m.rt.acidMassFlowIn) * 86400)
    print('Acid eq (mol/day): OH-', v(m.rt.ohNeutralizationDemand) * 86400, ' TAN', v(m.rt.tanProtonationDemand) * 86400,
          ' buffer', v(m.rt.bufferingAcidDemand) * 86400)
    print('================ CENTRIFUGE ======================')
    flowsheetTools.printStream('Centrifuge cake', m.cf.cake)
    flowsheetTools.printStream('Centrifuge centrate (effluent)', m.cf.centrate)
    print('Capture fraction:', v(m.cf.solidMassCaptured), ' cake TSS:', v(m.cf.cakeTSS), ' CaO wt frac in cake:', v(m.cf.caoWeightFractionInCake))
    print('Volume (m3):', v(m.cf.tankVolume), ' diameter (m):', v(m.cf.tankDiameter), ' rotation (rps):', v(m.cf.agitRotation))
    print('================ DRYER ===========================')
    flowsheetTools.printStream('Dryer product', m.dr.product)
    print('Water evaporated (kg/day):', v(m.dr.waterVaporFlowOut) * 86400, ' heat duty (kW):', v(m.dr.heatDuty), ' blower (kW):', v(m.dr.blowerPower))
    print('Inlet pH:', v(m.dr.inlet.pH), ' free-NH3 fraction:', v(m.dr.freeAmmoniaFrac), ' NH3-N volatilized (kg-N/day):', v(m.dr.nitrogenVapor) * 86400)
    flowsheetTools.printProductQuality('E-GROW PRODUCT (FS1)', m.quality)
    flowsheetTools.printBalance(
        m, [m.st.outlet],
        nLosses=[('Electrolyzer TAN loss', m.el.nitrogenLost), ('Dryer exhaust NH3', m.dr.nitrogenVapor)],
        otherOutletStreams=[('Centrifuge centrate', m.cf.centrate)],
    )
    print('================ COSTS ===========================')
    for label, blk in (('Prep tank', m.pt), ('Electrolyzer', m.el), ('Receive tank', m.rt),
                       ('Centrifuge', m.cf), ('Dryer', m.dr), ('Storage tank', m.st)):
        print(f'{label} capex / opex ($/t): {v(1.32 * blk.capexPerTonne):.4f} / {v(blk.opexPerTonne):.4f}')
    print('Total Capex ($/t):', v(m.capex))
    print('Total Opex ($/t):', v(m.opex))
    print('Total Cost ($/t wet product):', v(m.capex + m.opex))
    print('Total Cost ($/t dry product):', v(m.capex + m.opex) / v(m.dr.finalSolidsFrac))
    print('================ ELECTRICITY / CHEMICALS =========')
    print('Electrolyzer (kWh/day):', v(m.el.power) * 24.0)
    print('Centrifuge shaft (kWh/day):', v(m.cf.powerKW) * 24.0)
    print('Dryer heat duty / blower (kWh/day):', v(m.dr.heatDuty) * 24.0, v(m.dr.blowerPower) * 24.0)
    print('CaO (kg/day):', v(m.pt.totalCaO) * 86400, '  acid solution (kg/day):', v(m.rt.acidMassFlowIn) * 86400)


if __name__ == '__main__':
    m = buildModel()
    tol = 1e-7
    solver = pyo.SolverFactory('baron', options={'EpsA': 100 * tol, 'AbsConFeasTol': 1 * tol, 'TDo': 0, 'MDo': 0, 'OBTTDo': 1},
                               executable='C:/baron/baron.exe')
    sol = solver.solve(m, tee=True)
    printResults(m)
