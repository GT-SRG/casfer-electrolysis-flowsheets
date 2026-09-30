#------------------------------------------------------------------------------
# function:    flowsheet3.py                                                  #
# Description: FS3 process flow with a single combined CO2 + NH3 sparger.     #
#              Feed -> Prep Tank -> Electrolyzer -> NH3 Stripper ->           #
#              GO Membrane Dewatering -> Combined Sparger -> E-GROW storage   #
#              Stripper carrier gas (sweep gas + NH3) feeds the sparger;      #
#              GO permeate is neutralized to pH 7 with H2SO4 and returned    #
#              to the WWTP.                                                   #
#                                                                             #
#              Every unit carries the full component stream (streamTools):   #
#              this script only connects outlets to inlets, sets targets,    #
#              and assembles the cost objective. The CO2 dose is computed    #
#              inside sparger.py (co2Dosing).                                 #
#------------------------------------------------------------------------------

import os
import sys

repoRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repoRoot not in sys.path:
    sys.path.insert(0, repoRoot)

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import src.singleUnitModels.getParams as getParams
import src.singleUnitModels.getStandardParams as getStandardParams
import src.singleUnitModels.streamTools as streamTools
import src.singleUnitModels.flowsheetTools as flowsheetTools
import src.singleUnitModels.prepTank as prepTank
import src.singleUnitModels.electrolyzer as electrolyzer
import src.singleUnitModels.nh3Stripper as nh3Stripper
import src.singleUnitModels.goMembraneDewatering as goMembraneDewatering
import src.singleUnitModels.sparger as sparger
import src.singleUnitModels.storageTank as storageTank
import src.singleUnitModels.receiveTank as receiveTank


def buildModel():
    model = pyo.ConcreteModel()
    try:
        getParams.getParams(model)
    except Exception:
        pass
    getStandardParams.getStandardParams(model)

    # ------------------------------ Feeds ------------------------------
    flowsheetTools.addStandardFeeds(model, feedPH=7.5)
    model.targetProductTss = pyo.Param(initialize=0.25, mutable=True)
    model.targetNh3StripFraction = pyo.Param(initialize=0.95, mutable=True)

    # ------------------------------ Units ------------------------------
    prepTank.prepTank(model)
    electrolyzer.electrolyzer(model)
    nh3Stripper.nh3Stripper(model, 'nst')
    goMembraneDewatering.goMembraneDewatering(model)
    sparger.sparger(model, 'sg', 'Combined Sparger')
    storageTank.storageTank(model)
    receiveTank.receiveTank(model, 'nt')    # GO permeate neutralization to pH 7 before return to the WWTP
    model.nt.ohFromInletPH.set_value(1.0)

    model.pt.waterMassFlowIn.fix(0.0)
    model.go.targetSolidsConstraint.deactivate()   # product TSS target is set on the sparger outlet below

    # Combined sparger: full transfer of both gases, high-solids assumptions
    model.sg.co2TransferFraction.set_value(1.0)
    model.sg.nh3TransferFraction.set_value(1.0)
    model.sg.residenceTimeHr.set_value(0.25)
    model.sg.mixingPowerDensity.set_value(10.0)
    model.sg.gasInjectionPressureBar.set_value(4.0)
    model.sg.capexMultiplier.set_value(2.5)
    model.sg.costReference.set_value(5000.0)
    model.sg.volumeReference.set_value(10.0)
    model.sg.capexFactor.set_value(1.0)
    model.sg.blowerEff.set_value(0.7)

    # Standalone NH3 stripper energetics
    model.nst.mixingPowerDensity.set_value(1.5)
    model.nst.gasInjectionPressureBar.set_value(1.1)
    model.nst.blowerEff.set_value(0.7)

    # --------------------------- Connections ---------------------------
    streamTools.connectStreams(model, 'sludgeToPt', model.sludgeFeed.outlet, model.pt.sludgeIn)
    streamTools.connectStreams(model, 'centrateToPt', model.centrateFeed.outlet, model.pt.centrateIn)
    streamTools.connectStreams(model, 'ptToEl', model.pt.outlet, model.el.inlet)
    streamTools.connectStreams(model, 'elToNst', model.el.outlet, model.nst.inlet)
    streamTools.connectStreams(model, 'nstToGo', model.nst.outlet, model.go.inlet)
    streamTools.connectStreams(model, 'goToSg', model.go.retentate, model.sg.inlet)
    streamTools.connectStreams(model, 'sgToSt', model.sg.outlet, model.st.inlet)
    streamTools.connectStreams(model, 'goPermeateToNt', model.go.permeate, model.nt.inlet)

    # Gas side: stripper carrier gas (sweep gas + stripped NH3) -> sparger
    model.nh3CarrierGasToSg = pyo.Constraint(expr=model.sg.nh3CarrierGasMolFlowIn == model.nst.gasMolFlowOut)
    model.nh3SpeciesFracToSg = pyo.Constraint(expr=model.sg.nh3SpeciesFracIn == model.nst.nh3SpeciesFracOut)

    # ------------------------------ Targets ------------------------------
    model.productTssTarget = pyo.Constraint(
        expr=model.sg.outlet.solidsMass == model.targetProductTss * model.sg.outlet.totalMass
    )
    model.nh3StrippedFractionTarget = pyo.Constraint(expr=model.nst.nh3StrippedFraction >= model.targetNh3StripFraction)

    # ------------------------------ Product ------------------------------
    flowsheetTools.addProductQuality(model, model.st.outlet)
    model.nh3UnabsorbedN = pyo.Expression(
        expr=(model.nst.nh3StrippedMolFlow - model.sg.nh3TransferredMolS) * streamTools.molwtN
    )  # kg-N/s leaving with the sparger off-gas

    # ------------------------------ Costs ------------------------------
    units = (model.pt, model.el, model.nst, model.go, model.sg, model.st, model.nt)
    model.totalCapex = pyo.Expression(expr=1.35 * sum(u.capex for u in units))
    model.totalOpex = pyo.Expression(expr=sum(u.opex for u in units))
    model.totalCost = pyo.Objective(expr=(model.totalCapex + model.totalOpex) / 1e6, sense=pyo.minimize)

    model.totalProductMassTonnes = pyo.Expression(expr=model.st.outlet.totalMass * model.daysOperation * 1e-3)
    model.totalCostDollars = pyo.Expression(expr=model.totalCapex + model.totalOpex)
    model.totalCostPerTonneWet = pyo.Expression(expr=model.totalCostDollars / (model.totalProductMassTonnes + 1e-9))
    model.totalCapexPerTonneWet = pyo.Expression(expr=model.totalCapex / (model.totalProductMassTonnes + 1e-9))
    model.totalOpexPerTonneWet = pyo.Expression(expr=model.totalOpex / (model.totalProductMassTonnes + 1e-9))

    # Initial point for the stripper (Kremser needs S away from 1)
    model.nst.sweepGasMolFlowIn.set_value(300.0)
    model.nst.numberOfStages.set_value(5.0)
    return model


def printResults(model):
    v = flowsheetTools.safeValue
    print('================ FEED / TARGETS =====================')
    print('Feed mass flow (kg/day):', v(model.feedMassFlow) * 86400)
    print('Target product TSS:', v(model.targetProductTss), '  target NH3 strip fraction:', v(model.targetNh3StripFraction))
    print('================ PREP TANK / ELECTROLYZER ============')
    print('CaO (kg/day):', v(model.pt.totalCaO) * 86400, ' prep tank volume (m3):', v(model.pt.tankVolume))
    print('Electrolyzer area (m2):', v(model.el.area), ' power (kW):', v(model.el.power),
          ' liberated N (kg-N/day):', v(model.el.liberatedN) * 86400, ' TAN lost (kg-N/day):', v(model.el.nitrogenLost) * 86400)
    print('================ NH3 STRIPPER =========================')
    print('TAN in / out (kg-N/m3):', v(model.nst.concIn), v(model.nst.concOut), ' pH in / out:', v(model.nst.inlet.pH), v(model.nst.outlet.pH))
    print('Stages:', v(model.nst.numberOfStages), ' stripping factor:', v(model.nst.strippingFactor),
          ' stripped fraction:', v(model.nst.nh3StrippedFraction))
    print('Sweep gas (mol/s):', v(model.nst.sweepGasMolFlowIn), ' NH3 stripped (mol/s):', v(model.nst.nh3StrippedMolFlow),
          ' carrier NH3 mole fraction:', v(model.nst.nh3SpeciesFracOut))
    print('Tank volume (m3):', v(model.nst.tankVolume), ' mixing / blower (kW):', v(model.nst.mixingPower), v(model.nst.blowerPower))
    print('Charge balance residual (should be ~0):', v(model.nst.zNetOut + model.nst.hOut + model.nst.nh4Out - model.nst.ohOut))
    print('================ GO MEMBRANE DEWATERING ==============')
    flowsheetTools.printStream('GO retentate', model.go.retentate)
    flowsheetTools.printStream('GO permeate (effluent)', model.go.permeate)
    print('Area (m2):', v(model.go.area), ' deltaP (bar):', v(model.go.deltaP), ' delta-pi (bar):', v(model.go.deltaPi),
          ' pump (kW):', v(model.go.pumpPower))
    print('================ COMBINED SPARGER =====================')
    print('CO2 in (mol/s):', v(model.sg.co2GasMolFlowIn), ' CO2 (kg/day):', v(model.sg.co2MassFlowIn) * 86400,
          ' NH3 transferred (mol/s):', v(model.sg.nh3TransferredMolS))
    print('Product pH (live, TAN/carbonate system only):', v(model.sg.outlet.pH))
    print('Tank volume (m3):', v(model.sg.tankVolume), ' mixing / blower (kW):', v(model.sg.mixingPower), v(model.sg.blowerPower))
    print('================ PERMEATE NEUTRALIZATION ==============')
    print('H2SO4 (kg/day, pure):', v(model.nt.h2so4RequiredKgPerS) * 86400, ' inlet pH:', v(model.nt.inlet.pH),
          ' acid eq (mol/day): OH-', v(model.nt.ohNeutralizationDemand) * 86400, ' TAN', v(model.nt.tanProtonationDemand) * 86400,
          ' buffer', v(model.nt.bufferingAcidDemand) * 86400)
    flowsheetTools.printStream('Neutralized GO permeate to WWTP', model.nt.outlet)
    flowsheetTools.printStream('E-GROW product', model.st.outlet)
    flowsheetTools.printProductQuality('E-GROW PRODUCT (FS3)', model.quality)
    flowsheetTools.printBalance(
        model, [model.st.outlet],
        nLosses=[('Electrolyzer TAN loss', model.el.nitrogenLost), ('NH3 not re-absorbed in sparger', model.nh3UnabsorbedN)],
        otherOutletStreams=[('Neutralized GO permeate to WWTP', model.nt.outlet)],
    )
    print('================ COST BREAKDOWN =======================')
    wetT = v(model.totalProductMassTonnes)
    for label, blk in (('Prep tank', model.pt), ('Electrolyzer', model.el), ('NH3 stripper', model.nst),
                       ('GO membrane', model.go), ('Combined sparger', model.sg), ('Storage tank', model.st), ('Permeate neutralization', model.nt)):
        print(f'{label} capex / opex ($/wet t): {1.35 * v(blk.capex) / wetT:.4f} / {v(blk.opex) / wetT:.4f}')
    print('Total capex / opex ($/wet t):', v(model.totalCapexPerTonneWet), v(model.totalOpexPerTonneWet))
    print('Total cost ($/wet t product):', v(model.totalCostPerTonneWet))
    print('Total cost ($/dry t product):', v(model.totalCostDollars) / (v(model.st.outlet.solidsMass) * v(model.daysOperation) * 1e-3))
    print('================ ELECTRICITY / CHEMICALS ==============')
    print('Electrolyzer (kWh/day):', v(model.el.power) * 24.0)
    print('Stripper mixing + blower (kWh/day):', v(model.nst.mixingPower + model.nst.blowerPower) * 24.0)
    print('GO pump (kWh/day):', v(model.go.pumpPower) * 24.0)
    print('Sparger mixing + blower (kWh/day):', v(model.sg.mixingPower + model.sg.blowerPower) * 24.0)
    print('CaO (kg/day):', v(model.pt.totalCaO) * 86400, ' CO2 (kg/day):', v(model.sg.co2MassFlowIn) * 86400)


if __name__ == '__main__':
    model = buildModel()
    tol = 1e-5
    solver = pyo.SolverFactory('ipopt', options={'tol': tol, 'constr_viol_tol': tol})
    solution = solver.solve(model, tee=True, load_solutions=False)
    isGood = (
        solution.solver.status == SolverStatus.ok
        and solution.solver.termination_condition in (
            TerminationCondition.optimal, TerminationCondition.locallyOptimal, TerminationCondition.feasible)
    )
    if isGood:
        model.solutions.load_from(solution)
        printResults(model)
    else:
        print('FS3 solve did not converge to a loadable solution.')
        print('Solver status:', solution.solver.status)
        print('Termination condition:', solution.solver.termination_condition)