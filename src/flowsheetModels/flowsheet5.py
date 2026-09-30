#------------------------------------------------------------------------------
# function:    flowsheet5.py                                                  #
# Description: FS5 process flow.                                             #
#              Feed -> Prep Tank -> Electrolyzer -> Sparger (CO2 only) ->     #
#              Storage                                                        #
#              No ammonia gas stream: the sparger only bubbles CO2 to         #
#              neutralize the lime-derived base and reprotonate TAN.          #
#              The CO2 dose is set inside sparger.py by the outlet pH spec    #
#              (targetpH, default 8)                                          #
#                                                                             #
#              Every unit carries the full component stream (streamTools).   #
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
import src.singleUnitModels.sparger as sparger
import src.singleUnitModels.storageTank as storageTank


def buildModel():
    model = pyo.ConcreteModel()
    try:
        getParams.getParams(model)
    except Exception:
        pass
    getStandardParams.getStandardParams(model)

    flowsheetTools.addStandardFeeds(model, feedPH=7.5)

    prepTank.prepTank(model)
    electrolyzer.electrolyzer(model)
    sparger.sparger(model, 'sg', 'Combined Sparger')
    storageTank.storageTank(model)

    # CO2-only sparger: NH3 side off by construction
    model.sg.nh3CarrierGasMolFlowIn.fix(0.0)
    model.sg.nh3SpeciesFracIn.fix(0.0)
    model.sg.co2TransferFraction.set_value(1.0)

    # Mild sparger assumptions for this lower-solids stream
    model.sg.residenceTimeHr.set_value(0.25)
    model.sg.mixingPowerDensity.set_value(1.5)
    model.sg.gasInjectionPressureBar.set_value(1.1)
    model.sg.capexMultiplier.set_value(2.5)
    model.sg.costReference.set_value(5000.0)
    model.sg.volumeReference.set_value(10.0)
    model.sg.capexFactor.set_value(1.0)
    model.sg.blowerEff.set_value(0.7)

    streamTools.connectStreams(model, 'sludgeToPt', model.sludgeFeed.outlet, model.pt.sludgeIn)
    streamTools.connectStreams(model, 'centrateToPt', model.centrateFeed.outlet, model.pt.centrateIn)
    streamTools.connectStreams(model, 'ptToEl', model.pt.outlet, model.el.inlet)
    streamTools.connectStreams(model, 'elToSg', model.el.outlet, model.sg.inlet)
    streamTools.connectStreams(model, 'sgToSt', model.sg.outlet, model.st.inlet)

    flowsheetTools.addProductQuality(model, model.st.outlet)

    units = (model.pt, model.el, model.sg, model.st)
    model.totalCapex = pyo.Expression(expr=1.35 * sum(u.capex for u in units))
    model.totalOpex = pyo.Expression(expr=sum(u.opex for u in units))
    model.totalCost = pyo.Objective(expr=(model.totalCapex + model.totalOpex) / 1e6, sense=pyo.minimize)

    model.totalProductMassTonnes = pyo.Expression(expr=model.st.outlet.totalMass * model.daysOperation * 1e-3)
    model.totalCostDollars = pyo.Expression(expr=model.totalCapex + model.totalOpex)
    model.totalCostPerTonneWet = pyo.Expression(expr=model.totalCostDollars / (model.totalProductMassTonnes + 1e-9))
    model.totalCapexPerTonneWet = pyo.Expression(expr=model.totalCapex / (model.totalProductMassTonnes + 1e-9))
    model.totalOpexPerTonneWet = pyo.Expression(expr=model.totalOpex / (model.totalProductMassTonnes + 1e-9))

    # Initial point for the carbonate charge balance (at the pH spec)
    model.sg.outlet.pH.set_value(8.0)
    model.sg.hOut.set_value(1e-5)
    model.sg.ohOut.set_value(1e-3)
    return model


def printResults(model):
    v = flowsheetTools.safeValue
    print('================ FEED =====================')
    print('Feed mass flow (kg/day):', v(model.feedMassFlow) * 86400)
    print('================ PREP TANK / ELECTROLYZER ================')
    print('CaO (kg/day):', v(model.pt.totalCaO) * 86400, ' prep tank volume (m3):', v(model.pt.tankVolume))
    print('Electrolyzer area (m2):', v(model.el.area), ' power (kW):', v(model.el.power),
          ' liberated N (kg-N/day):', v(model.el.liberatedN) * 86400, ' TAN lost (kg-N/day):', v(model.el.nitrogenLost) * 86400)
    print('================ SPARGER (CO2 only) =====================')
    print('CO2 (mol/s):', v(model.sg.co2GasMolFlowIn), ' CO2 (kg/day):', v(model.sg.co2MassFlowIn) * 86400)
    print('NH3 transferred (mol/s, should be 0):', v(model.sg.nh3TransferredMolS))
    print('Spectator charge in (eq/s):', v(model.sg.spectatorMolPerS), ' CaCO3 precipitated (kg/day):', v(model.sg.caco3PrecipMolS) * 0.10009 * 86400,
          ' dissolved C (mol/L):', v(model.sg.carbonTotalConc))
    print('Product pH (spec):', v(model.sg.outlet.pH), ' free NH3 fraction of TAN:', v(model.sg.nh3FreeFracOut))
    print('Tank volume (m3):', v(model.sg.tankVolume), ' mixing / blower (kW):', v(model.sg.mixingPower), v(model.sg.blowerPower))
    flowsheetTools.printStream('E-GROW product', model.st.outlet)
    flowsheetTools.printProductQuality('E-GROW PRODUCT (FS5, no target enforced)', model.quality)
    flowsheetTools.printBalance(model, [model.st.outlet], nLosses=[('Electrolyzer TAN loss', model.el.nitrogenLost)])
    print('================ COST BREAKDOWN =======================')
    wetT = v(model.totalProductMassTonnes)
    for label, blk in (('Prep tank', model.pt), ('Electrolyzer', model.el), ('Sparger', model.sg), ('Storage tank', model.st)):
        print(f'{label} capex / opex ($/wet t): {1.35 * v(blk.capex) / wetT:.4f} / {v(blk.opex) / wetT:.4f}')
    print('Total capex / opex ($/wet t):', v(model.totalCapexPerTonneWet), v(model.totalOpexPerTonneWet))
    print('Total cost ($/wet t product):', v(model.totalCostPerTonneWet))
    print('Total cost ($/dry t product):', v(model.totalCostDollars) / (v(model.st.outlet.solidsMass) * v(model.daysOperation) * 1e-3))
    print('Electrolyzer (kWh/day):', v(model.el.power) * 24.0,
          ' sparger mixing + blower (kWh/day):', v(model.sg.mixingPower + model.sg.blowerPower) * 24.0)


if __name__ == '__main__':
    model = buildModel()
    tol = 1e-4
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
        print('FS5 solve did not converge to a loadable solution.')
        print('Solver status:', solution.solver.status)
        print('Termination condition:', solution.solver.termination_condition)