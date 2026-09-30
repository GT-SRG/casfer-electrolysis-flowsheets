#------------------------------------------------------------------------------
# function:    flowsheet4.py                                                  #
# Description: FS4 process flow -- the simplest variant.                      #
#              Feed -> Prep Tank -> Electrolyzer -> Receive Tank -> Storage   #
#              No separation step and no product target: the whole feed      #
#              (solids and liquid) reports to the product; product TSS and    #
#              N/P/K are outcomes, not constraints. Closer to a costed        #
#              simulation than an optimization, by design.                    #
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
import src.singleUnitModels.receiveTank as receiveTank
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
    receiveTank.receiveTank(model)
    storageTank.storageTank(model)

    streamTools.connectStreams(model, 'sludgeToPt', model.sludgeFeed.outlet, model.pt.sludgeIn)
    streamTools.connectStreams(model, 'centrateToPt', model.centrateFeed.outlet, model.pt.centrateIn)
    streamTools.connectStreams(model, 'ptToEl', model.pt.outlet, model.el.inlet)
    streamTools.connectStreams(model, 'elToRt', model.el.outlet, model.rt.inlet)
    streamTools.connectStreams(model, 'rtToSt', model.rt.outlet, model.st.inlet)

    flowsheetTools.addProductQuality(model, model.st.outlet)

    units = (model.pt, model.el, model.rt, model.st)
    model.totalCapex = pyo.Expression(expr=1.35 * sum(u.capex for u in units))
    model.totalOpex = pyo.Expression(expr=sum(u.opex for u in units))
    model.totalCost = pyo.Objective(expr=(model.totalCapex + model.totalOpex) / 1e6, sense=pyo.minimize)

    model.totalProductMassTonnes = pyo.Expression(expr=model.st.outlet.totalMass * model.daysOperation * 1e-3)
    model.totalCostDollars = pyo.Expression(expr=model.totalCapex + model.totalOpex)
    model.totalCostPerTonneWet = pyo.Expression(expr=model.totalCostDollars / (model.totalProductMassTonnes + 1e-9))
    model.totalCapexPerTonneWet = pyo.Expression(expr=model.totalCapex / (model.totalProductMassTonnes + 1e-9))
    model.totalOpexPerTonneWet = pyo.Expression(expr=model.totalOpex / (model.totalProductMassTonnes + 1e-9))
    return model


def printResults(model):
    v = flowsheetTools.safeValue
    print('================ FEED =====================')
    print('Feed mass flow (kg/day):', v(model.feedMassFlow) * 86400)
    print('================ PREP TANK / ELECTROLYZER / RECEIVE TANK ================')
    print('CaO (kg/day):', v(model.pt.totalCaO) * 86400, ' prep tank volume (m3):', v(model.pt.tankVolume))
    print('Electrolyzer area (m2):', v(model.el.area), ' power (kW):', v(model.el.power),
          ' liberated N (kg-N/day):', v(model.el.liberatedN) * 86400, ' TAN lost (kg-N/day):', v(model.el.nitrogenLost) * 86400)
    print('H2SO4 (kg/day, pure):', v(model.rt.h2so4RequiredKgPerS) * 86400, ' acid solution (kg/day):', v(model.rt.acidMassFlowIn) * 86400)
    print('Acid eq (mol/day): OH-', v(model.rt.ohNeutralizationDemand) * 86400, ' TAN', v(model.rt.tanProtonationDemand) * 86400,
          ' buffer', v(model.rt.bufferingAcidDemand) * 86400)
    flowsheetTools.printStream('E-GROW product', model.st.outlet)
    flowsheetTools.printProductQuality('E-GROW PRODUCT (FS4, no target enforced)', model.quality)
    flowsheetTools.printBalance(model, [model.st.outlet], nLosses=[('Electrolyzer TAN loss', model.el.nitrogenLost)])
    print('================ COST BREAKDOWN =======================')
    wetT = v(model.totalProductMassTonnes)
    for label, blk in (('Prep tank', model.pt), ('Electrolyzer', model.el), ('Receive tank', model.rt), ('Storage tank', model.st)):
        print(f'{label} capex / opex ($/wet t): {1.35 * v(blk.capex) / wetT:.4f} / {v(blk.opex) / wetT:.4f}')
    print('Total capex / opex ($/wet t):', v(model.totalCapexPerTonneWet), v(model.totalOpexPerTonneWet))
    print('Total cost ($/wet t product):', v(model.totalCostPerTonneWet))
    print('Total cost ($/dry t product):', v(model.totalCostDollars) / (v(model.st.outlet.solidsMass) * v(model.daysOperation) * 1e-3))
    print('Electrolyzer (kWh/day):', v(model.el.power) * 24.0)


if __name__ == '__main__':
    model = buildModel()
    tol = 1e-4
    solver = pyo.SolverFactory('baron', options={'EpsA': tol, 'AbsConFeasTol': tol, 'TDo': 0, 'MDo': 0, 'OBTTDo': 1},
                               executable='C:/baron/baron.exe')
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
        print('FS4 solve did not converge to a loadable solution.')
        print('Solver status:', solution.solver.status)
        print('Termination condition:', solution.solver.termination_condition)
