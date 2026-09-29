#------------------------------------------------------------------------------
# function:    flowsheet0.py                                                  #
# Description: Baseline optimization model for wastewater treatment process   #
#              Process flow: Feed (sludge + centrate) -> Mixer ->             #
#              Centrifuge -> Dryer -> Storage Tank                            #
#                                                                             #
#              Every unit carries the full component stream (streamTools):    #
#              this script only connects outlets to inlets, sets targets,     #
#              and assembles the cost objective.                              #
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
    m.targetProductTSS = pyo.Param(initialize=0.90, mutable=True)   # 90 wt% solids

    # Centrifuge always on in FS0
    m.y_cf = pyo.Param(initialize=1)

    # ------------------------------ Units ------------------------------
    mixer(m, 'mx', ['sludge', 'centrate'], pHOut=7.5)   # FS0 has no pH adjustment: dryer sees pH 7.5
    centrifuge.centrifuge(m)
    dryer.dryer(m)
    storageTank.storageTank(m)

    # Minimum / maximum practical centrifuge sizing
    m.vMinCf = pyo.Param(initialize=0.1, mutable=True)    # m3
    m.nMinCf = pyo.Param(initialize=20.0, mutable=True)   # rps
    m.vMaxCf = pyo.Param(initialize=2.0, mutable=True)    # m3
    m.cfMinVolume = pyo.Constraint(expr=m.cf.tankVolume >= m.vMinCf)
    m.cfMinRpm = pyo.Constraint(expr=m.cf.agitRotation >= m.nMinCf)
    m.cfMaxVolume = pyo.Constraint(expr=m.cf.tankVolume <= m.vMaxCf)

    # --------------------------- Connections ---------------------------
    streamTools.connectStreams(m, 'sludgeToMixer', m.sludgeFeed.outlet, m.mx.sludge)
    streamTools.connectStreams(m, 'centrateToMixer', m.centrateFeed.outlet, m.mx.centrate)
    streamTools.connectStreams(m, 'mixerToCentrifuge', m.mx.outlet, m.cf.inlet)
    streamTools.connectStreams(m, 'cakeToDryer', m.cf.cake, m.dr.inlet)
    streamTools.connectStreams(m, 'dryerToStorage', m.dr.product, m.st.inlet)

    m.productTssTarget = pyo.Constraint(expr=m.dr.finalSolidsFrac == m.targetProductTSS)

    # ------------------------------ Product ------------------------------
    flowsheetTools.addProductQuality(m, m.st.outlet)

    # ------------------------------ Costs ------------------------------
    m.tonnesProductTotal = pyo.Expression(expr=m.dr.productMassFlow * 1e-3 * m.daysOperation)  # wet tonnes, lifetime
    m.cf.capexPerTonne = pyo.Expression(expr=m.cf.capex / (m.tonnesProductTotal + 1e-6))
    m.cf.opexPerTonne = pyo.Expression(expr=m.cf.opex / (m.tonnesProductTotal + 1e-6))
    m.dr.capexPerTonne = pyo.Expression(expr=m.dr.capex / (m.tonnesProductTotal + 1e-6))
    m.dr.opexPerTonne = pyo.Expression(expr=m.dr.opex / (2 * m.tonnesProductTotal + 1e-6))
    m.st.capexPerTonne = pyo.Expression(expr=m.st.capex / (m.tonnesProductTotal + 1e-6))
    m.st.opexPerTonne = pyo.Expression(expr=m.st.opex / (m.tonnesProductTotal + 1e-6))

    # 1.32 covers installation, engineering, and contingency
    m.capex = pyo.Expression(expr=(m.cf.capexPerTonne + m.dr.capexPerTonne + m.st.capexPerTonne) * 1.32)
    m.opex = pyo.Expression(expr=m.cf.opexPerTonne + m.dr.opexPerTonne + m.st.opexPerTonne)
    m.totalCost = pyo.Objective(expr=m.capex + m.opex, sense=pyo.minimize)

    # Initial point for the centrifuge
    m.cf.agitRotation.set_value(60.0)
    m.cf.tankDiameter.set_value(0.8)
    m.cf.tankVolume.set_value(0.6)
    return m


def printResults(m):
    v = flowsheetTools.safeValue
    print('================ FEED / TARGETS ==================')
    print('Feed mass flow (kg/day):', v(m.feedMassFlow) * 86400)
    print('Feed dry solids (kg/day):', v(m.feedDrySolids) * 86400)
    print('Target product TSS (fraction):', v(m.targetProductTSS))
    print('================ CENTRIFUGE ======================')
    flowsheetTools.printStream('Centrifuge inlet', m.cf.inlet)
    flowsheetTools.printStream('Centrifuge cake', m.cf.cake)
    flowsheetTools.printStream('Centrifuge centrate (effluent)', m.cf.centrate)
    print('Capture fraction:', v(m.cf.solidMassCaptured), ' cake TSS:', v(m.cf.cakeTSS))
    print('Volume (m3):', v(m.cf.tankVolume), ' diameter (m):', v(m.cf.tankDiameter),
          ' rotation (rps):', v(m.cf.agitRotation), ' residence time (s):', v(m.cf.residenceTime))
    print('Shaft power (kW):', v(m.cf.powerKW), ' consolidation pressure (bar):', v(m.cf.consolidationPressure) / 1e5)
    print('================ DRYER ===========================')
    flowsheetTools.printStream('Dryer product', m.dr.product)
    print('Water evaporated (kg/day):', v(m.dr.waterVaporFlowOut) * 86400)
    print('Air flow (m3/s):', v(m.dr.airFlowIn), ' air T in/out (C):', v(m.dr.airTempIn), v(m.dr.airTempOut))
    print('Heat duty (kW):', v(m.dr.heatDuty), ' blower (kW):', v(m.dr.blowerPower))
    print('Free-NH3 fraction:', v(m.dr.freeAmmoniaFrac), ' stripping fraction:', v(m.dr.strippingFraction),
          ' NH3-N volatilized (kg-N/day):', v(m.dr.nitrogenVapor) * 86400)
    flowsheetTools.printProductQuality('E-GROW PRODUCT (FS0)', m.quality)
    flowsheetTools.printBalance(
        m, [m.st.outlet],
        nLosses=[('Dryer exhaust NH3', m.dr.nitrogenVapor)],
        otherOutletStreams=[('Centrifuge centrate', m.cf.centrate)],
    )
    print('================ COSTS ===========================')
    print('Centrifuge capex / opex ($):', v(m.cf.capex), v(m.cf.opex))
    print('Dryer capex / opex ($):', v(m.dr.capex), v(m.dr.opex))
    print('Storage tank capex / opex ($):', v(m.st.capex), v(m.st.opex))
    print('Centrifuge capex / opex ($/t):', v(1.32 * m.cf.capexPerTonne), v(m.cf.opexPerTonne))
    print('Dryer capex / opex ($/t):', v(1.32 * m.dr.capexPerTonne), v(m.dr.opexPerTonne))
    print('Storage tank capex / opex ($/t):', v(1.32 * m.st.capexPerTonne), v(m.st.opexPerTonne))
    print('Total Capex ($/t):', v(m.capex))
    print('Total Opex ($/t):', v(m.opex))
    print('Total Cost ($/t wet product):', v(m.capex + m.opex))
    print('Total Cost ($/t dry product):', v(m.capex + m.opex) / v(m.dr.finalSolidsFrac))
    print('================ ELECTRICITY =====================')
    print('Centrifuge shaft (kWh/day):', v(m.cf.powerKW) * 24.0)
    print('Dryer heat duty (kWh/day):', v(m.dr.heatDuty) * 24.0)
    print('Dryer blower (kWh/day):', v(m.dr.blowerPower) * 24.0)


if __name__ == '__main__':
    m = buildModel()
    tol = 1e-2
    solver = pyo.SolverFactory('baron', options={'EpsA': 100 * tol, 'AbsConFeasTol': 1 * tol, 'TDo': 0, 'MDo': 0, 'OBTTDo': 1},
                               executable='C:/baron/baron.exe')
    sol = solver.solve(m, tee=True)
    printResults(m)
