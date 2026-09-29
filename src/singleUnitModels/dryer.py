#------------------------------------------------------------------------------
# function:     dryer.py                                                      #
# Description:  Convective dryer with ammonia volatilization.                 #
#                                                                             #
#               Streams: inlet -> product                                     #
#                                                                             #
#               - all solids and solid-bound N/P/K report to the product      #
#               - water evaporates down to finalSolidsFrac                    #
#               - only TAN can volatilize: free-NH3 fraction at the inlet pH  #
#                 (T-dependent pKa) times a single-stage stripping fraction   #
#               - orgN, K, P, Ca, Mg are non-volatile and stay in the product #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m.dr                                                        #
#------------------------------------------------------------------------------

import math
import pyomo.environ as pyo
try:
    from . import getParams
    from . import streamTools
except ImportError:
    import getParams
    import streamTools

_LN10 = math.log(10.0)


def dryer(m):

    m.dr = pyo.Block()
    blk = m.dr

    dryerParams = getParams.params['Dryer']

    blk.costReference    = pyo.Param(initialize=dryerParams['Cost Reference'])                 # $
    blk.dutyReference    = pyo.Param(initialize=dryerParams['Duty Reference'])                 # kg-water/s
    blk.capexFactor      = pyo.Param(initialize=dryerParams['Capex Factor'])
    blk.blowerEfficiency = pyo.Param(initialize=dryerParams['Blower Efficiency'], default=0.7)
    blk.latentHeat       = pyo.Param(initialize=dryerParams['Latent Heat of Vaporization'])    # kWh/kg-water
    blk.airDensity       = pyo.Param(initialize=dryerParams['Air Density'], default=1.225)     # kg/m3
    blk.airSpecificHeat  = pyo.Param(initialize=dryerParams['Air Specific Heat'])              # kWh/kg-K
    blk.heatLossFactor   = pyo.Param(initialize=dryerParams['Heat Loss Factor'], default=0.15)
    blk.ambientTemp      = pyo.Param(initialize=dryerParams['Ambient Temperature'] - 273.15, default=25.0)  # C
    blk.blowerSEC        = pyo.Param(initialize=dryerParams.get('Blower Specific Energy', 5000.0), mutable=True)  # J/m3
    blk.airTempIn        = pyo.Param(initialize=dryerParams.get('Air Inlet Temperature C', 180.0), mutable=True)  # C

    # -------------------- Streams --------------------
    streamTools.addStream(blk, 'inlet')
    streamTools.addStream(blk, 'product')
    sIn, prod = blk.inlet, blk.product

    # -------------------- Decision / state variables --------------------
    blk.airTempOut        = pyo.Var(initialize=80.0, within=pyo.NonNegativeReals, bounds=(35.0, 95.0))  # C
    blk.airFlowIn         = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals)                         # m3/s
    blk.finalSolidsFrac   = pyo.Var(initialize=0.9, within=pyo.NonNegativeReals, bounds=(0, 1))
    blk.waterVaporFlowOut = pyo.Var(initialize=0.7, within=pyo.NonNegativeReals)                         # kg/s
    blk.heatDuty          = pyo.Var(initialize=1000.0, within=pyo.NonNegativeReals)                      # kW
    blk.blowerPower       = pyo.Var(initialize=50.0, within=pyo.NonNegativeReals)                        # kW
    blk.capex             = pyo.Var(initialize=1e6, within=pyo.NonNegativeReals)                         # $
    blk.opex              = pyo.Var(initialize=1e5, within=pyo.NonNegativeReals)                         # $

    blk.solidsFlow = pyo.Expression(expr=sIn.solidsMass)              # kg/s
    blk.waterMassFlowIn = pyo.Expression(expr=sIn.flow['liquid'])     # kg/s
    blk.productMassFlow = pyo.Expression(expr=prod.totalMass)         # kg/s
    blk.waterMassFlowOut = pyo.Expression(expr=prod.flow['liquid'])   # kg/s
    blk.sludgeTSSIn = pyo.Expression(expr=sIn.tss)                    # reporting
    blk.caoWeightFractionInProduct = pyo.Expression(expr=prod.flow['caoSolids'] / (prod.solidsMass + 1e-9))

    # -------------------- Solids and water --------------------
    streamTools.passComponents(blk, 'solidsToProduct', sIn, prod, streamTools.solidsFollowing)
    blk.productTSSDef = pyo.Constraint(expr=prod.solidsMass == blk.finalSolidsFrac * prod.totalMass)
    blk.waterEvaporation = pyo.Constraint(expr=blk.waterVaporFlowOut == sIn.flow['liquid'] - prod.flow['liquid'])

    # -------------------- Ammonia volatilization (TAN only) --------------------
    blk.airTempOutK = pyo.Expression(expr=blk.airTempOut + 273.15)
    blk.pKaNH3A = pyo.Param(initialize=0.09018, mutable=True)
    blk.pKaNH3B = pyo.Param(initialize=2729.92, mutable=True)   # K
    blk.pKaNH3T = pyo.Expression(expr=blk.pKaNH3A + blk.pKaNH3B / blk.airTempOutK)
    blk.molarMassN   = pyo.Param(initialize=0.014007, mutable=True)
    blk.molarMassNH3 = pyo.Param(initialize=0.017031, mutable=True)
    blk.molarMassAir = pyo.Param(initialize=0.028970, mutable=True)
    blk.molarMassWater = pyo.Param(initialize=0.018015, mutable=True)

    blk.henryNH3Ref = pyo.Param(initialize=58.5, mutable=True)          # mol/L/atm at the reference T
    blk.henryNH3RefTempK = pyo.Param(initialize=298.15, mutable=True)
    blk.enthalpySolutionNH3 = pyo.Param(initialize=-34200.0, mutable=True)  # J/mol
    blk.gasConstant = pyo.Param(initialize=8.314, mutable=True)
    blk.pressureTotalAtm = pyo.Param(initialize=1.0, mutable=True)
    blk.latentHeatNH3 = pyo.Param(initialize=0.381, mutable=True)        # kWh/kg NH3

    blk.henryNH3T = pyo.Expression(
        expr=blk.henryNH3Ref * pyo.exp(
            (-blk.enthalpySolutionNH3 / blk.gasConstant) * (1.0 / blk.airTempOutK - 1.0 / blk.henryNH3RefTempK)
        )
    )
    blk.freeAmmoniaFrac = pyo.Expression(expr=1.0 / (1.0 + 10 ** (blk.pKaNH3T - sIn.pH)))
    blk.nitrogenAvailable = pyo.Expression(expr=blk.freeAmmoniaFrac * sIn.flow['tan'])   # kg-N/s

    blk.airMolarFlow = pyo.Expression(expr=(blk.airFlowIn * blk.airDensity) / blk.molarMassAir)       # mol/s
    blk.waterVaporMolarFlow = pyo.Expression(expr=blk.waterVaporFlowOut / blk.molarMassWater)          # mol/s
    blk.gasMolarFlowTotal = pyo.Expression(expr=blk.airMolarFlow + blk.waterVaporMolarFlow)            # mol/s
    blk.liquidVolFlowInLps = pyo.Expression(expr=sIn.liquidVol * 1000.0)                               # L/s

    blk.strippingFactor = pyo.Expression(
        expr=blk.gasMolarFlowTotal / (blk.henryNH3T * blk.pressureTotalAtm * blk.liquidVolFlowInLps + 1e-9)
    )
    blk.strippingFraction = pyo.Expression(expr=blk.strippingFactor / (1.0 + blk.strippingFactor))
    blk.nitrogenVapor = pyo.Expression(expr=blk.strippingFraction * blk.nitrogenAvailable)            # kg-N/s
    blk.ammoniaVapor = pyo.Expression(expr=blk.nitrogenVapor * (blk.molarMassNH3 / blk.molarMassN))   # kg/s

    blk.tanBalance = pyo.Constraint(expr=prod.flow['tan'] == sIn.flow['tan'] - blk.nitrogenVapor)
    streamTools.passComponents(blk, 'nonVolatileBalance', sIn, prod, ['orgN', 'liqP', 'liqK', 'ca', 'mg'])
    blk.pHConstr = pyo.Constraint(expr=prod.pH == sIn.pH)

    # -------------------- Psychrometric limit --------------------
    blk.antoineA = pyo.Param(initialize=8.07131, mutable=True)
    blk.antoineB = pyo.Param(initialize=1730.63, mutable=True)
    blk.antoineC = pyo.Param(initialize=233.426, mutable=True)
    blk.humiditySafetyFactor = pyo.Param(initialize=0.80, mutable=True)
    blk.ambientHumidityRatio = pyo.Param(initialize=0.010, mutable=True)   # kg water / kg dry air

    blk.satVaporPressureMmHg = pyo.Expression(
        expr=pyo.exp(_LN10 * (blk.antoineA - blk.antoineB / (blk.antoineC + blk.airTempOut)))
    )
    blk.satVaporPressureAtm = pyo.Expression(expr=blk.satVaporPressureMmHg / 760.0)
    blk.humidityRatioSat = pyo.Expression(
        expr=0.622 * blk.satVaporPressureAtm / (blk.pressureTotalAtm - blk.satVaporPressureAtm + 1e-9)
    )
    blk.airMassFlowDry = pyo.Expression(expr=blk.airFlowIn * blk.airDensity)                          # kg/s
    blk.totalVolatileMassOut = pyo.Expression(expr=blk.waterVaporFlowOut + blk.ammoniaVapor)          # kg/s
    blk.humiditySaturationConstr = pyo.Constraint(
        expr=blk.ambientHumidityRatio * blk.airMassFlowDry + blk.totalVolatileMassOut
        <= blk.humiditySafetyFactor * blk.humidityRatioSat * blk.airMassFlowDry
    )

    # -------------------- Energy and costs --------------------
    blk.capexConstr = pyo.Constraint(
        expr=blk.capex == blk.costReference * (blk.waterVaporFlowOut / (blk.dutyReference + 1e-9)) ** blk.capexFactor
    )
    blk.heatDutyConstr = pyo.Constraint(
        expr=blk.heatDuty == blk.airSpecificHeat * blk.airDensity * blk.airFlowIn
        * (blk.airTempIn - blk.ambientTemp) * 3600.0 / ((1.0 - blk.heatLossFactor) + 1e-9)
    )
    blk.airEnergyBalance = pyo.Constraint(
        expr=blk.airSpecificHeat * blk.airDensity * blk.airFlowIn * (blk.airTempIn - blk.airTempOut) * 3600.0
        == (blk.latentHeat * blk.waterVaporFlowOut + blk.latentHeatNH3 * blk.ammoniaVapor) * 3600.0
    )
    blk.blowerPowerConstr = pyo.Constraint(expr=blk.blowerPower == blk.airFlowIn * blk.blowerSEC / 1000.0)
    blk.opexConstr = pyo.Constraint(
        expr=blk.opex == (blk.blowerPower + blk.heatDuty) * (m.daysOperation / 3600.0 + 1e-9) * m.elecPrice
    )

    return blk
