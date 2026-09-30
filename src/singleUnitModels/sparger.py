#------------------------------------------------------------------------------
# function:     sparger.py                                                    #
# Description:  Combined CO2 + NH3 absorption sparger.                        #
#                                                                             #
#               Streams: inlet -> outlet (liquid side)                        #
#               Gas side: CO2 feed, NH3 carrier gas (from the stripper)       #
#                                                                             #
#                                                                             #
# Input:        - model : Pyomo concrete model                                #
#               - blockName : block name                                      #
#               - paramsKey : params.xlsx unit name                           #
#                                                                             #
# Output:       - model.<blockName>                                           #
#------------------------------------------------------------------------------

import math
import pyomo.environ as pyo
try:
    from . import getParams
    from . import streamTools
except ImportError:
    import getParams
    import streamTools


def _getFiniteParam(params, key, defaultValue):
    value = params.get(key, defaultValue)
    try:
        valueFloat = float(value)
    except Exception:
        return defaultValue
    if not math.isfinite(valueFloat):
        return defaultValue
    return valueFloat


def sparger(model, blockName, paramsKey='Combined Sparger'):

    blk = pyo.Block()
    model.add_component(blockName, blk)

    params = getParams.params.get(paramsKey, {})

    blk.residenceTimeHr = pyo.Param(initialize=_getFiniteParam(params, 'Residence Time', 0.25), mutable=True)
    blk.mixingPowerDensity = pyo.Param(initialize=_getFiniteParam(params, 'Mixing Power Density', 10.0), mutable=True)
    blk.gasInjectionPressureBar = pyo.Param(initialize=_getFiniteParam(params, 'Gas Injection Pressure', 4.0), mutable=True)
    blk.capexMultiplier = pyo.Param(initialize=_getFiniteParam(params, 'Capex Multiplier', 2.5), mutable=True)
    blk.costReference = pyo.Param(initialize=_getFiniteParam(params, 'Cost Reference', 5000.0), mutable=True)
    blk.volumeReference = pyo.Param(initialize=_getFiniteParam(params, 'Volume Reference', 10.0), mutable=True)
    blk.capexFactor = pyo.Param(initialize=_getFiniteParam(params, 'Capex Factor', 1.0), mutable=True)
    blk.blowerEff = pyo.Param(initialize=_getFiniteParam(params, 'Blower Efficiency', 0.7), mutable=True)

    blk.gasTempK = pyo.Param(initialize=_getFiniteParam(params, 'Gas Temperature', 298.15), mutable=True)
    blk.co2HenryConst = pyo.Param(initialize=_getFiniteParam(params, 'CO2 Henry Constant', 0.033), mutable=True)
    blk.pKaNH4 = pyo.Param(initialize=_getFiniteParam(params, 'NH4 pKa', 9.25), mutable=True)
    blk.useAmmoniaSpeciation = pyo.Param(initialize=1.0, mutable=True)
    blk.co2TransferFraction = pyo.Param(initialize=1.0, mutable=True)
    blk.nh3TransferFraction = pyo.Param(initialize=1.0, mutable=True)
    blk.targetpH = pyo.Param(initialize=8.0, mutable=True)                      # outlet pH spec
    blk.spectatorPrecipFrac = pyo.Param(initialize=1.0, mutable=True)           # fraction of Z leaving as CaCO3
    blk.alkalinityBufferMolPerM3 = pyo.Param(initialize=20.0, mutable=True)     # mol/m3, same as receiving tank

    blk.R = pyo.Param(initialize=8.314462618)
    blk.barToPa = pyo.Param(initialize=1e5)
    blk.barToAtm = pyo.Param(initialize=0.986923)
    blk.molwtN = pyo.Param(initialize=streamTools.molwtN)          # kg/mol

    blk.minCapex = pyo.Param(initialize=30000.0, mutable=True)
    blk.co2CostPerKg = pyo.Param(initialize=0.60, mutable=True)    # $/kg CO2 (upper-bound proxy)
    blk.molwtCO2 = pyo.Param(initialize=0.04401, mutable=True)     # kg/mol

    # -------------------- Streams --------------------
    streamTools.addStream(blk, 'inlet', initPH=13.0)
    streamTools.addStream(blk, 'outlet', initPH=8.0)
    sIn, out = blk.inlet, blk.outlet

    blk.co2GasMolFlowIn = pyo.Var(initialize=0.01, within=pyo.NonNegativeReals)
    blk.co2GasMolFlowOut = pyo.Var(initialize=0.01, within=pyo.NonNegativeReals)
    blk.nh3CarrierGasMolFlowIn = pyo.Var(initialize=0.01, within=pyo.NonNegativeReals)
    blk.nh3CarrierGasMolFlowOut = pyo.Var(initialize=0.01, within=pyo.NonNegativeReals)
    blk.nh3SpeciesFracIn = pyo.Var(initialize=0.001, within=pyo.NonNegativeReals, bounds=(0, 1))
    blk.co2TransferredMolS = pyo.Var(initialize=0.0, within=pyo.NonNegativeReals)
    blk.nh3TransferredMolS = pyo.Var(initialize=0.0, within=pyo.NonNegativeReals)
    blk.tankVolume = pyo.Var(initialize=10.0, within=pyo.NonNegativeReals)

    blk.liquidVolFlowIn = pyo.Expression(expr=sIn.liquidVol)     # m3/s
    blk.liquidVolFlowOut = pyo.Expression(expr=out.liquidVol)    # m3/s
    blk.bulkVolFlowIn = pyo.Expression(expr=sIn.bulkVol)         # m3/s, vessel sizing

    # -------------------- Gas transfer --------------------
    blk.co2TransferBalance = pyo.Constraint(
        expr=blk.co2TransferredMolS == blk.co2TransferFraction * blk.co2GasMolFlowIn
    )
    blk.nh3TransferBalance = pyo.Constraint(
        expr=blk.nh3TransferredMolS == blk.nh3TransferFraction * blk.nh3CarrierGasMolFlowIn * blk.nh3SpeciesFracIn
    )
    blk.co2GasBalance = pyo.Constraint(expr=blk.co2GasMolFlowOut == blk.co2GasMolFlowIn - blk.co2TransferredMolS)
    blk.nh3GasBalance = pyo.Constraint(expr=blk.nh3CarrierGasMolFlowOut == blk.nh3CarrierGasMolFlowIn - blk.nh3TransferredMolS)
    blk.nh3SpeciesFracOut = pyo.Expression(
        expr=(blk.nh3CarrierGasMolFlowIn * blk.nh3SpeciesFracIn - blk.nh3TransferredMolS)
        / (blk.nh3CarrierGasMolFlowOut + 1e-12)
    )

    # -------------------- Liquid-side component balances --------------------
    blk.tanBalance = pyo.Constraint(expr=out.flow['tan'] == sIn.flow['tan'] + blk.nh3TransferredMolS * blk.molwtN)
    streamTools.passComponents(
        blk, 'passBalance', sIn, out,
        ['liquid', 'orgSolids', 'caoSolids', 'solidN', 'solidP', 'solidK', 'orgN', 'liqP', 'liqK', 'ca', 'mg']
    )

    # -------------------- Carbonate / ammonium chemistry --------------------
    blk.ka1 = pyo.Param(initialize=10 ** (-6.35), mutable=True)
    blk.ka2 = pyo.Param(initialize=10 ** (-10.33), mutable=True)
    blk.kaNH4 = pyo.Expression(expr=10 ** (-blk.pKaNH4))

    # Inlet spectator charge (mol/L): strong-base excess left by lime conditioning
    blk.hInReal = pyo.Expression(expr=10 ** (-sIn.pH))                     # mol/L
    blk.ohInReal = pyo.Expression(expr=10 ** (sIn.pH - 14.0))              # mol/L
    blk.tanInConc = pyo.Expression(expr=sIn.flow['tan'] / blk.molwtN / (blk.liquidVolFlowIn * 1000.0 + 1e-12))   # mol/L
    blk.nh4InConc = pyo.Expression(
        expr=blk.useAmmoniaSpeciation * blk.tanInConc * blk.hInReal / (blk.hInReal + blk.kaNH4 + 1e-30)
    )
    blk.spectatorChargeIn = pyo.Expression(expr=blk.ohInReal - blk.hInReal - blk.nh4InConc)   # mol/L
    blk.spectatorMolPerS = pyo.Expression(expr=blk.spectatorChargeIn * blk.liquidVolFlowIn * 1000.0)   # eq/s

    # CaCO3 precipitation of the lime-derived spectator (Ca2+ + CO3 2- -> CaCO3)
    blk.caco3PrecipMolS = pyo.Expression(expr=blk.spectatorPrecipFrac * blk.spectatorMolPerS / 2.0)   # mol/s
    blk.dissolvedSpectatorConc = pyo.Expression(
        expr=(1.0 - blk.spectatorPrecipFrac) * blk.spectatorMolPerS / (blk.liquidVolFlowOut * 1000.0 + 1e-12)
        + blk.alkalinityBufferMolPerM3 / 1000.0
    )  # mol/L

    blk.dissolvedCarbonMolS = pyo.Expression(expr=blk.co2TransferredMolS - blk.caco3PrecipMolS)   # mol/s
    blk.dissolvedCarbonNonNegative = pyo.Constraint(expr=blk.dissolvedCarbonMolS >= 0.0)
    blk.carbonTotalConc = pyo.Expression(expr=blk.dissolvedCarbonMolS / (blk.liquidVolFlowOut * 1000.0 + 1e-12))  # mol/L
    blk.ammoniaTotalConc = pyo.Expression(
        expr=out.flow['tan'] / blk.molwtN / (blk.liquidVolFlowOut * 1000.0 + 1e-12)
    )  # mol/L, TAN only

    # -------------------- Outlet pH --------------------
    blk.hOut = pyo.Var(initialize=1e-5, bounds=(1e-8, 1e6), within=pyo.NonNegativeReals)    # mol/m3
    blk.ohOut = pyo.Var(initialize=1e-3, bounds=(1e-8, 1e10), within=pyo.NonNegativeReals)  # mol/m3
    blk.phDef = pyo.Constraint(expr=10 ** (3 - out.pH) == blk.hOut)
    blk.waterEq = pyo.Constraint(expr=blk.hOut * blk.ohOut == 1e-8)
    blk.hReal = pyo.Expression(expr=blk.hOut / 1000.0)    # mol/L
    blk.ohReal = pyo.Expression(expr=blk.ohOut / 1000.0)  # mol/L

    blk.carbonateDenominator = pyo.Expression(expr=blk.hReal ** 2 + blk.ka1 * blk.hReal + blk.ka1 * blk.ka2)
    blk.hco3Conc = pyo.Expression(expr=blk.carbonTotalConc * blk.ka1 * blk.hReal / (blk.carbonateDenominator + 1e-30))
    blk.co3Conc = pyo.Expression(expr=blk.carbonTotalConc * blk.ka1 * blk.ka2 / (blk.carbonateDenominator + 1e-30))
    blk.nh4Conc = pyo.Expression(
        expr=blk.useAmmoniaSpeciation * blk.ammoniaTotalConc * blk.hReal / (blk.hReal + blk.kaNH4 + 1e-30)
    )
    blk.chargeBalance = pyo.Constraint(
        expr=blk.dissolvedSpectatorConc + blk.hReal + blk.nh4Conc == blk.ohReal + blk.hco3Conc + 2.0 * blk.co3Conc
    )
    blk.pHOut = pyo.Expression(expr=out.pH)
    blk.nh3FreeFracOut = pyo.Expression(expr=blk.kaNH4 / (blk.hReal + blk.kaNH4))

    # CO2 dose: whatever brings the outlet to the target pH
    blk.phSpec = pyo.Constraint(expr=out.pH == blk.targetpH)

    # Reporting in the old units (g-N/m3 of liquid, TAN)
    blk.nitrogenConcIn = pyo.Expression(expr=1000.0 * sIn.flow['tan'] / (blk.liquidVolFlowIn + 1e-12))
    blk.nitrogenConcOut = pyo.Expression(expr=1000.0 * out.flow['tan'] / (blk.liquidVolFlowOut + 1e-12))

    # -------------------- Gas flows and blower --------------------
    blk.gasPressurePa = pyo.Expression(expr=blk.gasInjectionPressureBar * blk.barToPa)
    blk.gasMolarDensity = pyo.Expression(expr=blk.gasPressurePa / (blk.R * blk.gasTempK + 1e-12))
    blk.co2GasFlowInM3S = pyo.Expression(expr=blk.co2GasMolFlowIn / (blk.gasMolarDensity + 1e-12))
    blk.nh3CarrierGasFlowInM3S = pyo.Expression(expr=blk.nh3CarrierGasMolFlowIn / (blk.gasMolarDensity + 1e-12))
    blk.totalGasFlowInM3S = pyo.Expression(expr=blk.co2GasFlowInM3S + blk.nh3CarrierGasFlowInM3S)
    blk.atmosphericPressureBar = pyo.Param(initialize=1.01325, mutable=True)
    blk.gasPressureRiseBar = pyo.Expression(expr=blk.gasInjectionPressureBar - blk.atmosphericPressureBar)
    blk.gasPressureRisePa = pyo.Expression(expr=blk.gasPressureRiseBar * blk.barToPa)

    # -------------------- Sizing and costs --------------------
    blk.volumeSizing = pyo.Constraint(expr=blk.tankVolume == blk.residenceTimeHr * 3600.0 * blk.bulkVolFlowIn)
    blk.mixingPower = pyo.Expression(expr=blk.mixingPowerDensity * blk.tankVolume)   # kW
    blk.blowerPower = pyo.Expression(
        expr=(blk.totalGasFlowInM3S * blk.gasPressureRisePa) / (blk.blowerEff * 1000.0 + 1e-12)
    )  # kW
    blk.capex = pyo.Expression(
        expr=blk.minCapex + 1.64 * blk.capexMultiplier * blk.costReference
        * (blk.tankVolume / (blk.volumeReference + 1e-12)) ** blk.capexFactor
    )
    blk.co2MassFlowIn = pyo.Expression(expr=blk.co2GasMolFlowIn * blk.molwtCO2)   # kg/s
    blk.co2Cost = pyo.Expression(expr=blk.co2MassFlowIn * blk.co2CostPerKg * model.daysOperation)
    blk.opex = pyo.Expression(
        expr=(blk.mixingPower + blk.blowerPower) * model.elecPrice * (model.daysOperation / 3600.0) + blk.co2Cost
    )

    return blk