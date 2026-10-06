#------------------------------------------------------------------------------
# function:     nh3Stripper.py                                                #
# Description:  Sweep-gas NH3 stripper, N-stage Kremser countercurrent.       #
#                                                                             #
#               Streams: inlet -> outlet (liquid side)                        #
#               Gas side: sweepGasMolFlowIn -> gasMolFlowOut (carrier + NH3)  #
#                                                                             #
#               Only TAN is strippable. Dissolved organic N, solids and all   #
#               other species pass through. Outlet pH from a live charge      #
#               balance on the TAN system (spectator charge Z_net conserved). #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#               - blockName : block name (default 'nst')                      #
#                                                                             #
# Output:       - m.nst                                                       #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import getParams
    from . import streamTools
except ImportError:
    import getParams
    import streamTools


def nh3Stripper(m, blockName='nst'):

    blk = pyo.Block()
    m.add_component(blockName, blk)

    spargerParams = getParams.params.get('Sparger', {}) or {}

    def _get(key, default):
        try:
            return float(spargerParams.get(key, default))
        except Exception:
            return default

    # -------------------- Parameters --------------------
    blk.residenceTimeHr = pyo.Param(initialize=_get('Residence Time', 0.25), mutable=True)          # hr per stage
    blk.mixingPowerDensity = pyo.Param(initialize=_get('Mixing Power Density', 1.5), mutable=True)  # kW/m3
    blk.blowerEff = pyo.Param(initialize=_get('Blower Efficiency', 0.7), mutable=True)
    blk.gasInjectionPressureBar = pyo.Param(initialize=_get('Gas Injection Pressure', 1.1), mutable=True)  # bar abs
    blk.atmosphericPressureBar = pyo.Param(initialize=1.01325, mutable=True)
    blk.gasTempK = pyo.Param(initialize=_get('Gas Temperature', 298.15), mutable=True)

    blk.costReference = pyo.Param(initialize=_get('Cost Reference', 5000.0), mutable=True)    # $ at volumeReference
    blk.volumeReference = pyo.Param(initialize=_get('Volume Reference', 10.0), mutable=True)  # m3
    blk.capexFactor = pyo.Param(initialize=_get('Capex Factor', 0.7), mutable=True)
    blk.bareModuleFactor = pyo.Param(initialize=4.07, mutable=True)   # Turton CS vertical process vessel/tower, ambient P

    blk.henryConst = pyo.Param(initialize=_get('NH3 Henry Constant', 58.5), mutable=True)  # mol/L/atm
    blk.pKaNH3 = pyo.Param(initialize=_get('NH4 pKa', 9.25), mutable=True)
    blk.molwtN = pyo.Param(initialize=streamTools.molwtN, mutable=True)                    # kg/mol

    # -------------------- Streams --------------------
    streamTools.addStream(blk, 'inlet', initPH=13.0)
    streamTools.addStream(blk, 'outlet', initPH=12.5)
    sIn, out = blk.inlet, blk.outlet

    blk.numberOfStages = pyo.Var(initialize=3.0, within=pyo.NonNegativeReals, bounds=(1.0, 10.0))
    blk.concIn = pyo.Var(initialize=0.5, within=pyo.NonNegativeReals)    # kg-N/m3 TAN in
    blk.concOut = pyo.Var(initialize=0.05, within=pyo.NonNegativeReals)  # kg-N/m3 TAN out
    blk.sweepGasMolFlowIn = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals)   # mol/s
    blk.nh3StrippedMolFlow = pyo.Var(initialize=0.01, within=pyo.NonNegativeReals)  # mol/s

    blk.liquidVolFlowIn = pyo.Expression(expr=sIn.liquidVol)             # m3/s
    blk.liquidVolFlowLps = pyo.Expression(expr=blk.liquidVolFlowIn * 1000.0)
    blk.pHIn = pyo.Expression(expr=sIn.pH)

    # -------------------- Stream balances --------------------
    blk.concInDef = pyo.Constraint(expr=sIn.flow['tan'] == blk.concIn * blk.liquidVolFlowIn)
    blk.concOutDef = pyo.Constraint(expr=out.flow['tan'] == blk.concOut * blk.liquidVolFlowIn)
    blk.nitrogenBalance = pyo.Constraint(
        expr=out.flow['tan'] == sIn.flow['tan'] - blk.nh3StrippedMolFlow * blk.molwtN
    )
    streamTools.passComponents(blk, 'passBalance', sIn, out, streamTools.componentsExcept('tan'))

    # -------------------- Inlet speciation (mol/m3) --------------------
    blk.hIn = pyo.Expression(expr=10 ** (3 - sIn.pH))
    blk.ohIn = pyo.Expression(expr=1e-8 / (blk.hIn + 1e-30))
    blk.tanInMolM3 = pyo.Expression(expr=blk.concIn / blk.molwtN)
    blk.alphaIn = pyo.Expression(expr=1.0 / (1.0 + 10 ** (blk.pKaNH3 - sIn.pH)))
    blk.nh4In = pyo.Expression(expr=blk.tanInMolM3 * (1.0 - blk.alphaIn))
    blk.zNetIn = pyo.Expression(expr=blk.ohIn - blk.hIn - blk.nh4In)

    # -------------------- Live outlet speciation (mol/m3) --------------------
    blk.hOut = pyo.Var(initialize=1e-4, bounds=(1e-11, 1e4), within=pyo.NonNegativeReals)
    blk.ohOut = pyo.Var(initialize=1e2, bounds=(1e-11, 1e6), within=pyo.NonNegativeReals)
    blk.nh4Out = pyo.Var(initialize=0.01, within=pyo.NonNegativeReals)
    blk.nh3Out = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals)
    blk.zNetOut = pyo.Var(initialize=0.0, bounds=(-1e6, 1e6))

    blk.zOutConstr = pyo.Constraint(expr=blk.zNetOut == blk.zNetIn)
    blk.waterEq = pyo.Constraint(expr=blk.hOut * blk.ohOut == 1e-8)
    blk.ammoniaEq = pyo.Constraint(expr=(10 ** (-blk.pKaNH3) * 1000.0) * blk.nh4Out == blk.hOut * blk.nh3Out)
    blk.tanOutMolM3 = pyo.Expression(expr=blk.concOut / blk.molwtN)
    blk.tanOutDef = pyo.Constraint(expr=blk.tanOutMolM3 == blk.nh3Out + blk.nh4Out)
    blk.chargeBalance = pyo.Constraint(expr=blk.zNetOut + blk.hOut + blk.nh4Out == blk.ohOut)
    blk.phDef = pyo.Constraint(expr=10 ** (3 - out.pH) == blk.hOut)
    blk.pHOut = pyo.Expression(expr=out.pH)
    blk.alphaOut = pyo.Expression(expr=1.0 / (1.0 + 10 ** (blk.pKaNH3 - out.pH)))

    blk.tanInMolFlow = pyo.Expression(expr=sIn.flow['tan'] / blk.molwtN)   # mol/s
    blk.nh3AvailableMolFlow = pyo.Expression(expr=blk.tanInMolFlow)

    # -------------------- Kremser --------------------
    blk.strippingFactorRaw = pyo.Expression(
        expr=blk.sweepGasMolFlowIn / (blk.henryConst * 1.0 * blk.liquidVolFlowLps + 1e-12)
    )
    blk.strippingFactor = pyo.Expression(expr=blk.strippingFactorRaw + 1e-6)  # keeps S away from exactly 1
    blk.kremserNumerator = pyo.Expression(expr=blk.strippingFactor ** (blk.numberOfStages + 1.0) - blk.strippingFactor)
    blk.kremserDenominator = pyo.Expression(expr=blk.strippingFactor ** (blk.numberOfStages + 1.0) - 1.0)
    blk.nh3StrippedFraction = pyo.Expression(expr=blk.kremserNumerator / blk.kremserDenominator)
    blk.nh3StrippedDef = pyo.Constraint(
        expr=blk.nh3StrippedMolFlow == blk.nh3StrippedFraction * blk.nh3AvailableMolFlow
    )

    # Carrier gas leaving (sweep gas + stripped NH3)
    blk.gasMolFlowOut = pyo.Expression(expr=blk.sweepGasMolFlowIn + blk.nh3StrippedMolFlow)
    blk.nh3SpeciesFracOut = pyo.Expression(expr=blk.nh3StrippedMolFlow / (blk.gasMolFlowOut + 1e-12))

    # -------------------- Sizing --------------------
    blk.tankVolume = pyo.Var(initialize=10.0, within=pyo.NonNegativeReals)
    blk.volumeSizing = pyo.Constraint(
        expr=blk.tankVolume == blk.numberOfStages * blk.residenceTimeHr * 3600.0 * blk.liquidVolFlowIn
    )
    blk.mixingPower = pyo.Expression(expr=blk.mixingPowerDensity * blk.tankVolume)  # kW

    # -------------------- Blower --------------------
    blk.R = pyo.Param(initialize=8.314462618)
    blk.barToPa = pyo.Param(initialize=1e5)
    blk.gasPressurePa = pyo.Expression(expr=blk.gasInjectionPressureBar * blk.barToPa)
    blk.gasMolarDensity = pyo.Expression(expr=blk.gasPressurePa / (blk.R * blk.gasTempK + 1e-12))
    blk.gasFlowInM3S = pyo.Expression(expr=blk.sweepGasMolFlowIn / (blk.gasMolarDensity + 1e-12))
    blk.gasPressureRiseBar = pyo.Expression(expr=blk.gasInjectionPressureBar - blk.atmosphericPressureBar)
    blk.gasPressureRisePa = pyo.Expression(expr=blk.gasPressureRiseBar * blk.barToPa)
    blk.blowerPower = pyo.Expression(
        expr=(blk.gasFlowInM3S * blk.gasPressureRisePa) / (blk.blowerEff * 1000.0 + 1e-12)
    )

    # -------------------- Costs --------------------
    blk.purchaseCost = pyo.Expression(
        expr=blk.costReference * (blk.tankVolume / (blk.volumeReference + 1e-12)) ** blk.capexFactor
    )
    blk.bareModuleCost = pyo.Expression(expr=blk.bareModuleFactor * blk.purchaseCost)
    blk.capex = pyo.Expression(expr=blk.bareModuleCost)   # bare-module cost, $
    blk.opex = pyo.Expression(expr=(blk.mixingPower + blk.blowerPower) * m.elecPrice * (m.daysOperation / 3600.0))

    return blk