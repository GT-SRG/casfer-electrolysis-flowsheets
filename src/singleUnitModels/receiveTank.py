#------------------------------------------------------------------------------
# function:     receiveTank.py                                                #
# Description:  Receiving (acid) tank: reprotonates the pH-13 stream to the   #
#               target pH with H2SO4.                                         #
#                                                                             #
#               Streams: inlet -> outlet                                      #
#                                                                             #
#               Acid demand (mol H+/s):                                       #
#                 residual OH- (residualOHMolPerM3 x inlet liquid volume)     #
#               + dissolved TAN (NH3 -> NH4+, 1 H+ per N)                     #
#               + sludge buffering (alkalinityBufferMolPerM3 x liquid volume) #
#               - acidEqCredit (excess H+ supplied by another unit, e.g. the  #
#                 GPM product in FS2; fixed at 0 unless the flowsheet frees  #
#                 and links it)                                               #
#               Solid-bound N and dissolved organic N carry no acid demand.   #
#               The acid solution adds to the liquid mass.                    #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#               - blockName : block name (default 'rt')                       #
#                                                                             #
# Output:       - m.<blockName> (default 'rt'); also used as the reject-     #
#                 stream neutralization tank (blockName 'nt')                #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import getParams
    from . import streamTools
except ImportError:
    import getParams
    import streamTools


def receiveTank(m, blockName='rt'):

    blk = pyo.Block()
    m.add_component(blockName, blk)

    receiveTankParams = getParams.params['Receiving Tank']

    blk.residenceTime   = pyo.Param(initialize=receiveTankParams['Residence Time'])    # s
    blk.costReference   = pyo.Param(initialize=receiveTankParams['Cost Reference'])    # $
    blk.volumeReference = pyo.Param(initialize=receiveTankParams['Volume Reference'])  # m3
    blk.capexFactor     = pyo.Param(initialize=receiveTankParams['Capex Factor'])
    blk.acidCost        = pyo.Param(initialize=receiveTankParams['Acid Cost'])         # $/kg pure H2SO4
    blk.acidDensity     = pyo.Param(initialize=receiveTankParams['Acid Density'])      # kg/m3

    blk.targetpH                 = pyo.Param(initialize=7.0, mutable=True)
    blk.molwtH2SO4               = pyo.Param(initialize=0.098, mutable=True)   # kg/mol
    blk.acidSolutionWtFraction   = pyo.Param(initialize=0.93, mutable=True)    # kg H2SO4 / kg solution
    blk.residualOHMolPerM3       = pyo.Param(initialize=100.0, mutable=True)   # mol OH- / m3 liquid
    blk.alkalinityBufferMolPerM3 = pyo.Param(initialize=20.0, mutable=True)    # mol H+ / m3 liquid
    blk.minCapex                 = pyo.Param(initialize=10000.0, mutable=True)

    # -------------------- Streams --------------------
    streamTools.addStream(blk, 'inlet', initPH=13.0)
    streamTools.addStream(blk, 'outlet', initPH=7.0)
    sIn, out = blk.inlet, blk.outlet

    blk.acidMassFlowIn = pyo.Var(initialize=0.1, within=pyo.NonNegativeReals)   # kg/s acid solution
    blk.tankVolume     = pyo.Var(initialize=50.0, within=pyo.NonNegativeReals)  # m3

    # Excess acid equivalents supplied elsewhere (mol H+/s); fixed at zero by default
    blk.acidEqCredit = pyo.Var(initialize=0.0, within=pyo.NonNegativeReals)
    blk.acidEqCredit.fix(0.0)

    # -------------------- Stoichiometric acid demand --------------------
    # ohFromInletPH = 0: fixed residual OH- (pH-13 conditioned sludge, as before);
    # ohFromInletPH = 1: free OH- from the inlet pH, [OH-] = 10^(pH-11) mol/m3 (used to neutralize reject streams)
    blk.ohFromInletPH = pyo.Param(initialize=0.0, mutable=True)
    blk.ohNeutralizationDemand = pyo.Expression(
        expr=((1.0 - blk.ohFromInletPH) * blk.residualOHMolPerM3 + blk.ohFromInletPH * 10 ** (sIn.pH - 11.0)) * sIn.liquidVol
    )  # mol/s
    blk.tanProtonationDemand = pyo.Expression(expr=sIn.flow['tan'] / streamTools.molwtN)            # mol/s
    blk.bufferingAcidDemand = pyo.Expression(expr=blk.alkalinityBufferMolPerM3 * sIn.liquidVol)     # mol/s
    blk.grossAcidEqPerS = pyo.Expression(
        expr=blk.ohNeutralizationDemand + blk.tanProtonationDemand + blk.bufferingAcidDemand
    )
    blk.totalAcidEqPerS = pyo.Expression(expr=blk.grossAcidEqPerS - blk.acidEqCredit)                # mol/s
    blk.h2so4RequiredMolPerS = pyo.Expression(expr=blk.totalAcidEqPerS / 2.0)
    blk.h2so4RequiredKgPerS = pyo.Expression(expr=blk.h2so4RequiredMolPerS * blk.molwtH2SO4)         # kg/s pure
    blk.acidSolutionMassFlowKgPerS = pyo.Expression(expr=blk.h2so4RequiredKgPerS / blk.acidSolutionWtFraction)

    blk.acidFlowConstr = pyo.Constraint(expr=blk.acidMassFlowIn == blk.acidSolutionMassFlowKgPerS)
    blk.netAcidNonNegative = pyo.Constraint(expr=blk.totalAcidEqPerS >= 0.0)

    # -------------------- Component balances --------------------
    blk.liquidBalance = pyo.Constraint(expr=out.flow['liquid'] == sIn.flow['liquid'] + blk.acidMassFlowIn)
    streamTools.passComponents(
        blk, 'passBalance', sIn, out,
        ['orgSolids', 'caoSolids', 'solidN', 'solidP', 'solidK', 'tan', 'orgN', 'liqP', 'liqK', 'ca', 'mg']
    )
    blk.pHConstr = pyo.Constraint(expr=out.pH == blk.targetpH)

    # -------------------- Sizing and costs --------------------
    blk.receiveTankVolume = pyo.Constraint(expr=blk.tankVolume == blk.residenceTime * sIn.bulkVol)
    blk.capex = pyo.Expression(
        expr=blk.minCapex + 1.64 * blk.costReference * (blk.tankVolume / blk.volumeReference) ** blk.capexFactor
    )
    blk.opex = pyo.Expression(expr=blk.acidCost * blk.h2so4RequiredKgPerS * m.daysOperation)  # $ lifetime, pure acid basis

    return blk