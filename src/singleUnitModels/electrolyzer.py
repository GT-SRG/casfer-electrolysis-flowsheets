#------------------------------------------------------------------------------
# function:     electrolyzer.py                                               #
# Description:  Electrolyzer: liberates part of the solid-bound N into the    #
#               liquid phase. CaO solids, P, K, Ca, Mg pass through.          #
#                                                                             #
#               Mass transfer: the liberated N mass moves from orgSolids to   #
#               liquid (as dissolved organic N). Total mass is conserved;     #
#               the solids fraction decreases by liberatedN / inlet mass.     #
#                                                                             #
#               Streams: inlet -> outlet                                      #
#                                                                             #
#               N transformation:                                             #
#                 liberated = solidsNToLiquidFrac x inlet solidN              #
#                 TAN share of liberated = liberatedTanFrac (default 0.0:     #
#                 all liberated N is organic, so liquid TAN comes only from   #
#                 the feed); the rest becomes dissolved orgN.                 #
#               Volatilization/oxidation loss (nitrogenLossFraction) acts on  #
#               TAN only; orgN is non-volatile (orgNLossFraction, default 0). #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m.el                                                        #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import getParams
    from . import streamTools
except ImportError:
    import getParams
    import streamTools


def electrolyzer(m):

    m.el = pyo.Block()
    blk = m.el

    electrolyzerParams = getParams.params['Electrolyzer']

    blk.residenceTime = pyo.Param(initialize=electrolyzerParams['Residence Time'])     # s
    blk.costReference = pyo.Param(initialize=electrolyzerParams['Cost Reference'])     # $/m2
    blk.areaReference = pyo.Param(initialize=electrolyzerParams['Area Reference'])     # m2
    blk.capexFactor   = pyo.Param(initialize=electrolyzerParams['Capex Factor'])
    blk.SEC           = pyo.Param(initialize=electrolyzerParams['Specific Energy Consumption'])  # kWh/kg-N (solid-bound N processed)

    # N liberation
    blk.solidsNToLiquidFrac  = pyo.Param(initialize=0.219, mutable=True)  # fraction of solidN liberated
    blk.liberatedTanFrac     = pyo.Param(initialize=0.0, mutable=True)    # TAN share of liberated N
    blk.nitrogenLossFraction = pyo.Param(initialize=0.05, mutable=True)   # fraction of liquid TAN lost (volatilization + oxidation)
    blk.orgNLossFraction     = pyo.Param(initialize=0.0, mutable=True)    # fraction of dissolved orgN lost

    # -------------------- Streams --------------------
    streamTools.addStream(blk, 'inlet', initPH=13.0)
    streamTools.addStream(blk, 'outlet', initPH=13.0)
    sIn, out = blk.inlet, blk.outlet

    blk.area = pyo.Var(initialize=5.0, within=pyo.NonNegativeReals)  # m2

    # -------------------- Nitrogen --------------------
    blk.solidsNIn = pyo.Expression(expr=sIn.flow['solidN'])                          # kg-N/s
    blk.liberatedN = pyo.Expression(expr=blk.solidsNToLiquidFrac * blk.solidsNIn)     # kg-N/s
    blk.liberatedTan = pyo.Expression(expr=blk.liberatedTanFrac * blk.liberatedN)     # kg-N/s
    blk.liberatedOrgN = pyo.Expression(expr=blk.liberatedN - blk.liberatedTan)        # kg-N/s

    blk.tanAvailableForLoss = pyo.Expression(expr=sIn.flow['tan'] + blk.liberatedTan)                  # kg-N/s
    blk.orgNAvailableForLoss = pyo.Expression(expr=sIn.flow['orgN'] + blk.liberatedOrgN)               # kg-N/s
    blk.tanLost = pyo.Expression(expr=blk.nitrogenLossFraction * blk.tanAvailableForLoss)              # kg-N/s
    blk.orgNLost = pyo.Expression(expr=blk.orgNLossFraction * blk.orgNAvailableForLoss)                # kg-N/s
    blk.nitrogenLost = pyo.Expression(expr=blk.tanLost + blk.orgNLost)                                 # kg-N/s

    blk.solidNBalance = pyo.Constraint(expr=out.flow['solidN'] == sIn.flow['solidN'] - blk.liberatedN)
    blk.tanBalance = pyo.Constraint(expr=out.flow['tan'] == blk.tanAvailableForLoss - blk.tanLost)
    blk.orgNBalance = pyo.Constraint(expr=out.flow['orgN'] == blk.orgNAvailableForLoss - blk.orgNLost)

    # -------------------- Bulk mass transfer (liberated N: solids -> liquid) --------------------
    blk.orgSolidsBalance = pyo.Constraint(expr=out.flow['orgSolids'] == sIn.flow['orgSolids'] - blk.liberatedN)
    blk.liquidBalance = pyo.Constraint(expr=out.flow['liquid'] == sIn.flow['liquid'] + blk.liberatedN)

    # Everything else passes through unchanged
    streamTools.passComponents(
        blk, 'passBalance', sIn, out,
        streamTools.componentsExcept('solidN', 'tan', 'orgN', 'orgSolids', 'liquid')
    )
    blk.pHBalance = pyo.Constraint(expr=out.pH == sIn.pH)

    # -------------------- Area sizing (bench capacity basis) --------------------
    # Capacity is per kg of TOTAL dry solids leaving (organic + undissolved CaO)
    blk.sludgeOutDryBasis = pyo.Expression(expr=out.solidsMass)                    # kg/s
    blk.sludgeOutDryBasisKgPerDay = pyo.Expression(expr=blk.sludgeOutDryBasis * 86400.0)

    blk.elCapacityKgDsPerM2PerBatch = pyo.Param(initialize=14.286, mutable=True)
    blk.batchDurationHr = pyo.Param(initialize=0.5, mutable=True)
    blk.cleaningDowntimeFrac = pyo.Param(initialize=0.0, mutable=True)
    blk.operationHoursPerDay = pyo.Param(initialize=24.0, mutable=True)

    blk.batchesPerDay = pyo.Expression(
        expr=blk.operationHoursPerDay * (1.0 - blk.cleaningDowntimeFrac) / blk.batchDurationHr
    )
    blk.dailyCapacityPerM2 = pyo.Expression(expr=blk.elCapacityKgDsPerM2PerBatch * blk.batchesPerDay)
    blk.elAreaRequired = pyo.Expression(expr=blk.sludgeOutDryBasisKgPerDay / (blk.dailyCapacityPerM2 + 1e-9))
    blk.areaFromCapacity = pyo.Constraint(expr=blk.area == blk.elAreaRequired)

    # -------------------- CAPEX --------------------
    # Purchased cost of the complete electrolyzer system from the prototype: stack cost per m2
    # times the prototype system factor (1.64, covers power supply, pumps, tanks and the rest of
    # the balance of plant), for numStacks stacks (duty + standby). C_BM = F_BM * C_p.
    blk.stackUnitCost = pyo.Param(initialize=6000.0, mutable=True)        # $/m2
    blk.systemFactor = pyo.Param(initialize=1.64, mutable=True)           # system / stack cost, prototype
    blk.numStacks = pyo.Param(initialize=2.0, mutable=True)               # duty + standby
    blk.bareModuleFactor = pyo.Param(initialize=1.4, mutable=True)        # packaged skid
    blk.purchaseCost = pyo.Expression(expr=blk.numStacks * blk.systemFactor * blk.stackUnitCost * blk.area)
    blk.bareModuleCost = pyo.Expression(expr=blk.bareModuleFactor * blk.purchaseCost)
    blk.capex = pyo.Expression(expr=blk.bareModuleCost)   # bare-module cost, $

    # -------------------- OPEX --------------------
    blk.power = pyo.Expression(expr=blk.SEC * blk.solidsNIn * 3600.0)                      # kW
    blk.opex = pyo.Expression(expr=blk.power * m.elecPrice * (m.daysOperation / 3600.0))   # $ lifetime

    return blk