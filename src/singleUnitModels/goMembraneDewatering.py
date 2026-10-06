#------------------------------------------------------------------------------
# function:     goMembraneDewatering.py                                       #
# Description:  GO membrane dewatering unit.                                  #
#                                                                             #
#               Streams: inlet -> retentate (E-GROW side), permeate           #
#                                                                             #
#               Split rules:                                                  #
#                 - all solids and solid-bound N/P/K -> retentate             #
#                 - liquid: permeate = permeateMassFlow (membrane flux)       #
#                 - TAN, Ca, Mg, dissolved K, dissolved P: constant observed  #
#                   rejection, C_perm = (1 - R) C_in                          #
#                 - dissolved organic N: fixed retention fraction to the      #
#                   retentate (orgNRetentionFrac, default 0.5), independent   #
#                   of the water split                                        #
#               Osmotic pressure from TAN, Ca, Mg (coefficients as before).   #
#                                                                             #
#               Costs: bare-module cost C_BM = F_BM x C_p (packaged skid,     #
#               F_BM = 1.4); membrane replacement is an operating cost at     #
#               membraneReplFrac of the membrane purchase cost per year.      #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m.go                                                        #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import getParams
    from . import streamTools
except ImportError:
    import getParams
    import streamTools


rejectedSpecies = ['tan', 'ca', 'mg', 'liqK', 'liqP']
osmoticSpecies = ['tan', 'ca', 'mg']


def goMembraneDewatering(m):

    m.go = pyo.Block()
    blk = m.go

    goParams = getParams.params.get('GO Membrane Dewatering') or {}

    blk.membraneCost    = pyo.Param(initialize=goParams.get('Membrane Cost', 500.0))              # $/m2 (purchased)
    blk.membraneLp      = pyo.Param(initialize=goParams.get('Hydraulic Permeability', 50.0))      # L/m2/h/bar
    blk.pumpEfficiency  = pyo.Param(initialize=goParams.get('Pump Efficiency', 0.75))
    blk.targetSolids    = pyo.Param(initialize=goParams.get('Target Solids', 0.25), mutable=True)  # retentate TSS
    blk.maxDeltaP       = pyo.Param(initialize=goParams.get('Max Pressure Drop', 30.0), mutable=True)  # bar
    blk.minDrivingForce = pyo.Param(initialize=0.5, mutable=True)  # bar
    blk.bareModuleFactor = pyo.Param(initialize=1.4, mutable=True)                                 # packaged skid
    blk.membraneReplFrac = pyo.Param(initialize=goParams.get('Membrane Replacement Fraction', 0.2), mutable=True)  # 1/yr (5-yr life)

    # Osmotic coefficients, bar per kg/m3 (TAN per kg-N/m3)
    blk.osmoticCoeff = pyo.Param(osmoticSpecies, initialize={'tan': 3.186, 'ca': 1.577, 'mg': 2.601}, mutable=True)

    # Observed rejection coefficients
    blk.rejection = pyo.Param(
        rejectedSpecies,
        initialize={'tan': 0.05, 'ca': 0.05, 'mg': 0.05, 'liqK': 0.0, 'liqP': 0.0},
        mutable=True,
    )
    # Fraction of dissolved organic N kept in the retentate
    blk.orgNRetentionFrac = pyo.Param(initialize=0.5, mutable=True)

    # -------------------- Streams --------------------
    streamTools.addStream(blk, 'inlet', initPH=13.0)
    streamTools.addStream(blk, 'retentate', initPH=13.0)
    streamTools.addStream(blk, 'permeate', initPH=13.0)
    sIn, ret, perm = blk.inlet, blk.retentate, blk.permeate

    blk.permeateMassFlow = pyo.Var(initialize=5.0, within=pyo.NonNegativeReals)            # kg/s
    blk.area   = pyo.Var(initialize=100.0, within=pyo.NonNegativeReals, bounds=(1e-6, None))  # m2
    blk.deltaP = pyo.Var(initialize=2.0, within=pyo.NonNegativeReals, bounds=(0, None))       # bar

    # -------------------- Bulk split --------------------
    streamTools.passComponents(blk, 'solidsToRetentate', sIn, ret, streamTools.solidsFollowing)

    def _zeroSolidsPermRule(b, c):
        return perm.flow[c] == 0.0
    blk.noSolidsInPermeate = pyo.Constraint(streamTools.solidsFollowing, rule=_zeroSolidsPermRule)

    blk.permeateLiquid = pyo.Constraint(expr=perm.flow['liquid'] == blk.permeateMassFlow)
    blk.retentateLiquid = pyo.Constraint(expr=ret.flow['liquid'] == sIn.flow['liquid'] - blk.permeateMassFlow)

    # Retentate solids target (linear), active by default; deactivate to let the optimizer choose
    blk.targetSolidsConstraint = pyo.Constraint(expr=ret.solidsMass == blk.targetSolids * ret.totalMass)

    # -------------------- Dissolved species (constant rejection) --------------------
    blk.concIn  = pyo.Var(rejectedSpecies, initialize=0.5, within=pyo.NonNegativeReals)  # kg/m3 (TAN: kg-N/m3)
    blk.concRet = pyo.Var(osmoticSpecies, initialize=0.5, within=pyo.NonNegativeReals)   # kg/m3

    blk.concInDef = pyo.Constraint(
        rejectedSpecies, rule=lambda b, s: sIn.flow[s] == b.concIn[s] * sIn.liquidVol
    )
    blk.concPerm = pyo.Expression(
        rejectedSpecies, rule=lambda b, s: (1.0 - b.rejection[s]) * b.concIn[s]
    )
    blk.permeateSpecies = pyo.Constraint(
        rejectedSpecies, rule=lambda b, s: perm.flow[s] == b.concPerm[s] * perm.liquidVol
    )
    blk.retentateSpecies = pyo.Constraint(
        rejectedSpecies, rule=lambda b, s: ret.flow[s] == sIn.flow[s] - perm.flow[s]
    )
    blk.concRetDef = pyo.Constraint(
        osmoticSpecies, rule=lambda b, s: ret.flow[s] == b.concRet[s] * ret.liquidVol
    )

    # -------------------- Dissolved organic N (fixed retention) --------------------
    blk.retentateOrgN = pyo.Constraint(expr=ret.flow['orgN'] == blk.orgNRetentionFrac * sIn.flow['orgN'])
    blk.permeateOrgN = pyo.Constraint(expr=perm.flow['orgN'] == sIn.flow['orgN'] - ret.flow['orgN'])

    # -------------------- pH --------------------
    blk.pHRetentate = pyo.Constraint(expr=ret.pH == sIn.pH)
    blk.pHPermeate = pyo.Constraint(expr=perm.pH == sIn.pH)

    # -------------------- Osmotic pressure and flux --------------------
    blk.piFeedIn = pyo.Expression(expr=sum(blk.osmoticCoeff[s] * blk.concIn[s] for s in osmoticSpecies))
    blk.piRetentate = pyo.Expression(expr=sum(blk.osmoticCoeff[s] * blk.concRet[s] for s in osmoticSpecies))
    blk.piFeedAvg = pyo.Expression(expr=0.5 * (blk.piFeedIn + blk.piRetentate))
    blk.piPermeate = pyo.Expression(expr=sum(blk.osmoticCoeff[s] * blk.concPerm[s] for s in osmoticSpecies))
    blk.deltaPi = pyo.Expression(expr=blk.piFeedAvg - blk.piPermeate)

    blk.positiveFluxConstr = pyo.Constraint(expr=blk.deltaP >= blk.deltaPi + blk.minDrivingForce)
    blk.membraneFlux = pyo.Constraint(
        expr=perm.liquidVol == blk.area * blk.membraneLp * (blk.deltaP - blk.deltaPi) * (1e-3 / 3600.0)
    )
    blk.pressureLimit = pyo.Constraint(expr=blk.deltaP <= blk.maxDeltaP)

    # Back-compatible names for reporting
    blk.concInTAN = pyo.Expression(expr=blk.concIn['tan'])
    blk.concRetTAN = pyo.Expression(expr=blk.concRet['tan'])
    blk.concPermTAN = pyo.Expression(expr=blk.concPerm['tan'])
    blk.liquidRetentionFrac = pyo.Expression(expr=ret.flow['liquid'] / (sIn.flow['liquid'] + 1e-12))  # reporting

    # -------------------- Costs --------------------
    blk.feedVolFlowBulk = pyo.Expression(expr=sIn.bulkVol)                    # m3/s, pump sizing
    blk.purchaseCost = pyo.Expression(expr=blk.membraneCost * blk.area)
    blk.bareModuleCost = pyo.Expression(expr=blk.bareModuleFactor * blk.purchaseCost)
    blk.capex = pyo.Expression(expr=blk.bareModuleCost)   # bare-module cost, $
    blk.feedFlowM3h = pyo.Expression(expr=blk.feedVolFlowBulk * 3600.0)
    blk.pumpPower = pyo.Expression(expr=(blk.feedFlowM3h * blk.deltaP) / (36.0 * (blk.pumpEfficiency + 1e-9)))  # kW
    blk.projectYears = pyo.Expression(expr=m.daysOperation / (365.0 * 24.0 * 3600.0))
    blk.membraneReplOpex = pyo.Expression(expr=blk.membraneReplFrac * blk.purchaseCost * blk.projectYears)
    blk.pumpOpex = pyo.Expression(expr=blk.pumpPower * m.elecPrice * (m.daysOperation / 3600.0))
    blk.opex = pyo.Expression(expr=blk.pumpOpex + blk.membraneReplOpex)

    return blk