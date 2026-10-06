#------------------------------------------------------------------------------
# function:     nanofiltration.py                                             #
# Description:  Nanofiltration (bivalent removal) on a solids-free liquid.    #
#                                                                             #
#               Streams: inlet -> permeate (to GPM), retentate (reject)       #
#                                                                             #
#               - water recovery = permeate liquid / inlet liquid (>= target) #
#               - Ca, Mg: constant observed rejection                         #
#               - TAN: pH-dependent rejection R_NH4 x (1 - alpha_NH3)         #
#               - dissolved K, P, orgN: constant rejection parameters         #
#                 (placeholders, default 0; no NF data for these streams)     #
#               - any solids entering (should be none) go to the retentate    #
#                                                                             #
#               Costs: bare-module cost C_BM = F_BM x C_p (packaged skid,     #
#               F_BM = 1.4). At ~$500/m2 and pH 13 the membrane is a ceramic  #
#               NF element: 20-yr life, replacement charged as operating cost #
#               at membraneReplFrac (0.05/yr) of the membrane purchase cost.  #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m.nf                                                        #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import getParams
    from . import streamTools
except ImportError:
    import getParams
    import streamTools


rejectedSpecies = ['tan', 'ca', 'mg', 'liqK', 'liqP', 'orgN']
osmoticSpecies = ['tan', 'ca', 'mg']


def nf(m):

    m.nf = pyo.Block()
    blk = m.nf

    nfParams = (
        getParams.params.get('Nanofiltration')
        or getParams.params.get('nf')
        or getParams.params.get('Selective Membrane')
        or {}
    )

    blk.membraneCost   = pyo.Param(initialize=nfParams.get('Membrane Cost', 500.0))            # $/m2 (purchased)
    blk.membraneLp     = pyo.Param(initialize=nfParams.get('Hydraulic Permeability', 5.0))     # L/m2/h/bar
    blk.pumpEfficiency = pyo.Param(initialize=nfParams.get('Pump Efficiency', 0.75))
    blk.maxDeltaP      = pyo.Param(initialize=nfParams.get('Max Pressure Drop', 20.0), mutable=True)  # bar
    blk.targetRecovery = pyo.Param(initialize=nfParams.get('Target Recovery', 0.70), mutable=True)
    blk.membraneReplFrac = pyo.Param(initialize=nfParams.get('Membrane Replacement Fraction', 0.05), mutable=True)  # 1/yr (20-yr ceramic life)
    blk.bareModuleFactor = pyo.Param(initialize=1.4, mutable=True)                                 # packaged skid
    blk.minDrivingForce = pyo.Param(initialize=0.5, mutable=True)  # bar

    blk.rejectionNH4Intrinsic = pyo.Param(initialize=nfParams.get('NH4 Intrinsic Rejection', 0.90), mutable=True)
    blk.pKaNH3 = pyo.Param(initialize=9.25, mutable=True)
    blk.fixedRejection = pyo.Param(
        ['ca', 'mg', 'liqK', 'liqP', 'orgN'],
        initialize={
            'ca': nfParams.get('Ca Rejection', 0.93),
            'mg': nfParams.get('Mg Rejection', 0.88),
            'liqK': 0.0, 'liqP': 0.0, 'orgN': 0.0,
        },
        mutable=True,
    )
    blk.osmoticCoeff = pyo.Param(osmoticSpecies, initialize={'tan': 3.186, 'ca': 1.577, 'mg': 2.601}, mutable=True)

    # -------------------- Streams --------------------
    streamTools.addStream(blk, 'inlet', initPH=13.0)
    streamTools.addStream(blk, 'permeate', initPH=13.0)
    streamTools.addStream(blk, 'retentate', initPH=13.0)
    sIn, perm, ret = blk.inlet, blk.permeate, blk.retentate

    blk.recovery = pyo.Var(initialize=0.7, within=pyo.NonNegativeReals, bounds=(0.001, 0.999))
    blk.area = pyo.Var(initialize=10.0, within=pyo.NonNegativeReals, bounds=(1e-6, None))  # m2
    blk.deltaP = pyo.Var(initialize=2.0, within=pyo.NonNegativeReals, bounds=(0, None))    # bar

    blk.volFlowIn = pyo.Expression(expr=sIn.liquidVol)
    blk.permeateVolFlow = pyo.Expression(expr=perm.liquidVol)
    blk.retentateVolFlow = pyo.Expression(expr=ret.liquidVol)
    blk.alphaIn = pyo.Expression(expr=1.0 / (1.0 + 10 ** (blk.pKaNH3 - sIn.pH)))

    # -------------------- Water and solids --------------------
    blk.recoveryDef = pyo.Constraint(expr=perm.flow['liquid'] == blk.recovery * sIn.flow['liquid'])
    blk.retentateLiquid = pyo.Constraint(expr=ret.flow['liquid'] == sIn.flow['liquid'] - perm.flow['liquid'])
    blk.recoveryTarget = pyo.Constraint(expr=blk.recovery >= blk.targetRecovery)
    streamTools.passComponents(blk, 'solidsToRetentate', sIn, ret, streamTools.solidsFollowing)
    blk.noSolidsInPermeate = pyo.Constraint(streamTools.solidsFollowing, rule=lambda b, c: perm.flow[c] == 0.0)

    # -------------------- Dissolved species --------------------
    def _rejection(b, s):
        if s == 'tan':
            return b.rejectionNH4Intrinsic * (1.0 - b.alphaIn)
        return b.fixedRejection[s]
    blk.rejection = pyo.Expression(rejectedSpecies, rule=_rejection)
    blk.effectiveRejectionTAN = pyo.Expression(expr=blk.rejection['tan'])

    blk.concIn = pyo.Var(rejectedSpecies, initialize=0.5, within=pyo.NonNegativeReals)   # kg/m3
    blk.concRet = pyo.Var(osmoticSpecies, initialize=0.5, within=pyo.NonNegativeReals)   # kg/m3
    blk.concInDef = pyo.Constraint(rejectedSpecies, rule=lambda b, s: sIn.flow[s] == b.concIn[s] * blk.volFlowIn)
    blk.concPerm = pyo.Expression(rejectedSpecies, rule=lambda b, s: (1.0 - b.rejection[s]) * b.concIn[s])
    blk.permeateSpecies = pyo.Constraint(
        rejectedSpecies, rule=lambda b, s: perm.flow[s] == b.concPerm[s] * blk.permeateVolFlow
    )
    blk.retentateSpecies = pyo.Constraint(
        rejectedSpecies, rule=lambda b, s: ret.flow[s] == sIn.flow[s] - perm.flow[s]
    )
    blk.concRetDef = pyo.Constraint(
        osmoticSpecies, rule=lambda b, s: ret.flow[s] == b.concRet[s] * blk.retentateVolFlow
    )

    blk.pHPermeate = pyo.Constraint(expr=perm.pH == sIn.pH)
    blk.pHRetentate = pyo.Constraint(expr=ret.pH == sIn.pH)

    # -------------------- Osmotic pressure and flux --------------------
    blk.piFeedIn = pyo.Expression(expr=sum(blk.osmoticCoeff[s] * blk.concIn[s] for s in osmoticSpecies))
    blk.piRetentate = pyo.Expression(expr=sum(blk.osmoticCoeff[s] * blk.concRet[s] for s in osmoticSpecies))
    blk.piFeedAvg = pyo.Expression(expr=0.5 * (blk.piFeedIn + blk.piRetentate))
    blk.piPermeate = pyo.Expression(expr=sum(blk.osmoticCoeff[s] * blk.concPerm[s] for s in osmoticSpecies))
    blk.deltaPi = pyo.Expression(expr=blk.piFeedAvg - blk.piPermeate)

    blk.positiveFluxConstr = pyo.Constraint(expr=blk.deltaP >= blk.deltaPi + blk.minDrivingForce)
    blk.membraneFlux = pyo.Constraint(
        expr=blk.permeateVolFlow == blk.area * blk.membraneLp * (blk.deltaP - blk.deltaPi) * (1e-3 / 3600.0)
    )
    blk.pressureLimit = pyo.Constraint(expr=blk.deltaP <= blk.maxDeltaP)

    # -------------------- Reporting --------------------
    blk.concInTAN = pyo.Expression(expr=blk.concIn['tan'])
    blk.concPermTAN = pyo.Expression(expr=blk.concPerm['tan'])
    blk.nRecoveryToPermeate = pyo.Expression(expr=perm.flow['tan'] / (sIn.flow['tan'] + 1e-12))
    blk.caRejectionToRetentate = pyo.Expression(expr=ret.flow['ca'] / (sIn.flow['ca'] + 1e-12))
    blk.mgRejectionToRetentate = pyo.Expression(expr=ret.flow['mg'] / (sIn.flow['mg'] + 1e-12))

    # -------------------- Costs --------------------
    blk.purchaseCost = pyo.Expression(expr=blk.membraneCost * blk.area)
    blk.bareModuleCost = pyo.Expression(expr=blk.bareModuleFactor * blk.purchaseCost)
    blk.capex = pyo.Expression(expr=blk.bareModuleCost)   # bare-module cost, $
    blk.feedFlowM3h = pyo.Expression(expr=blk.volFlowIn * 3600.0)
    blk.pumpPower = pyo.Expression(expr=(blk.feedFlowM3h * blk.deltaP) / (36.0 * (blk.pumpEfficiency + 1e-9)))  # kW
    blk.projectYears = pyo.Expression(expr=m.daysOperation / (365.0 * 24.0 * 3600.0))
    blk.membraneReplOpex = pyo.Expression(expr=blk.membraneReplFrac * blk.purchaseCost * blk.projectYears)
    blk.pumpOpex = pyo.Expression(expr=blk.pumpPower * m.elecPrice * (m.daysOperation / 3600.0))
    blk.opex = pyo.Expression(expr=blk.pumpOpex + blk.membraneReplOpex)

    return blk