#------------------------------------------------------------------------------
# function:     centrifuge.py                                                 #
# Description:  Decanter centrifuge.                                          #
#                                                                             #
#               Streams: inlet -> cake, centrate                              #
#                                                                             #
#               Split rules:                                                  #
#                 - every solid component (organic, CaO, and solid-bound      #
#                   N/P/K) is captured by the same fraction                   #
#                   solidMassCaptured = 1 - exp(-captureRate x tau)           #
#                 - cake liquid follows from the cake TSS (compressibility    #
#                   model); dissolved species split with the liquid           #
#                   (cakeLiquidFrac), i.e. the centrifuge is non-selective    #
#                   for anything dissolved                                    #
#                                                                             #
#               Capital cost: bare-module cost C_BM = F_BM x C_p (Turton      #
#               centrifuge, F_BM = 1.57).                                     #
#                                                                             #
#               Uses m.y_cf (binary or Param) for the geometry big-M if the  #
#               flowsheet defines it; otherwise the unit is always on.        #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m.cf                                                        #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
from math import pi
try:
    from . import getParams
    from . import streamTools
except ImportError:
    import getParams
    import streamTools


def centrifuge(m):

    m.cf = pyo.Block()
    blk = m.cf

    centrifugeParams = getParams.params['Centrifuge']

    blk.costReference    = pyo.Param(initialize=centrifugeParams['Cost Reference'])     # $ (purchased)
    blk.volumeReference  = pyo.Param(initialize=centrifugeParams['Volume Reference'])   # m3/s
    blk.capexFactor      = pyo.Param(initialize=centrifugeParams['Capex Factor'])
    blk.bareModuleFactor = pyo.Param(initialize=1.57, mutable=True)                     # Turton centrifuge
    blk.beta             = pyo.Param(initialize=centrifugeParams['Beta'])
    blk.solidDiameter    = pyo.Param(initialize=centrifugeParams['Particle Diameter'])  # m
    blk.particleDensity  = pyo.Param(initialize=centrifugeParams.get('Particle Density', 2650))  # kg/m3

    # -------------------- Streams --------------------
    streamTools.addStream(blk, 'inlet')
    streamTools.addStream(blk, 'cake')
    streamTools.addStream(blk, 'centrate')
    sIn, cake, cent = blk.inlet, blk.cake, blk.centrate

    # -------------------- Geometry / operating variables --------------------
    blk.tankVolume   = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals, bounds=(0.0, 50.0))  # m3
    blk.agitRotation = pyo.Var(initialize=100, within=pyo.NonNegativeReals, bounds=(0, 250))     # rps
    blk.tankDiameter = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals, bounds=(0, 10))      # m
    blk.cakeTSS      = pyo.Var(initialize=0.20, within=pyo.NonNegativeReals, bounds=(0, 1))
    blk.cakeLiquidFrac = pyo.Var(initialize=0.05, within=pyo.NonNegativeReals, bounds=(0, 1))    # fraction of inlet liquid to cake

    blk.sludgeVolFlowIn = pyo.Expression(expr=sIn.bulkVol)  # m3/s

    # Geometry big-M, relaxed when the flowsheet switches the unit off
    yCf = m.y_cf if hasattr(m, 'y_cf') else 1.0
    blk.mGeo = pyo.Param(initialize=1e4, mutable=True)
    blk.tankDiameterUpper = pyo.Constraint(
        expr=blk.tankVolume - 1.178 * blk.tankDiameter ** 3 <= blk.mGeo * (1 - yCf)
    )
    blk.tankDiameterLower = pyo.Constraint(
        expr=blk.tankVolume - 1.178 * blk.tankDiameter ** 3 >= -blk.mGeo * (1 - yCf)
    )

    # -------------------- Capture model --------------------
    blk.residenceTime = pyo.Expression(expr=blk.tankVolume / (blk.sludgeVolFlowIn + 1e-8))       # s
    blk.omega = pyo.Expression(expr=2 * pi * blk.agitRotation)                                  # rad/s
    blk.gFactor = pyo.Expression(expr=(blk.omega ** 2) * (blk.tankDiameter / 2) / m.accGravity)
    blk.vt = pyo.Expression(
        expr=(blk.solidDiameter ** 2) * (blk.particleDensity - m.sludgeDensity) * m.accGravity
        / (18 * m.sludgeViscosity)
    )  # m/s, Stokes
    blk.vc = pyo.Expression(expr=blk.gFactor * blk.vt)                   # m/s
    blk.h = pyo.Expression(expr=blk.beta * blk.solidDiameter)            # m
    blk.captureRate = pyo.Expression(expr=blk.vc / blk.h)                # 1/s
    blk.solidMassCaptured = pyo.Expression(expr=1 - pyo.exp(-blk.captureRate * blk.residenceTime))

    # -------------------- Component split --------------------
    blk.cakeSolids = pyo.Constraint(
        streamTools.solidsFollowing,
        rule=lambda b, c: cake.flow[c] == b.solidMassCaptured * sIn.flow[c]
    )
    blk.cakeDissolved = pyo.Constraint(
        streamTools.liquidFollowing,
        rule=lambda b, c: cake.flow[c] == b.cakeLiquidFrac * sIn.flow[c]
    )
    blk.centrateBalance = pyo.Constraint(
        m.streamComponents, rule=lambda b, c: cent.flow[c] == sIn.flow[c] - cake.flow[c]
    )
    blk.cakeTSSDef = pyo.Constraint(expr=cake.solidsMass == blk.cakeTSS * cake.totalMass)

    blk.pHCake = pyo.Constraint(expr=cake.pH == sIn.pH)
    blk.pHCentrate = pyo.Constraint(expr=cent.pH == sIn.pH)

    # Back-compatible reporting names
    blk.solidsS = pyo.Expression(expr=cake.solidsMass)            # kg/s solids in cake
    blk.solidsL = pyo.Expression(expr=cent.solidsMass)            # kg/s solids lost to centrate
    blk.caoWeightFractionInCake = pyo.Expression(expr=cake.flow['caoSolids'] / (cake.solidsMass + 1e-9))

    # -------------------- Cake compressibility --------------------
    blk.compressibilityExponent = pyo.Param(initialize=centrifugeParams.get('Compressibility Exponent', 0.5))
    blk.referencePressure = pyo.Param(initialize=centrifugeParams.get('Reference Pressure', 1e6))  # Pa
    blk.referenceTSS = pyo.Param(initialize=centrifugeParams.get('TSS Reference', 0.2))

    blk.consolidationPressure = pyo.Expression(
        expr=0.5 * m.sludgeDensity * blk.omega ** 2 * ((blk.tankDiameter / 2) ** 2 - (blk.tankDiameter / 4) ** 2)
    )  # Pa
    blk.cakeTSSCompressibility = pyo.Constraint(
        expr=blk.cakeTSS == blk.referenceTSS * (blk.consolidationPressure / blk.referencePressure) ** blk.compressibilityExponent
    )
    blk.maxCakeTSS = pyo.Param(initialize=0.20, mutable=True)
    blk.cakeTSSCeiling = pyo.Constraint(expr=blk.cakeTSS <= blk.maxCakeTSS)

    # -------------------- Costs --------------------
    blk.purchaseCost = pyo.Expression(expr=blk.costReference * (blk.sludgeVolFlowIn / blk.volumeReference) ** blk.capexFactor)
    blk.bareModuleCost = pyo.Expression(expr=blk.bareModuleFactor * blk.purchaseCost)
    blk.capex = pyo.Expression(expr=blk.bareModuleCost)   # bare-module cost, $
    blk.powerNumber = pyo.Param(initialize=centrifugeParams['Power Number'] / 4)
    blk.powerW = pyo.Expression(expr=blk.powerNumber * m.sludgeDensity * blk.agitRotation ** 3 * blk.tankDiameter ** 5)
    blk.powerKW = pyo.Expression(expr=blk.powerW / 1000.0)
    blk.operatingHours = pyo.Expression(expr=m.daysOperation / 3600.0)
    blk.opex = pyo.Expression(expr=blk.powerKW * blk.operatingHours * m.elecPrice)

    return blk