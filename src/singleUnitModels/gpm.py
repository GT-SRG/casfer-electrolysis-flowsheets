#-------------------------------------------------------------------------------
# function:     gpm.py                                                         #
# Description:  Gas-permeable membrane (H2SO4 draw) with recirculation loop    #
#               and pressure-drop-based pump cost.                             #
#                                                                              #
#               Streams: inlet (solids-free liquid) -> raffinate, product      #
#                 raffinate: inlet minus captured TAN and minus the water      #
#                            vapor that crosses to the draw side               #
#                 product  : ammonium sulfate solution (liquid mass =          #
#                            productMassFlow, TAN = captured N)                #
#               Only TAN crosses the membrane; orgN, K, P, Ca, Mg stay in      #
#               the raffinate.                                                 #
#               product.pH is solved from the draw-side charge balance:        #
#               NH4+/NH3 (pKa 9.25) and HSO4-/SO4 2- (pKa2 1.99) on the        #
#               product liquid volume, with total N = captured N and total     #
#               S = all H2SO4 dosed (reacted + residual).                      #
#                                                                              #
# Input:        - m : Pyomo concrete model                                     #
#                                                                              #
# Output:       - m.gpm                                                        #
#-------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import getParams
    from . import streamTools
except ImportError:
    import getParams
    import streamTools


def gpm(m):

    m.gpm = pyo.Block()
    blk = m.gpm

    gpmParams = getParams.params['Gas Permeable Membrane']

    # Core parameters
    blk.pKa = pyo.Param(initialize=gpmParams['pKa'], within=pyo.Any)
    blk.acidCost = pyo.Param(initialize=gpmParams['Acid Cost'] / 2.0)   # $/kg solution
    blk.inletDensity = pyo.Param(initialize=gpmParams['Inlet Density'])
    blk.inletViscosity = pyo.Param(initialize=gpmParams['Inlet Viscosity'])
    blk.inletDiffusivity = pyo.Param(initialize=gpmParams['Inlet Diffusivity'])
    blk.hydDiameter = pyo.Param(initialize=0.001)
    blk.moduleLength = pyo.Param(initialize=0.5)
    blk.modulesInSeries = pyo.Param(initialize=20.0, mutable=True)
    blk.effLength = pyo.Expression(expr=blk.moduleLength * blk.modulesInSeries)

    blk.acidDensity = pyo.Param(initialize=gpmParams['Acid Density'])
    blk.costReference = pyo.Param(initialize=125 * 130)
    blk.areaReference = pyo.Param(initialize=130)
    blk.capexFactor = pyo.Param(initialize=gpmParams['Capex Factor'])
    blk.pumpEff = pyo.Param(initialize=gpmParams['Pump Efficiency'])
    blk.costExponent = pyo.Param(initialize=0.7)
    blk.bareModuleFactor = pyo.Param(initialize=1.4, mutable=True)        # packaged skid
    blk.membraneReplFrac = pyo.Param(initialize=0.2, mutable=True)        # 1/yr (5-yr membrane life)

    blk.recircEffect = pyo.Param(initialize=1.0, mutable=True)
    blk.minorK = pyo.Param(initialize=3.0, mutable=True)
    blk.staticDP = pyo.Param(initialize=10000.0, mutable=True)   # Pa

    blk.molwtN = pyo.Param(initialize=0.01401)
    blk.molwtCaO = pyo.Param(initialize=0.05608)
    blk.molwtH2SO4 = pyo.Param(initialize=0.09808, mutable=True)
    blk.molwtAmSulfate = pyo.Param(initialize=0.13214, mutable=True)
    blk.targetNwtPercent = pyo.Param(initialize=12.17, mutable=True)
    # Ammonium sulfate product density (~40 wt% (NH4)2SO4 at 25 C, near the solubility ceiling)
    blk.productDensity = pyo.Param(
        initialize=gpmParams.get('Product Density', 1230.0), mutable=True
    )  # kg/m3
    blk.pKaHSO4 = pyo.Param(initialize=1.99, mutable=True)   # HSO4- <-> H+ + SO4 2-

    # -------------------- Streams --------------------
    streamTools.addStream(blk, 'inlet', initPH=13.0)
    streamTools.addStream(blk, 'raffinate', initPH=12.0)
    streamTools.addStream(blk, 'product', initPH=5.0)
    sIn, raff, prod = blk.inlet, blk.raffinate, blk.product

    blk.massFlowIn = pyo.Expression(expr=sIn.totalMass)       # kg/s
    blk.volFlowIn = pyo.Expression(expr=sIn.liquidVol)        # m3/s
    blk.massFlowOut = pyo.Expression(expr=raff.totalMass)     # kg/s
    blk.volFlowOut = pyo.Expression(expr=raff.liquidVol)      # m3/s
    blk.inletpH = pyo.Expression(expr=sIn.pH)

    blk.concInFresh = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals)   # kg-N/m3, fresh feed
    blk.concIn = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals)        # kg-N/m3, after recirculation mixing
    blk.concOut = pyo.Var(initialize=0.1, within=pyo.NonNegativeReals)       # kg-N/m3
    blk.nRemoved = pyo.Var(initialize=0.1, within=pyo.NonNegativeReals)      # kg-N/s

    blk.concInFreshDef = pyo.Constraint(expr=sIn.flow['tan'] == blk.concInFresh * blk.volFlowIn)
    # The inlet pH must stay inside the chemistry's validity range
    blk.inletPHBounds = pyo.Constraint(expr=pyo.inequality(1.0, sIn.pH, 13.0))

    # Design and loop variables
    blk.area = pyo.Var(initialize=100.0, bounds=(5.0, 20000.0))    # m2
    blk.recircRatio = pyo.Var(initialize=1.0, bounds=(0.0, 10.0))  # Qrecirc/Qfeed
    blk.velocity = pyo.Var(initialize=0.2, bounds=(0.02, 2.0))     # m/s

    # Mass transfer
    blk.Re = pyo.Var(initialize=1000.0, within=pyo.NonNegativeReals)
    blk.Sc = pyo.Var(initialize=500.0, within=pyo.NonNegativeReals)
    blk.Sh = pyo.Var(initialize=100.0, within=pyo.NonNegativeReals)
    blk.kL = pyo.Var(initialize=1e-5, within=pyo.NonNegativeReals)
    blk.kO = pyo.Var(initialize=1e-5, within=pyo.NonNegativeReals)

    # Acid side
    blk.acidWtFracIn = pyo.Var(initialize=0.5, bounds=(0.1, 0.5), within=pyo.NonNegativeReals)
    blk.acidFlowIn = pyo.Var(initialize=0.01, within=pyo.NonNegativeReals)   # m3/s

    # Optional base addition, disabled
    blk.baseFlow = pyo.Var(initialize=0.0, within=pyo.NonNegativeReals)
    blk.baseFlow.fix(0.0)

    # Outlet (raffinate) chemistry, mol/m3
    blk.hOut = pyo.Var(initialize=1e-7, bounds=(1e-12, 1e2), within=pyo.NonNegativeReals)
    blk.ohOut = pyo.Var(initialize=1e-7, bounds=(1e-12, 1e5), within=pyo.NonNegativeReals)
    blk.nh4Out = pyo.Var(initialize=10.0, within=pyo.NonNegativeReals)
    blk.nh3Out = pyo.Var(initialize=10.0, within=pyo.NonNegativeReals)
    blk.zNetOut = pyo.Var(initialize=0.0, bounds=(-1e6, 1e6))
    blk.pHOut = pyo.Var(initialize=10.0, bounds=(8.0, 13.0), within=pyo.NonNegativeReals)

    # Inlet chemistry
    blk.hIn = pyo.Expression(expr=10 ** (3 - sIn.pH))
    blk.ohIn = pyo.Expression(expr=1e-8 / (blk.hIn + 1e-12))
    blk.tanInMolar = pyo.Expression(expr=blk.concIn / blk.molwtN)
    blk.alphaIn = pyo.Expression(expr=1.0 / (1.0 + 10 ** (blk.pKa - sIn.pH)))
    blk.nh4In = pyo.Expression(expr=blk.tanInMolar * (1.0 - blk.alphaIn))
    blk.zNetIn = pyo.Expression(expr=blk.ohIn - blk.hIn - blk.nh4In)

    blk.loopFlow = pyo.Expression(expr=blk.volFlowIn * (1.0 + blk.recircRatio))

    blk.inletConcRecirculation = pyo.Constraint(
        expr=blk.concIn * blk.loopFlow == blk.concInFresh * blk.volFlowIn + blk.concOut * blk.recircRatio * blk.volFlowIn
    )

    # Geometry and hydraulics
    blk.crossArea = pyo.Expression(expr=(blk.hydDiameter * blk.area) / (4.0 * blk.effLength))
    blk.velocityConstr = pyo.Constraint(expr=blk.velocity == blk.loopFlow / (blk.crossArea + 1e-12))
    blk.reConstr = pyo.Constraint(expr=blk.Re == (blk.velocity * blk.inletDensity * blk.hydDiameter) / (blk.inletViscosity + 1e-12))
    blk.scConstr = pyo.Constraint(expr=blk.Sc == blk.inletViscosity / (blk.inletDensity * blk.inletDiffusivity + 1e-18))
    blk.shConstr = pyo.Constraint(expr=blk.Sh == 1.62 * (blk.Re * blk.Sc * blk.hydDiameter / blk.moduleLength) ** (1 / 3))
    blk.kLConstr = pyo.Constraint(expr=blk.kL == (blk.Sh * blk.inletDiffusivity) / (blk.hydDiameter + 1e-12))

    blk.porosity = pyo.Param(initialize=0.7)
    blk.tortuosity = pyo.Param(initialize=2.5)
    blk.thickness = pyo.Param(initialize=50e-6)        # m
    blk.gasDiffusivity = pyo.Param(initialize=2e-5)    # m2/s, NH3 in air
    blk.km = pyo.Expression(expr=(blk.porosity * blk.gasDiffusivity) / (blk.tortuosity * blk.thickness))
    blk.kOverallConstr = pyo.Constraint(expr=blk.kO == 1.0 / ((1.0 / (blk.kL + 1e-12)) + (1.0 / (blk.km + 1e-12))))

    # Outlet chemistry
    blk.zOutConstr = pyo.Constraint(
        expr=blk.zNetOut == blk.zNetIn + 2.0 * (blk.baseFlow / blk.molwtCaO) / (blk.volFlowIn + 1e-9)
    )
    blk.waterEq = pyo.Constraint(expr=blk.hOut * blk.ohOut == 1e-8)
    blk.ammoniaEq = pyo.Constraint(expr=(10 ** (-blk.pKa) * 1000.0) * blk.nh4Out == blk.hOut * blk.nh3Out)
    blk.tanOutDef = pyo.Constraint(expr=blk.concOut / blk.molwtN == blk.nh3Out + blk.nh4Out)
    blk.chargeBalance = pyo.Constraint(expr=blk.zNetOut + blk.hOut + blk.nh4Out == blk.ohOut)
    blk.phDef = pyo.Constraint(expr=10 ** (3 - blk.pHOut) == blk.hOut)

    blk.alphaOut = pyo.Expression(expr=blk.nh3Out / (blk.nh3Out + blk.nh4Out + 1e-9))
    blk.alphaAvg = pyo.Expression(expr=0.5 * (blk.alphaIn + blk.alphaOut))

    blk.performanceConstr = pyo.Constraint(
        expr=blk.concOut == blk.concIn * pyo.exp(-1.0 * blk.kO * blk.area * blk.alphaAvg / (blk.loopFlow + 1e-9))
    )
    blk.nRemovedConstr = pyo.Constraint(expr=blk.nRemoved == blk.volFlowIn * (blk.concInFresh - blk.concOut))

    # Acid stoichiometry (H2SO4 basis)
    nPerH2SO4 = 2.0 * 14.01 / 98.08
    blk.acidMassFlow = pyo.Expression(expr=blk.acidFlowIn * blk.acidDensity)
    blk.h2so4MassFlow = pyo.Expression(expr=blk.acidMassFlow * blk.acidWtFracIn)
    blk.acidNCapacity = pyo.Expression(expr=blk.h2so4MassFlow * nPerH2SO4)
    blk.acidStoichEq = pyo.Constraint(expr=blk.acidNCapacity == blk.nRemoved)

    # -------------------- Water transport across the membrane (declared here, closed below) ----------
    blk.gasConstR = pyo.Param(initialize=8.314, mutable=True)            # J/(mol.K)
    blk.operatingTempK = pyo.Param(initialize=298.15, mutable=True)      # K
    blk.waterMolarVolume = pyo.Param(initialize=1.807e-5, mutable=True)  # m3/mol
    blk.waterPermeability = pyo.Param(initialize=5.3e-15, mutable=True)  # m3/(m2.s.Pa) = 5.3e-10 m/(s.bar)
    blk.osmoticCoefficient = pyo.Param(initialize=0.7, mutable=True)     # feed side only (dilute)

    blk.jWaterVapor = pyo.Var(initialize=1e-6, bounds=(-1e-2, 1e-2))    # m3/s, positive = feed -> draw
    blk.jWaterVaporMassFlow = pyo.Expression(expr=blk.jWaterVapor * 1000.0)   # kg/s

    blk.saltMassFlow = pyo.Expression(expr=blk.nRemoved * (132.14 / 28.02))
    blk.h2so4Left = pyo.Expression(expr=blk.h2so4MassFlow - (blk.nRemoved / nPerH2SO4))

    # -------------------- Ammonium sulfate solubility ceiling --------------------
    blk.productMassFlowRaw = pyo.Expression(
        expr=blk.acidMassFlow * (1.0 - blk.acidWtFracIn) + blk.h2so4Left + blk.saltMassFlow + blk.jWaterVaporMassFlow
    )
    blk.asSolubilityPer100gWater = pyo.Param(initialize=76.4, mutable=True)
    blk.wASMax = pyo.Expression(expr=blk.asSolubilityPer100gWater / (100.0 + blk.asSolubilityPer100gWater))
    blk.wNMax = pyo.Expression(expr=blk.wASMax * (2.0 * blk.molwtN / blk.molwtAmSulfate))
    blk.productMassFlowMinForSolubility = pyo.Expression(expr=blk.nRemoved / (blk.wNMax + 1e-12))
    blk.dilutionSmoothingEps = pyo.Param(initialize=1e-6, mutable=True)
    blk.productMassFlow = pyo.Expression(
        expr=0.5 * (blk.productMassFlowRaw + blk.productMassFlowMinForSolubility)
        + 0.5 * pyo.sqrt((blk.productMassFlowRaw - blk.productMassFlowMinForSolubility) ** 2 + blk.dilutionSmoothingEps ** 2)
    )
    blk.extraDilutionWaterNeeded = pyo.Expression(expr=blk.productMassFlow - blk.productMassFlowRaw)
    blk.nWtPercent = pyo.Expression(expr=100.0 * blk.nRemoved / (blk.productMassFlow + 1e-9))

    # -------------------- Water flux closure --------------------
    # Feed: dilute, ideal van 't Hoff on TAN
    blk.feedTotalSoluteConc = pyo.Expression(expr=blk.tanInMolar)   # mol/m3
    blk.osmoticPressureFeed = pyo.Expression(
        expr=blk.osmoticCoefficient * blk.feedTotalSoluteConc * blk.gasConstR * blk.operatingTempK
    )  # Pa
    # Draw inle
    blk.awAcidB = pyo.Param(initialize=-0.118, mutable=True)
    blk.awAcidC = pyo.Param(initialize=-2.34, mutable=True)
    blk.awDrawIn = pyo.Expression(expr=1.0 + blk.awAcidB * blk.acidWtFracIn + blk.awAcidC * blk.acidWtFracIn ** 2)
    blk.awAsC1 = pyo.Param(initialize=-2.715e-3, mutable=True)
    blk.awAsC2 = pyo.Param(initialize=3.113e-5, mutable=True)
    blk.awAsC3 = pyo.Param(initialize=-2.336e-6, mutable=True)
    blk.awAsC4 = pyo.Param(initialize=1.412e-8, mutable=True)
    blk.productAsWtPercent = pyo.Expression(expr=100.0 * blk.saltMassFlow / (blk.productMassFlow + 1e-12))
    blk.awDrawOut = pyo.Expression(
        expr=1.0 + blk.awAsC1 * blk.productAsWtPercent + blk.awAsC2 * blk.productAsWtPercent ** 2
        + blk.awAsC3 * blk.productAsWtPercent ** 3 + blk.awAsC4 * blk.productAsWtPercent ** 4
    )
    blk.osmoticPressureDrawIn = pyo.Expression(
        expr=-(blk.gasConstR * blk.operatingTempK / blk.waterMolarVolume) * pyo.log(blk.awDrawIn)
    )  # Pa
    blk.osmoticPressureDrawOut = pyo.Expression(
        expr=-(blk.gasConstR * blk.operatingTempK / blk.waterMolarVolume) * pyo.log(blk.awDrawOut)
    )  # Pa
    # drawInletWeight
    blk.drawInletWeight = pyo.Param(initialize=0.0, mutable=True)
    blk.osmoticPressureDraw = pyo.Expression(
        expr=blk.drawInletWeight * blk.osmoticPressureDrawIn + (1.0 - blk.drawInletWeight) * blk.osmoticPressureDrawOut
    )
    # Written in kg/day (x 8.64e7)
    blk.waterFluxScale = pyo.Param(initialize=86400.0 * 1000.0, mutable=True)
    blk.waterFluxConstr = pyo.Constraint(
        expr=blk.waterFluxScale * blk.jWaterVapor
        == blk.waterFluxScale * blk.waterPermeability * blk.area * (blk.osmoticPressureDraw - blk.osmoticPressureFeed)
    )
    blk.waterFluxKgM2h = pyo.Expression(expr=blk.jWaterVaporMassFlow * 3600.0 / blk.area)   # reporting

    # -------------------- Outlet streams --------------------
    blk.raffinateLiquid = pyo.Constraint(expr=raff.flow['liquid'] == sIn.flow['liquid'] - blk.jWaterVaporMassFlow)
    blk.raffinateTan = pyo.Constraint(expr=raff.flow['tan'] == sIn.flow['tan'] - blk.nRemoved)
    streamTools.passComponents(blk, 'raffinatePass', sIn, raff, streamTools.componentsExcept('liquid', 'tan'))
    blk.raffinatePH = pyo.Constraint(expr=raff.pH == blk.pHOut)

    blk.productLiquid = pyo.Constraint(expr=prod.flow['liquid'] == blk.productMassFlow)
    blk.productTan = pyo.Constraint(expr=prod.flow['tan'] == blk.nRemoved)
    blk.productOther = pyo.Constraint(
        streamTools.componentsExcept('liquid', 'tan'),
        rule=lambda b, c: prod.flow[c] == 0.0
    )

    # -------------------- Product pH (live charge balance, mol/m3) --------------------
    # [H+] + [NH4+] = [OH-] + [HSO4-] + 2[SO4 2-]; first H2SO4 dissociation complete.
    blk.productLiquidVol = pyo.Expression(expr=blk.productMassFlow / blk.productDensity)          # m3/s
    blk.productNTotal = pyo.Expression(expr=(blk.nRemoved / blk.molwtN) / (blk.productLiquidVol + 1e-12))       # mol/m3
    blk.productSTotal = pyo.Expression(expr=(blk.h2so4MassFlow / blk.molwtH2SO4) / (blk.productLiquidVol + 1e-12))  # mol/m3
    blk.kaNH4M3 = pyo.Expression(expr=10 ** (-blk.pKa) * 1000.0)       # mol/m3
    blk.kaHSO4M3 = pyo.Expression(expr=10 ** (-blk.pKaHSO4) * 1000.0)  # mol/m3

    blk.hProd = pyo.Var(initialize=5e-3, bounds=(1e-11, 1e5), within=pyo.NonNegativeReals)   # mol/m3
    blk.ohProd = pyo.Var(initialize=2e-6, bounds=(1e-13, 1e5), within=pyo.NonNegativeReals)  # mol/m3
    blk.productPhDef = pyo.Constraint(expr=10 ** (3 - prod.pH) == blk.hProd)
    blk.productWaterEq = pyo.Constraint(expr=blk.hProd * blk.ohProd == 1e-8)

    blk.nh4Prod = pyo.Expression(expr=blk.productNTotal * blk.hProd / (blk.hProd + blk.kaNH4M3))
    blk.hso4Prod = pyo.Expression(expr=blk.productSTotal * blk.hProd / (blk.hProd + blk.kaHSO4M3))
    blk.so4Prod = pyo.Expression(expr=blk.productSTotal * blk.kaHSO4M3 / (blk.hProd + blk.kaHSO4M3))
    blk.productChargeBalance = pyo.Constraint(
        expr=blk.hProd + blk.nh4Prod == blk.ohProd + blk.hso4Prod + 2.0 * blk.so4Prod
    )
    blk.productPH = pyo.Expression(expr=prod.pH)

    # -------------------- Product pH estimate (weak-acid NH4+ only, cross-check) --------------------
    blk.nWtFrac = pyo.Expression(expr=blk.nWtPercent / 100.0)
    blk.nh4ConcMolL = pyo.Expression(expr=(blk.nWtFrac * blk.productDensity) / (blk.molwtN * 1000.0))
    blk.kaNH4 = pyo.Expression(expr=10 ** (-blk.pKa))
    blk.hProdMolL = pyo.Expression(
        expr=0.5 * (-blk.kaNH4 + pyo.sqrt(blk.kaNH4 ** 2 + 4.0 * blk.kaNH4 * blk.nh4ConcMolL))
    )
    blk.productPHEst = pyo.Expression(expr=-pyo.log(blk.hProdMolL + 1e-16) / pyo.log(10.0))

    # -------------------- Pump --------------------
    blk.fLam = pyo.Expression(expr=64.0 / (blk.Re + 1e-9))
    blk.fTurb = pyo.Expression(expr=0.3164 * (blk.Re + 1e-9) ** (-0.25))
    blk.fBlend = pyo.Expression(expr=1.0 / (1.0 + pyo.exp(-(blk.Re - 3000.0) / 400.0)))
    blk.fricFactor = pyo.Expression(expr=(1.0 - blk.fBlend) * blk.fLam + blk.fBlend * blk.fTurb)
    blk.dynamicHead = pyo.Expression(expr=(blk.inletDensity * blk.velocity ** 2) / 2.0)
    blk.deltaP = pyo.Expression(
        expr=(blk.fricFactor * (blk.effLength / (blk.hydDiameter + 1e-12)) + blk.minorK * blk.modulesInSeries) * blk.dynamicHead
        + blk.staticDP
    )
    blk.pumpPower = pyo.Expression(expr=(blk.loopFlow * blk.deltaP) / (blk.pumpEff * 1000.0))   # kW

    # -------------------- Economics --------------------
    # Purchased cost of the membrane modules ($125/m2 at the 130 m2 reference, scaled with exponent 0.7)
    blk.purchaseCost = pyo.Expression(
        expr=blk.costReference * (blk.area / blk.areaReference) ** blk.costExponent
    )
    blk.bareModuleCost = pyo.Expression(expr=blk.bareModuleFactor * blk.purchaseCost)
    blk.capex = pyo.Expression(expr=blk.bareModuleCost)   # bare-module cost, $
    blk.totalAcidCost = pyo.Expression(expr=blk.acidMassFlow * blk.acidCost * m.daysOperation)
    blk.pumpOpex = pyo.Expression(expr=blk.pumpPower * m.elecPrice * (m.daysOperation / 3600.0))
    blk.projectYears = pyo.Expression(expr=m.daysOperation / (365.0 * 24.0 * 3600.0))
    blk.membraneReplOpex = pyo.Expression(expr=blk.membraneReplFrac * blk.purchaseCost * blk.projectYears)
    blk.opex = pyo.Expression(expr=blk.totalAcidCost + blk.pumpOpex + blk.membraneReplOpex)

    return blk