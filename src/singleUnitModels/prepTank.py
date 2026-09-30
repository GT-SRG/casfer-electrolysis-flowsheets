#------------------------------------------------------------------------------
# function:     prepTank.py                                                   #
# Description:  Preparation tank: blends dewatered sludge and centrate,       #
#               doses CaO to pH 13, optional dilution water.                 #
#                                                                             #
#               Streams: sludgeIn, centrateIn -> outlet                       #
#                                                                             #
#               CaO dose (two-part):                                          #
#                 sludge  : caoPerKgDS x organic dry solids of sludgeIn       #
#                 centrate: (TAN + residual OH- + alkalinity buffer) / 2      #
#               CaO split: baseSolubility x outlet liquid volume dissolves    #
#               (reports to liquid mass), the rest stays as caoSolids.        #
#               Dissolved Ca/Mg released on conditioning (origin-based):      #
#                 Ca = dissolvedCaPerKgCaO x total CaO                        #
#                 Mg = dissolvedMgPerKgDS  x organic dry solids of sludgeIn   #
#               Dissolved P co-precipitates with Ca at pH 13 and moves to     #
#               solidP (pPrecipitationFrac, default 1.0).                     #
#               N species are not transformed here.                           #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m.pt                                                        #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import getParams
    from . import streamTools
except ImportError:
    import getParams
    import streamTools


def prepTank(m):

    m.pt = pyo.Block()
    blk = m.pt

    prepTankParams = getParams.params['Preparation Tank']

    blk.residenceTime           = pyo.Param(initialize=prepTankParams['Residence Time'])        # s
    blk.costReference           = pyo.Param(initialize=prepTankParams['Cost Reference'])        # $
    blk.volumeReference         = pyo.Param(initialize=prepTankParams['Volume Reference'])      # m3
    blk.capexFactor             = pyo.Param(initialize=prepTankParams['Capex Factor'])
    blk.baseCost                = pyo.Param(initialize=prepTankParams['Base Cost'])             # $/kg CaO
    blk.baseDensity             = pyo.Param(initialize=prepTankParams['Base Density'])          # kg/m3
    blk.baseSolubility          = pyo.Param(initialize=prepTankParams['Base Solubility'])       # kg/m3
    blk.limeTankCostReference   = pyo.Param(initialize=prepTankParams['Lime Tank Cost'])        # $
    blk.limeTankVolumeReference = pyo.Param(initialize=prepTankParams['Lime Tank Volume'])      # m3
    blk.limeTankCapexFactor     = pyo.Param(initialize=prepTankParams['Lime Tank Capex Factor'])
    blk.sludgeTSSOutTarget      = pyo.Param(initialize=prepTankParams['Target TSS'], mutable=True)

    # CaO dosing basis
    blk.caoPerKgDS       = pyo.Param(initialize=0.224, mutable=True)   # kg CaO / kg organic dry solids
    blk.alkalinityBuffer = pyo.Param(initialize=20.0, mutable=True)    # mol OH- / m3 centrate
    blk.targetOHConc     = pyo.Param(initialize=100.0, mutable=True)   # mol OH- / m3 at pH 13
    blk.mwCaO            = pyo.Param(initialize=0.05608)               # kg/mol
    blk.targetpH         = pyo.Param(initialize=13.0, mutable=True)

    # Dissolved species released on conditioning
    blk.dissolvedCaPerKgCaO = pyo.Param(initialize=0.0468, mutable=True)   # kg-Ca / kg-CaO
    blk.dissolvedMgPerKgDS  = pyo.Param(initialize=0.00350, mutable=True)  # kg-Mg / kg organic DS
    blk.pPrecipitationFrac  = pyo.Param(initialize=1.0, mutable=True)      # fraction of dissolved P -> solidP

    blk.minCapex         = pyo.Param(initialize=10000.0, mutable=True)
    blk.minLimeTankCapex = pyo.Param(initialize=5000.0, mutable=True)

    # -------------------- Streams --------------------
    streamTools.addStream(blk, 'sludgeIn')
    streamTools.addStream(blk, 'centrateIn')
    streamTools.addStream(blk, 'outlet', initPH=13.0)
    sIn, cIn, out = blk.sludgeIn, blk.centrateIn, blk.outlet

    blk.waterMassFlowIn = pyo.Var(initialize=0.0, within=pyo.NonNegativeReals)  # kg/s dilution water (decision)
    blk.tankVolume      = pyo.Var(initialize=50.0, within=pyo.NonNegativeReals)  # m3
    blk.limeTankVolume  = pyo.Var(initialize=20.0, within=pyo.NonNegativeReals)  # m3

    # -------------------- CaO dose --------------------
    blk.sludgeDrySolidsFlow = pyo.Expression(expr=sIn.flow['orgSolids'])                   # kg/s
    blk.caoForSludge = pyo.Expression(expr=blk.caoPerKgDS * blk.sludgeDrySolidsFlow)       # kg/s

    blk.centrateOhDemandTan = pyo.Expression(expr=cIn.flow['tan'] / streamTools.molwtN)   # mol/s
    blk.centrateOhDemandResidual = pyo.Expression(expr=blk.targetOHConc * cIn.liquidVol)    # mol/s
    blk.centrateOhDemandAlkalinity = pyo.Expression(expr=blk.alkalinityBuffer * cIn.liquidVol)  # mol/s
    blk.centrateTotalOhDemand = pyo.Expression(
        expr=blk.centrateOhDemandTan + blk.centrateOhDemandResidual + blk.centrateOhDemandAlkalinity
    )
    blk.caoForCentrate = pyo.Expression(expr=(blk.centrateTotalOhDemand / 2.0) * blk.mwCaO)  # kg/s, 1 CaO -> 2 OH-

    blk.totalCaO = pyo.Expression(expr=blk.caoForSludge + blk.caoForCentrate)  # kg/s

    # Dissolved CaO
    blk.dissolvedCaO = pyo.Expression(expr=blk.baseSolubility * out.liquidVol)  # kg/s
    blk.undissolvedCaO = pyo.Expression(expr=blk.totalCaO - blk.dissolvedCaO)    # kg/s

    # -------------------- Component balances --------------------
    def _mixed(c):
        return sIn.flow[c] + cIn.flow[c]

    blk.liquidBalance = pyo.Constraint(
        expr=out.flow['liquid'] == _mixed('liquid') + blk.waterMassFlowIn + blk.dissolvedCaO
    )
    blk.orgSolidsBalance = pyo.Constraint(expr=out.flow['orgSolids'] == _mixed('orgSolids'))
    blk.caoSolidsBalance = pyo.Constraint(expr=out.flow['caoSolids'] == _mixed('caoSolids') + blk.undissolvedCaO)

    blk.solidNBalance = pyo.Constraint(expr=out.flow['solidN'] == _mixed('solidN'))
    blk.solidKBalance = pyo.Constraint(expr=out.flow['solidK'] == _mixed('solidK'))
    blk.solidPBalance = pyo.Constraint(
        expr=out.flow['solidP'] == _mixed('solidP') + blk.pPrecipitationFrac * _mixed('liqP')
    )
    blk.liqPBalance = pyo.Constraint(expr=out.flow['liqP'] == (1.0 - blk.pPrecipitationFrac) * _mixed('liqP'))

    blk.tanBalance  = pyo.Constraint(expr=out.flow['tan'] == _mixed('tan'))
    blk.orgNBalance = pyo.Constraint(expr=out.flow['orgN'] == _mixed('orgN'))
    blk.liqKBalance = pyo.Constraint(expr=out.flow['liqK'] == _mixed('liqK'))
    blk.caBalance   = pyo.Constraint(expr=out.flow['ca'] == _mixed('ca') + blk.dissolvedCaPerKgCaO * blk.totalCaO)
    blk.mgBalance   = pyo.Constraint(expr=out.flow['mg'] == _mixed('mg') + blk.dissolvedMgPerKgDS * blk.sludgeDrySolidsFlow)

    blk.pHConstr = pyo.Constraint(expr=out.pH == blk.targetpH)

    # Optional outlet TSS target, OFF by default
    blk.targetTSSConstr = pyo.Constraint(expr=out.solidsMass == blk.sludgeTSSOutTarget * out.totalMass)
    blk.targetTSSConstr.deactivate()

    # -------------------- Sizing --------------------
    blk.feedBulkVolFlow = pyo.Expression(expr=sIn.bulkVol + cIn.bulkVol)          # m3/s
    blk.waterVolFlowIn = pyo.Expression(expr=blk.waterMassFlowIn / 1000.0)         # m3/s
    blk.baseVolFlowIn = pyo.Expression(expr=blk.totalCaO / blk.baseDensity)        # m3/s

    blk.prepTankVolume = pyo.Constraint(
        expr=blk.tankVolume == blk.residenceTime * (blk.feedBulkVolFlow + blk.waterVolFlowIn + blk.baseVolFlowIn)
    )
    blk.limeTankVolumeConstr = pyo.Constraint(expr=blk.limeTankVolume == blk.residenceTime * blk.baseVolFlowIn)

    # -------------------- Costs --------------------
    blk.capexTank = pyo.Expression(
        expr=blk.minCapex + blk.costReference * (blk.tankVolume / blk.volumeReference) ** blk.capexFactor
    )
    blk.capexLimeTank = pyo.Expression(
        expr=blk.minLimeTankCapex
        + blk.limeTankCostReference * (blk.limeTankVolume / blk.limeTankVolumeReference) ** blk.limeTankCapexFactor
    )
    blk.capex = pyo.Expression(expr=1.64 * (blk.capexTank + blk.capexLimeTank))
    blk.opex = pyo.Expression(expr=blk.baseCost * blk.totalCaO * m.daysOperation)  # $ lifetime, CaO only

    return blk
