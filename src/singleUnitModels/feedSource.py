#------------------------------------------------------------------------------
# function:     feedSource.py                                                 #
# Description:  Fixed-composition feed stream (e.g. dewatered sludge cake or  #
#               centrate). All specifications are mutable Params, so the      #
#               flowsheet can change them without touching the equations.    #
#                                                                             #
#               Solid-bound N/P/K are fractions of the ORGANIC dry solids     #
#               (never of total solids, which can include CaO).               #
#               Dissolved species are concentrations in the LIQUID phase,     #
#               converted to volume with the liquid-phase density.            #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#               - blockName : name of the new block                           #
#               - keyword specifications (see signature)                      #
#                                                                             #
# Output:       - m.<blockName> with an `outlet` stream                       #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import streamTools
except ImportError:
    import streamTools


def feedSource(m, blockName, volFlowM3s, density, tss,
               tanConcGm3=0.0, orgNConcGm3=0.0,
               solidsNFrac=0.0, solidsPFrac=0.0, solidsKFrac=0.0,
               liqPConcMgL=0.0, liqKConcMgL=0.0, caConcKgM3=0.0, mgConcKgM3=0.0,
               pH=7.0):

    blk = pyo.Block()
    m.add_component(blockName, blk)

    blk.volFlowM3s  = pyo.Param(initialize=volFlowM3s, mutable=True)   # m3/s, bulk
    blk.density     = pyo.Param(initialize=density, mutable=True)      # kg/m3, bulk
    blk.tss         = pyo.Param(initialize=tss, mutable=True)          # mass fraction
    blk.tanConcGm3  = pyo.Param(initialize=tanConcGm3, mutable=True)   # g-N/m3 liquid
    blk.orgNConcGm3 = pyo.Param(initialize=orgNConcGm3, mutable=True)  # g-N/m3 liquid
    blk.solidsNFrac = pyo.Param(initialize=solidsNFrac, mutable=True)  # kg-N/kg organic dry solids
    blk.solidsPFrac = pyo.Param(initialize=solidsPFrac, mutable=True)  # kg-P/kg organic dry solids
    blk.solidsKFrac = pyo.Param(initialize=solidsKFrac, mutable=True)  # kg-K/kg organic dry solids
    blk.liqPConcMgL = pyo.Param(initialize=liqPConcMgL, mutable=True)  # mg-P/L liquid
    blk.liqKConcMgL = pyo.Param(initialize=liqKConcMgL, mutable=True)  # mg-K/L liquid
    blk.caConcKgM3  = pyo.Param(initialize=caConcKgM3, mutable=True)   # kg/m3 liquid
    blk.mgConcKgM3  = pyo.Param(initialize=mgConcKgM3, mutable=True)   # kg/m3 liquid
    blk.pHSpec      = pyo.Param(initialize=pH, mutable=True)

    blk.totalMassFlow = pyo.Expression(expr=blk.volFlowM3s * blk.density)        # kg/s
    blk.drySolidsFlow = pyo.Expression(expr=blk.totalMassFlow * blk.tss)          # kg/s
    blk.liquidMassFlow = pyo.Expression(expr=blk.totalMassFlow * (1.0 - blk.tss)) # kg/s
    blk.liquidVolFlow = pyo.Expression(expr=blk.liquidMassFlow / m.liquidDensity) # m3/s

    streamTools.addStream(blk, 'outlet', initPH=pH)
    out = blk.outlet

    spec = {
        'liquid':    blk.liquidMassFlow,
        'orgSolids': blk.drySolidsFlow,
        'caoSolids': 0.0,
        'solidN':    blk.solidsNFrac * blk.drySolidsFlow,
        'solidP':    blk.solidsPFrac * blk.drySolidsFlow,
        'solidK':    blk.solidsKFrac * blk.drySolidsFlow,
        'tan':       blk.tanConcGm3 * blk.liquidVolFlow / 1000.0,
        'orgN':      blk.orgNConcGm3 * blk.liquidVolFlow / 1000.0,
        'liqP':      blk.liqPConcMgL * blk.liquidVolFlow / 1000.0,   # mg/L == g/m3
        'liqK':      blk.liqKConcMgL * blk.liquidVolFlow / 1000.0,
        'ca':        blk.caConcKgM3 * blk.liquidVolFlow,
        'mg':        blk.mgConcKgM3 * blk.liquidVolFlow,
    }

    def _specRule(b, c):
        return out.flow[c] == spec[c]
    blk.specConstr = pyo.Constraint(m.streamComponents, rule=_specRule)
    blk.pHConstr = pyo.Constraint(expr=out.pH == blk.pHSpec)

    return blk
