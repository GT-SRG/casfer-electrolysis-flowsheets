#------------------------------------------------------------------------------
# function:     mixer.py                                                      #
# Description:  Ideal stream mixer. Component flows add; the outlet pH is a   #
#               specification (mixing pH is not computed), default 7.5.      #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#               - blockName : name of the new block                           #
#               - inletNames : list of inlet stream names                     #
#                                                                             #
# Output:       - m.<blockName> with inlet streams and an `outlet` stream     #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import streamTools
except ImportError:
    import streamTools


def mixer(m, blockName, inletNames=('inlet1', 'inlet2'), pHOut=7.5):

    blk = pyo.Block()
    m.add_component(blockName, blk)

    inlets = [streamTools.addStream(blk, name) for name in inletNames]
    streamTools.addStream(blk, 'outlet')

    blk.pHOutSpec = pyo.Param(initialize=pHOut, mutable=True)

    def _mixRule(b, c):
        return b.outlet.flow[c] == sum(s.flow[c] for s in inlets)
    blk.mixBalance = pyo.Constraint(m.streamComponents, rule=_mixRule)
    blk.pHConstr = pyo.Constraint(expr=blk.outlet.pH == blk.pHOutSpec)

    return blk
