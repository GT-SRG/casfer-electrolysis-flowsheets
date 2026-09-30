#------------------------------------------------------------------------------
# function:     splitter.py                                                   #
# Description:  Non-selective stream splitter: every component is split by   #
#               the same fraction, pH is unchanged.                          #
#               splitFrac is the fraction sent to outlet1; the flowsheet can  #
#               fix it, bound it, or tie it to a binary (e.g. a unit bypass). #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#               - blockName : name of the new block                           #
#                                                                             #
# Output:       - m.<blockName> with `inlet`, `outlet1`, `outlet2` streams    #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import streamTools
except ImportError:
    import streamTools


def splitter(m, blockName):

    blk = pyo.Block()
    m.add_component(blockName, blk)

    streamTools.addStream(blk, 'inlet')
    streamTools.addStream(blk, 'outlet1')
    streamTools.addStream(blk, 'outlet2')

    blk.splitFrac = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals, bounds=(0.0, 1.0))

    def _split1Rule(b, c):
        return b.outlet1.flow[c] == b.splitFrac * b.inlet.flow[c]
    blk.split1 = pyo.Constraint(m.streamComponents, rule=_split1Rule)

    def _split2Rule(b, c):
        return b.outlet2.flow[c] == b.inlet.flow[c] - b.outlet1.flow[c]
    blk.split2 = pyo.Constraint(m.streamComponents, rule=_split2Rule)

    blk.pH1 = pyo.Constraint(expr=blk.outlet1.pH == blk.inlet.pH)
    blk.pH2 = pyo.Constraint(expr=blk.outlet2.pH == blk.inlet.pH)

    return blk
