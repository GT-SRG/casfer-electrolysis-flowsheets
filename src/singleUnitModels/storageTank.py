#------------------------------------------------------------------------------
# function:     storageTank.py                                                #
# Description:  Product storage tank, sized for storageTimeHrs of product.    #
#               Capital cost only. Pure pass-through of every component.      #
#                                                                             #
#               Streams: inlet -> outlet                                      #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#               - blockName : st                                              #
#                                                                             #
# Output:       - m.st                                                        #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import getParams
    from . import streamTools
except ImportError:
    import getParams
    import streamTools


def storageTank(m, blockName='st'):

    blk = pyo.Block()
    m.add_component(blockName, blk)

    storageTankParams = getParams.params.get('Storage Tank', {})

    def _getParam(key, default):
        try:
            return float(storageTankParams.get(key, default))
        except Exception:
            return default

    blk.costReference   = pyo.Param(initialize=_getParam('Cost Reference', 254842.0))    # $
    blk.volumeReference = pyo.Param(initialize=_getParam('Volume Reference', 249.83718))  # m3
    blk.capexFactor     = pyo.Param(initialize=_getParam('Capex Factor', 0.6))
    blk.storageTimeHrs  = pyo.Param(initialize=48.0, mutable=True)
    blk.minCapex        = pyo.Param(initialize=5000.0, mutable=True)
    blk.productDensity  = pyo.Param(initialize=_getParam('Product Density', 1200.0), mutable=True)  # kg/m3, sizing only

    streamTools.addStream(blk, 'inlet')
    streamTools.addStream(blk, 'outlet')
    streamTools.connectStreams(blk, 'passBalance', blk.inlet, blk.outlet)

    blk.productMassFlowIn = pyo.Expression(expr=blk.inlet.totalMass)     # kg/s
    blk.productMassFlowOut = pyo.Expression(expr=blk.outlet.totalMass)   # kg/s

    blk.tankVolume = pyo.Var(initialize=100.0, within=pyo.NonNegativeReals)  # m3
    blk.storageTankVolume = pyo.Constraint(
        expr=blk.tankVolume == blk.storageTimeHrs * 3600.0 * (blk.productMassFlowIn / blk.productDensity)
    )

    blk.capex = pyo.Expression(
        expr=blk.minCapex + 1.64 * blk.costReference * (blk.tankVolume / blk.volumeReference) ** blk.capexFactor
    )
    blk.opex = pyo.Expression(expr=0.0)

    return blk
