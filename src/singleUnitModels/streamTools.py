#------------------------------------------------------------------------------
# function:     streamTools.py                                                #
# Description:  Standard material stream shared by every unit model.          #
#                                                                             #
#               A stream is a Pyomo sub-block holding component MASS flows    #
#               (kg/s) indexed over one model-wide component set, plus pH.    #
#               Every unit reads its inlet stream(s) and writes its outlet    #
#               stream(s); flowsheets only connect outlets to inlets.         #
#                                                                             #
#               Component basis (all kg/s):                                   #
#                 Bulk mass (these three sum to the total stream mass):       #
#                   liquid     : liquid phase (water + dissolved solutes)     #
#                   orgSolids  : organic sludge solids                        #
#                   caoSolids  : undissolved CaO solids                       #
#                 Solid-bound tracers (inside orgSolids/caoSolids mass,       #
#                 travel with the solids):                                    #
#                   solidN     : kg-N/s bound to the solids                   #
#                   solidP     : kg-P/s in the solids (incl. Ca-precipitated) #
#                   solidK     : kg-K/s bound to the solids                   #
#                 Dissolved tracers (inside liquid mass, travel with liquid): #
#                   tan        : kg-N/s total ammonia nitrogen (NH3 + NH4+)   #
#                   orgN       : kg-N/s dissolved organic N (not strippable,  #
#                                not protonatable in the acid/CO2 demand)     #
#                   liqP       : kg-P/s dissolved P                           #
#                   liqK       : kg-K/s dissolved K                           #
#                   ca         : kg/s dissolved Ca                            #
#                   mg         : kg/s dissolved Mg                            #
#                                                                             #
#               Tracer flows are NOT added to the stream mass: they are       #
#               already part of the liquid or solids mass they sit in.        #
#------------------------------------------------------------------------------

import pyomo.environ as pyo

bulkComponents = ['liquid', 'orgSolids', 'caoSolids']
solidPhaseComponents = ['orgSolids', 'caoSolids']
solidBoundComponents = ['solidN', 'solidP', 'solidK']
dissolvedComponents = ['tan', 'orgN', 'liqP', 'liqK', 'ca', 'mg']
allComponents = bulkComponents + solidBoundComponents + dissolvedComponents

# Everything that follows the solids through a physical separation
solidsFollowing = solidPhaseComponents + solidBoundComponents
# Everything that follows the liquid through a physical separation
liquidFollowing = ['liquid'] + dissolvedComponents

molwtN = 0.014007    # kg/mol


def ensureComponentSet(model):
    if not hasattr(model, 'streamComponents'):
        model.streamComponents = pyo.Set(initialize=allComponents, ordered=True)
    return model.streamComponents


def addStream(parent, name, initFlow=None, initPH=7.0):
    """Create a stream sub-block `parent.<name>` and return it."""
    model = parent.model()
    comps = ensureComponentSet(model)
    initFlow = initFlow or {}

    stream = pyo.Block()
    parent.add_component(name, stream)

    def _init(b, c):
        return initFlow.get(c, 0.1 if c in bulkComponents else 1e-3)
    stream.flow = pyo.Var(comps, within=pyo.NonNegativeReals, initialize=_init)
    stream.pH = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.0, 14.0), initialize=initPH)

    stream.solidsMass = pyo.Expression(expr=sum(stream.flow[c] for c in solidPhaseComponents))  # kg/s
    stream.totalMass = pyo.Expression(expr=stream.flow['liquid'] + stream.solidsMass)           # kg/s
    stream.liquidVol = pyo.Expression(expr=stream.flow['liquid'] / model.liquidDensity)          # m3/s, liquid phase only
    stream.bulkVol = pyo.Expression(expr=stream.totalMass / model.sludgeDensity)                 # m3/s, sizing only
    stream.totalN = pyo.Expression(expr=stream.flow['solidN'] + stream.flow['tan'] + stream.flow['orgN'])  # kg-N/s
    stream.totalP = pyo.Expression(expr=stream.flow['solidP'] + stream.flow['liqP'])            # kg-P/s
    stream.totalK = pyo.Expression(expr=stream.flow['solidK'] + stream.flow['liqK'])            # kg-K/s

    # Reporting only (ratios): do not use inside constraints
    stream.tss = pyo.Expression(expr=stream.solidsMass / (stream.totalMass + 1e-12))
    stream.tanConcGm3 = pyo.Expression(expr=1000.0 * stream.flow['tan'] / (stream.liquidVol + 1e-12))    # g-N/m3 liquid
    stream.orgNConcGm3 = pyo.Expression(expr=1000.0 * stream.flow['orgN'] / (stream.liquidVol + 1e-12))  # g-N/m3 liquid
    return stream


def connectStreams(parent, name, source, destination, includePH=True):
    """Equate every component flow (and pH) of `destination` to `source`."""
    comps = ensureComponentSet(parent.model())

    def _rule(b, c):
        return destination.flow[c] == source.flow[c]
    parent.add_component(name, pyo.Constraint(comps, rule=_rule))
    if includePH:
        parent.add_component(name + 'PH', pyo.Constraint(expr=destination.pH == source.pH))


def passComponents(parent, name, source, destination, components):
    """destination.flow[c] == source.flow[c] for the listed components."""
    def _rule(b, c):
        return destination.flow[c] == source.flow[c]
    parent.add_component(name, pyo.Constraint(components, rule=_rule))


def fixStreamToZero(stream):
    """Fix every component flow of an unused stream to zero."""
    for c in stream.flow:
        stream.flow[c].fix(0.0)
