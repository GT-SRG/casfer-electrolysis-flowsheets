#------------------------------------------------------------------------------
# function:     flowsheetTools.py                                             #
# Description:  Shared flowsheet helpers:                                     #
#                 - addStandardFeeds: the two plant feeds (dewatered sludge   #
#                   cake + centrate) as feedSource blocks, one place for all  #
#                   feed composition assumptions                              #
#                 - addProductQuality: N/P/K wt% (wet and dry), P2O5/K2O and  #
#                   recoveries of a product stream, as Pyomo Expressions      #
#                 - printStream / printProductQuality / printBalance: common  #
#                   reporting, including a feed-to-outlet closure check for   #
#                   N, P and K                                                #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from .feedSource import feedSource
except ImportError:
    from feedSource import feedSource


def safeValue(expr):
    try:
        return pyo.value(expr)
    except Exception:
        return float('nan')


def addStandardFeeds(m, feedPH=7.5):
    """Dewatered sludge cake + centrate, Lubbock base case.

    The same liquid-phase composition is applied to the centrate and to the
    pore liquid of the sludge cake:
      TAN 750 g-N/m3; P 13 g-P/m3 (PO4 40 mg/L); K 275 g-K/m3 (250-300 mg/L);
      Ca 40 g/m3 (centrate 20-60 mg/L); Mg 7 g/m3 (centrate 4-10 mg/L).
    """
    feedLiquid = dict(
        tanConcGm3=750.0,     # g-N/m3 liquid
        liqPConcMgL=13.0,     # mg-P/L liquid (PO4 40 mg/L x 30.97/94.97)
        liqKConcMgL=275.0,    # mg-K/L liquid
        caConcKgM3=0.040,     # kg/m3 liquid (20-60 mg/L, midpoint)
        mgConcKgM3=0.007,     # kg/m3 liquid (4-10 mg/L, midpoint)
    )
    # Dewatered sludge cake: 76.5 m3/d at 1200 kg/m3, 20% TS
    feedSource(
        m, 'sludgeFeed',
        volFlowM3s=76.5 / 86400.0, density=1200.0, tss=0.20,
        solidsNFrac=0.05, solidsPFrac=0.025, solidsKFrac=0.0035,  # kg/kg organic dry solids
        pH=feedPH,
        **feedLiquid,
    )
    # Centrate: 214 m3/d, no solids
    feedSource(
        m, 'centrateFeed',
        volFlowM3s=214.0 / 86400.0, density=1000.0, tss=0.0,
        pH=feedPH,
        **feedLiquid,
    )
    feeds = (m.sludgeFeed.outlet, m.centrateFeed.outlet)
    m.feedMassFlow = pyo.Expression(expr=sum(s.totalMass for s in feeds))      # kg/s
    m.feedDrySolids = pyo.Expression(expr=sum(s.solidsMass for s in feeds))    # kg/s
    m.feedTotalN = pyo.Expression(expr=sum(s.totalN for s in feeds))           # kg-N/s
    m.feedTotalP = pyo.Expression(expr=sum(s.totalP for s in feeds))           # kg-P/s
    m.feedTotalK = pyo.Expression(expr=sum(s.totalK for s in feeds))           # kg-K/s
    m.feedTan = pyo.Expression(expr=sum(s.flow['tan'] for s in feeds))        # kg-N/s
    m.feedSolidN = pyo.Expression(expr=sum(s.flow['solidN'] for s in feeds))  # kg-N/s
    return feeds


def addProductQuality(m, product, name='quality'):
    """Expressions for N/P/K content of a product stream (wet and dry basis)."""
    blk = pyo.Block()
    m.add_component(name, blk)
    blk.wetMass = pyo.Expression(expr=product.totalMass)
    blk.dryMass = pyo.Expression(expr=product.solidsMass)
    blk.nMass = pyo.Expression(expr=product.totalN)
    blk.pMass = pyo.Expression(expr=product.totalP)
    blk.kMass = pyo.Expression(expr=product.totalK)
    blk.nWtPercentWet = pyo.Expression(expr=100.0 * blk.nMass / (blk.wetMass + 1e-12))
    blk.pWtPercentWet = pyo.Expression(expr=100.0 * blk.pMass / (blk.wetMass + 1e-12))
    blk.kWtPercentWet = pyo.Expression(expr=100.0 * blk.kMass / (blk.wetMass + 1e-12))
    blk.nWtPercentDry = pyo.Expression(expr=100.0 * blk.nMass / (blk.dryMass + 1e-12))
    blk.pWtPercentDry = pyo.Expression(expr=100.0 * blk.pMass / (blk.dryMass + 1e-12))
    blk.kWtPercentDry = pyo.Expression(expr=100.0 * blk.kMass / (blk.dryMass + 1e-12))
    blk.p2o5WtPercentWet = pyo.Expression(expr=blk.pWtPercentWet * 2.29)
    blk.k2oWtPercentWet = pyo.Expression(expr=blk.kWtPercentWet * 1.20)
    blk.p2o5WtPercentDry = pyo.Expression(expr=blk.pWtPercentDry * 2.29)
    blk.k2oWtPercentDry = pyo.Expression(expr=blk.kWtPercentDry * 1.20)
    blk.nRecovery = pyo.Expression(expr=100.0 * blk.nMass / (m.feedTotalN + 1e-12))
    blk.pRecovery = pyo.Expression(expr=100.0 * blk.pMass / (m.feedTotalP + 1e-12))
    blk.kRecovery = pyo.Expression(expr=100.0 * blk.kMass / (m.feedTotalK + 1e-12))
    blk.caMass = pyo.Expression(expr=product.totalCa)
    blk.mgMass = pyo.Expression(expr=product.totalMg)
    blk.caWtPercentDry = pyo.Expression(expr=100.0 * blk.caMass / (blk.dryMass + 1e-12))
    blk.mgWtPercentDry = pyo.Expression(expr=100.0 * blk.mgMass / (blk.dryMass + 1e-12))
    return blk


def printStream(label, stream):
    v = safeValue
    print(f'{label}:')
    print(f'  total mass (kg/day): {v(stream.totalMass) * 86400:.6g}   TSS: {v(stream.tss):.4f}   pH: {v(stream.pH):.3f}')
    print(f'  liquid / organic solids / CaO solids (kg/day): '
          f'{v(stream.flow["liquid"]) * 86400:.6g} / {v(stream.flow["orgSolids"]) * 86400:.6g} / {v(stream.flow["caoSolids"]) * 86400:.6g}')
    print(f'  N (kg-N/day): solid-bound {v(stream.flow["solidN"]) * 86400:.4g}, TAN {v(stream.flow["tan"]) * 86400:.4g}, '
          f'dissolved organic {v(stream.flow["orgN"]) * 86400:.4g}')
    print(f'  P (kg-P/day): solid {v(stream.flow["solidP"]) * 86400:.4g}, dissolved {v(stream.flow["liqP"]) * 86400:.4g};  '
          f'K (kg-K/day): solid {v(stream.flow["solidK"]) * 86400:.4g}, dissolved {v(stream.flow["liqK"]) * 86400:.4g}')
    print(f'  Ca (kg/day): solid {v(stream.flow["solidCa"]) * 86400:.4g}, dissolved {v(stream.flow["ca"]) * 86400:.4g};  '
          f'Mg (kg/day): solid {v(stream.flow["solidMg"]) * 86400:.4g}, dissolved {v(stream.flow["mg"]) * 86400:.4g}')


def printProductQuality(label, q):
    v = safeValue
    print(f'================ {label} ================')
    print(f'Product wet / dry mass (kg/day): {v(q.wetMass) * 86400:.6g} / {v(q.dryMass) * 86400:.6g}')
    print(f'N  wt% wet / dry: {v(q.nWtPercentWet):.4f} / {v(q.nWtPercentDry):.4f}   (N recovery {v(q.nRecovery):.2f}% of feed N)')
    print(f'P  wt% wet / dry: {v(q.pWtPercentWet):.4f} / {v(q.pWtPercentDry):.4f}   (P2O5 wet/dry {v(q.p2o5WtPercentWet):.4f} / {v(q.p2o5WtPercentDry):.4f}; P recovery {v(q.pRecovery):.2f}%)')
    print(f'K  wt% wet / dry: {v(q.kWtPercentWet):.4f} / {v(q.kWtPercentDry):.4f}   (K2O wet/dry {v(q.k2oWtPercentWet):.4f} / {v(q.k2oWtPercentDry):.4f}; K recovery {v(q.kRecovery):.2f}%)')
    print(f'Ca / Mg wt% dry: {v(q.caWtPercentDry):.4f} / {v(q.mgWtPercentDry):.4f}')


def printBalance(m, productStreams, nLosses, otherOutletStreams=()):
    """Feed-to-outlet closure for N, P, K (kg/day).

    productStreams     : streams counted as product
    nLosses            : list of (label, expression in kg-N/s) for N leaving as gas/loss terms
    otherOutletStreams : list of (label, stream) for liquid/solid streams leaving the plant
    """
    v = safeValue
    print('================ N / P / K BALANCE (kg/day) ================')
    nProd = sum(v(s.totalN) for s in productStreams)
    pProd = sum(v(s.totalP) for s in productStreams)
    kProd = sum(v(s.totalK) for s in productStreams)
    print(f'Feed N {v(m.feedTotalN) * 86400:.5g} (TAN {v(m.feedTan) * 86400:.5g}, solid-bound {v(m.feedSolidN) * 86400:.5g});  '
          f'feed P {v(m.feedTotalP) * 86400:.5g};  feed K {v(m.feedTotalK) * 86400:.5g}')
    print(f'Product N {nProd * 86400:.5g};  product P {pProd * 86400:.5g};  product K {kProd * 86400:.5g}')
    nOut, pOut, kOut = nProd, pProd, kProd
    for label, stream in otherOutletStreams:
        n, p, k = v(stream.totalN), v(stream.totalP), v(stream.totalK)
        nOut += n; pOut += p; kOut += k
        print(f'  {label}: N {n * 86400:.5g} (TAN {v(stream.flow["tan"]) * 86400:.4g}, organic {v(stream.flow["orgN"]) * 86400:.4g}, '
              f'solid {v(stream.flow["solidN"]) * 86400:.4g}), P {p * 86400:.4g}, K {k * 86400:.4g}')
    for label, expr in nLosses:
        n = v(expr)
        nOut += n
        print(f'  {label}: N {n * 86400:.5g}')
    print(f'Closure residual (feed - all outlets): N {(v(m.feedTotalN) - nOut) * 86400:.3e}, '
          f'P {(v(m.feedTotalP) - pOut) * 86400:.3e}, K {(v(m.feedTotalK) - kOut) * 86400:.3e}')