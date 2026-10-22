#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# flowsheetBreakdownFigure.py
# 8-row x 1-column grid. Each flowsheet occupies two consecutive rows:
#   CapEx stacked horizontal bar (own x-axis)
#   OpEx  stacked horizontal bar (own x-axis)
# A SINGLE per-flowsheet legend hovers to the right, vertically centered
# between that flowsheet's CapEx and OpEx rows (one legend per flowsheet,
# listing the union of its units). Absolute $/ton E-GROW; CVD-safe, globally
# consistent unit colors. FS2 values are PRELIMINARY placeholders.
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# 1. DATA  (absolute $/ton; edit these dictionaries directly)
# ---------------------------------------------------------------------------
fs0CapEx = {'Centrifuge': 8.828637097824716, 'Dryer': 20.150245355942946, 'Storage Tank': 0.9135857397589863}
fs0OpEx  = {'Centrifuge': 81.52958456178463, 'Dryer': 182.0412414017419}

fs1CapEx = {'Prep Tank': 0.0640, 'Electrolyzer': 2.2099, 'Receive Tank': 0.0486,
            'Centrifuge': 2.0066, 'Dryer': 1.8293, 'Storage Tank': 0.5014}
fs1OpEx  = {'Prep Tank': 8.4585, 'Electrolyzer': 7.1311, 'Receive Tank': 3.6695,
            'Centrifuge': 25.7035, 'Dryer': 22.4800}

fs2CapEx = {'Prep Tank': 0.0654, 'Electrolyzer': 2.2601, 'Receive Tank': 0.0206,
            'GO Membrane': 0.0350, 'NF': 0.1334, 'GPM': 0.0678,
            'Storage Tank': 0.5128, 'Post-Treatment': 0.0392}
fs2OpEx  = {'Prep Tank': 8.4585, 'Electrolyzer': 14.2622, 'Receive Tank': 0.8888,
            'GO Membrane': 0.1635, 'NF': 0.2382, 'GPM': 0.7550, 'Post-Treatment': 2.1470}
# ---------------------------------------------------------------------------

fs3CapEx = {'Prep Tank': 0.0654, 'Electrolyzer': 2.2601, 'NH3Stripper':  0.1081,
            'GO Membrane': 0.0348, 'Sparger': 0.0099, 'Storage Tank': 0.5128, 'Post-Treatment': 0.0390}
fs3OpEx  = {'Prep Tank': 8.4585, 'Electrolyzer': 7.1311, 'NH3Stripper': 2.7758,
            'GO Membrane': 0.1582, 'Sparger': 22.1403, 'Post-Treatment': 1.8144}

flowsheets = [
    ('FS0',  fs0CapEx, fs0OpEx),
    ('FS1',  fs1CapEx, fs1OpEx),
    ('FS2', fs2CapEx, fs2OpEx),
    ('FS3',  fs3CapEx, fs3OpEx),
]

# ---------------------------------------------------------------------------
# 2. STYLE
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Global CVD-safe (Okabe-Ito) color per unit, so a unit is one color everywhere.
unitColors = {
    'Prep Tank':     '#0072B2',
    'Electrolyzer':  '#E69F00',
    'Centrifuge':    '#009E73',
    'Dryer':         '#D55E00',
    'Receive Tank':  '#CC79A7',
    'GO Membrane':   '#56B4E9',
    'NF':            '#F0E442',
    'GPM':           '#000000',
    'Sparger':       '#009E73',
    'Storage Tank':  '#999999',
}
def colorFor(unit):
    return unitColors.get(unit, '#BAB0AC')

# ---------------------------------------------------------------------------
# 3. FIGURE  (8 rows x 1 col; thin bars; one legend per flowsheet pair)
# ---------------------------------------------------------------------------
nRows = 2 * len(flowsheets)
fig = plt.figure(figsize=(7.2, 6.2))
gs = fig.add_gridspec(nRows, 1, hspace=1.15,
                      left=0.10, right=0.80, top=0.96, bottom=0.07)

barHeight = 0.42

def drawBar(ax, breakdown, xMax, title):
    left = 0.0
    for unit, val in breakdown.items():
        ax.barh(0, val, height=barHeight, left=left,
                color=colorFor(unit), edgecolor='white', linewidth=0.5)
        left += val
    ax.set_xlim(0, xMax)
    ax.set_ylim(-0.4, 0.4)
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='x', labelsize=14, length=2)
    ax.grid(axis='x', linestyle=':', linewidth=0.4, color='#cccccc')
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left', pad=2)

rowIdx = 0
for name, cap, op in flowsheets:
    capTot = sum(cap.values())
    opTot = sum(op.values())

    axCap = fig.add_subplot(gs[rowIdx, 0]); rowIdx += 1
    axOp = fig.add_subplot(gs[rowIdx, 0]); rowIdx += 1

    drawBar(axCap, cap, capTot * 1.10, f'{name}  CapEx  (${capTot:.1f}/ton)')
    drawBar(axOp,  op,  opTot * 1.10, f'{name}  OpEx  (${opTot:.1f}/ton)')

    # single legend for this flowsheet = union of units across CapEx + OpEx,
    # placed to the right and vertically centered between the two rows.
    units = list(cap)
    for u in op:
        if u not in units:
            units.append(u)
    handles = [Patch(facecolor=colorFor(u), edgecolor='white', label=u)
               for u in units]
    # anchor on the CapEx axis but push down so it centers over both rows
    axCap.legend(handles=handles, loc='center left',
                 bbox_to_anchor=(1.02, -0.75), fontsize=14, frameon=False,
                 handlelength=1.0, handleheight=1.0, labelspacing=0.15,
                 borderpad=0.0)

fig.text(0.45, 0.015, 'Cost of manufacturing  ($ / ton E-GROW)',
         ha='center', fontsize=14)

plt.show()

fig.savefig('flowsheetResults.pdf', format='pdf', bbox_inches='tight')
# fig.savefig('flowsheetBreakdownFigure.png', format='png', dpi=300, bbox_inches='tight')
# fig.savefig('flowsheetResults.eps', format='eps', dpi=1200)
# print('saved flowsheetBreakdownFigure.pdf and .png')