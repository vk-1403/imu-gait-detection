"""Origin-like journal style (Times, closed frame, inward ticks on all sides, pure colours)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ['/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf',
          '/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf',
          '/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-italic.otf',
          '/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bolditalic.otf']:
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'Liberation Serif'], 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8,
    'axes.linewidth': 1.1, 'axes.edgecolor': 'black', 'axes.labelcolor': 'black',
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 4, 'ytick.major.size': 4, 'xtick.minor.size': 2, 'ytick.minor.size': 2,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0, 'xtick.color': 'black', 'ytick.color': 'black',
    'axes.spines.top': True, 'axes.spines.right': True, 'axes.grid': False,
    'legend.frameon': True, 'legend.edgecolor': 'black', 'legend.fancybox': False, 'legend.framealpha': 1.0,
    'lines.linewidth': 1.4, 'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
# Origin-like pure colours
BLUE, RED, MAGENTA, OLIVE, ORANGE, CYAN, VIOLET, GREEN, BLACK, GRAY = ('#0000FF', '#FF0000', '#FF00FF', '#808000', '#FF8000',
                                                                       '#008080', '#800080', '#008000', '#000000', '#808080')
GOLD, LIGHTBLUE = '#FFC000', '#9DC3E6'
