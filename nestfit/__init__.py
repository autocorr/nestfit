import warnings
from pathlib import Path


warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')


ROOT_DIR = Path('/lustre/aoc/users/bsvoboda/temp/nestfit')
DATA_DIR = ROOT_DIR / Path('data')
PLOT_DIR = ROOT_DIR / Path('plots')

TEX_LABELS = [
        r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$',
        r'$T_\mathrm{rot} \ [\mathrm{K}]$',
        r'$T_\mathrm{ex} \ [\mathrm{K}]$',
        r'$\log(N_\mathrm{tot}) \ [\mathrm{cm^{-2}}]$',
        r'$\sigma_\mathrm{v} \ [\mathrm{km\, s^{-1}}]$',
]


def get_par_labels(ncomp):
    return [
            f'{label}{n}'
            for label in ('v', 'Tk', 'Tx', 'N', 's')
            for n in range(1, ncomp+1)
    ]


