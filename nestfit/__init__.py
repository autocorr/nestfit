import warnings
from pathlib import Path


warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')


TEX_LABELS = [
        r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$',
        r'$T_\mathrm{rot} \ [\mathrm{K}]$',
        r'$T_\mathrm{ex} \ [\mathrm{K}]$',
        r'$\log(N_\mathrm{p}) \ [\mathrm{cm^{-2}}]$',
        r'$\sigma_\mathrm{v} \ [\mathrm{km\, s^{-1}}]$',
]

TEX_LABELS_NU = [
        r'$v_\mathrm{lsr}$',
        r'$T_\mathrm{rot}$',
        r'$T_\mathrm{ex}$',
        r'$\log(N_\mathrm{p})$',
        r'$\sigma_\mathrm{v}$',
]


def get_par_names(ncomp=None):
    if ncomp is not None:
        return [
                f'{label}{n}'
                for label in ('v', 'Tk', 'Tx', 'N', 's')
                for n in range(1, ncomp+1)
        ]
    else:
        return ['v', 'Tk', 'Tx', 'N', 's']


