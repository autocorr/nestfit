#!/usr/bin/env python3

import warnings
from pathlib import Path

from spectral_cube import SpectralCube
from astropy.wcs import FITSFixedWarning



DATA_PATH = Path(__file__).parent / "data"
NH3_RMS_K = 0.35


def get_ammonia_cube(trans_id=1):
    assert trans_id in (1, 2)
    transition = f"{trans_id}" * 2
    fpath = DATA_PATH / f"ammonia_{transition}_cutout.fits"
    # Filter WCS "obsfix" warning about multiple OBSGEO keywords
    warnings.filterwarnings(
            "ignore",
            message=R".*Set OBSGEO-. to .* from OBSGEO-\[XYZ\]",
            category=FITSFixedWarning,
    )
    cube = SpectralCube.read(str(fpath))
    cube = cube[:-1]  # last channel contains NaNs
    return cube


