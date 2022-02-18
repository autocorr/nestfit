#!/usr/bin/env python3

import pytest
import numpy as np

from . import (DATA_PATH, NH3_RMS_K, get_ammonia_cube)
from nestfit.main import (
        NoiseMap, NoiseMapUniform, DataCube, CubeStack
)


@pytest.fixture
def nmap():
    return NoiseMapUniform(NH3_RMS_K)


@pytest.fixture
def dcube(nmap):
    trans_id = 1
    cube = get_ammonia_cube(trans_id=trans_id)
    return DataCube(cube, nmap, trans_id=trans_id)


@pytest.fixture
def stack():
    return CubeStack([
            DataCube(get_ammonia_cube(trans_id=1), NH3_RMS_K),
            DataCube(get_ammonia_cube(trans_id=2), NH3_RMS_K),
    ])


def test_noise_map_uniform(nmap):
    rms = nmap.rms
    assert nmap.get_noise(1, 1) == rms
    assert nmap.shape is None


class TestDataCube:
    def test_read(self):
        cube = get_ammonia_cube(trans_id=1)
        assert DataCube(cube, NH3_RMS_K, trans_id=1)

    def test_properties(self, dcube):
        assert dcube.trans_id == 1
        assert dcube.dv
        assert dcube.shape == (20, 20, 379)
        assert dcube.spatial_shape == (20, 20)
        assert dcube.nchan == 379
        assert dcube.full_header
        assert dcube.simple_header
        xarr, arr, noise, trans_id, has_nans = dcube.get_spec_data(1, 1)
        assert not has_nans
        assert xarr[1] > xarr[0]  # ascending
        assert not np.any(np.isnan(arr))
        assert not np.isnan(noise)


class TestCubeStack:
    def test_properties(self, stack):
        assert stack.full_header
        assert stack.simple_header
        assert stack.shape == (20, 20, 379)
        assert stack.spatial_shape == (20, 20)

    def test_get_arrays(self, stack):
        all_spec_data, any_nans = stack.get_spec_data(1, 1)
        assert not any_nans
        assert all_spec_data

    def test_get_max_snr(self, stack):
        assert stack.get_max_snr(1, 1) > 0


