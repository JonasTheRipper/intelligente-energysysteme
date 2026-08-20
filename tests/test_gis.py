"""Unit tests for the pure-numpy real-DEM helpers in ``wildfire_cma.gis``.

These cover the SRTM/OpenTopography loader path added for real-terrain runs:

* ``_resample_bilinear`` / ``_resample_nearest`` -- pure-numpy resampling onto
  the model grid (no scipy / rasterio).
* ``_fuel_from_dem`` -- elevation-banded fuel derivation (ocean and alpine are
  non-burnable; the SoCal grass -> chaparral -> timber gradient in between).
* ``socal_from_srtm`` -- loads the cached DEM mosaic if present, else raises
  ``FileNotFoundError`` so callers can fall back to ``synthetic_socal``.

The module imports only numpy + ``wildfire_cma.gis`` (no pandapower / palaestrai
/ mosaik), so it stays in the fast ``unit`` CI stage. The ``socal_from_srtm``
test skips gracefully when the 71 MB DEM cache is not present (e.g. in CI).
"""

import numpy as np
import pytest

from wildfire_cma import gis, BASE_ROS_BY_FUEL


def test_resample_bilinear_shape_and_range():
    src = np.arange(36, dtype=float).reshape(6, 6)
    out = gis._resample_bilinear(src, (12, 12))
    assert out.shape == (12, 12)
    # bilinear interpolation must stay within the source value range
    assert out.min() >= src.min() - 1e-9
    assert out.max() <= src.max() + 1e-9
    # corners are preserved exactly
    assert out[0, 0] == pytest.approx(src[0, 0])
    assert out[-1, -1] == pytest.approx(src[-1, -1])


def test_resample_nearest_shape_and_values():
    src = np.array([[1, 2], [3, 4]], dtype=np.int16)
    out = gis._resample_nearest(src, (4, 4))
    assert out.shape == (4, 4)
    # nearest-neighbour only ever returns values from the source set
    assert set(np.unique(out)).issubset(set(np.unique(src)))


def test_fuel_from_dem_banding():
    # one row spanning ocean -> valley -> foothill -> chaparral -> montane ->
    # high timber -> alpine
    dem = np.array([[-50.0, 50.0, 300.0, 800.0, 1500.0, 2200.0, 3000.0]])
    fuel = gis._fuel_from_dem(dem, seed=7)
    assert fuel.shape == dem.shape
    assert fuel.dtype == np.int16
    # ocean (<=0) and alpine (>=2800) are non-burnable
    assert fuel[0, 0] == 0
    assert fuel[0, -1] == 0
    # the burnable interior carries a coarse grass->timber gradient
    assert fuel[0, 1] == 1  # valley grass
    assert fuel[0, 2] == 2  # foothill grass-shrub
    assert fuel[0, 3] == 3  # chaparral
    assert fuel[0, 4] == 4  # montane timber-understory
    assert fuel[0, 5] == 5  # high timber-litter


def test_fuel_from_dem_deterministic():
    rng = np.random.default_rng(0)
    dem = rng.uniform(-100, 3200, size=(40, 40))
    a = gis._fuel_from_dem(dem, seed=7)
    b = gis._fuel_from_dem(dem, seed=7)
    assert np.array_equal(a, b)


def test_socal_from_srtm_missing_cache_raises(tmp_path):
    missing = str(tmp_path / "does_not_exist.npz")
    with pytest.raises(FileNotFoundError):
        gis.socal_from_srtm(nrows=20, ncols=24, dem_npz=missing)


def test_socal_from_srtm_real_cache_if_present():
    """If the real DEM mosaic is cached, the loader builds a sane RasterStack."""
    import os

    if not os.path.exists(gis._DEM_NPZ):
        pytest.skip(
            "real SRTM DEM cache not present (regenerate via "
            "data/dem/fetch_dem_tiles.py)"
        )
    rs = gis.socal_from_srtm(nrows=30, ncols=40)
    assert rs.fuel.shape == (30, 40)
    assert rs.dem.shape == (30, 40)
    assert rs.delta_m > 0
    # fuel classes are within the BASE_ROS_BY_FUEL set
    assert set(np.unique(rs.fuel)) <= set(BASE_ROS_BY_FUEL)
    # SoCal spans sea level to high mountains
    assert rs.dem.min() <= 0
    assert rs.dem.max() > 1000
