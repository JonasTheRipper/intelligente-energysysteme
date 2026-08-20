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


# ---------------------------------------------------------------------------
# Built-up ("houses", class 9) cells -- see wildfire_cma.cma.HOUSE_FUEL_CLASS
# ---------------------------------------------------------------------------
# Settlement is encoded as a burnable fuel class rather than as non-burnable
# class 0, because the January-2025 fires destroyed thousands of structures.
# These tests pin the three properties that encoding has to satisfy: houses go
# on land, both raster builders produce them, and they survive the perimeter
# containment device that zeroes every other fuel outside the footprint.

HOUSE = gis.HOUSE_FUEL_CLASS

# Pinned fuel-map digest for (synthetic_socal, 64x80, seed 7). Regenerate ONLY
# with a deliberate change to the raster builders -- a surprise mismatch means
# the RNG draw order moved and every house cell was silently relabelled.
FUEL_MAP_SHA256_64x80_SEED7 = (
    "83e7ffb5598e44688c09c7a9d03002c8aec4394e8f948e47a77a1640d5ef0c49"
)

# Classes _fuel_from_dem / synthetic_socal are allowed to emit. Deliberately
# narrower than BASE_ROS_BY_FUEL: class 6 (slash-blowdown) is reachable only
# through the real-LANDFIRE fbfm40_to_class path, so an elevation-banded
# builder producing it means a banding rule has gone wrong.
_DEM_BUILDER_CLASSES = {0, 1, 2, 3, 4, 5, HOUSE}


def test_fuel_from_dem_emits_only_its_declared_classes():
    rng = np.random.default_rng(0)
    dem = rng.uniform(-100, 3200, size=(120, 120))
    fuel = gis._fuel_from_dem(dem, seed=7)
    assert set(np.unique(fuel)) <= _DEM_BUILDER_CLASSES


def test_fuel_from_dem_puts_no_houses_in_the_water():
    """Regression: the urban scatter used to paint over the ocean mask.

    ``low = dem < 250`` also selects every below-sea-level cell, and the
    scatter runs after ``fuel[dem <= 0] = 0``. While the scatter assigned 0
    this was a no-op; assigning the burnable house class turned ~55% of all
    "houses" into floating, burnable cells that let fire cross open water.
    """
    rng = np.random.default_rng(0)
    dem = rng.uniform(-100, 3200, size=(200, 200))
    fuel = gis._fuel_from_dem(dem, seed=7)
    assert not ((fuel == HOUSE) & (dem <= 0)).any()
    # ... and the scatter still fires on land, i.e. the guard did not kill it
    assert (fuel == HOUSE).any()


def test_synthetic_socal_has_houses_and_none_in_the_ocean():
    """The synthetic fallback must not be silently house-free.

    ``use_real_dem`` degrades to this raster in the v0.1 driver and in any
    checkout without the git-ignored 71 MB DEM mosaic. If only the real-DEM
    builder emitted houses, every structure metric would read zero there with
    no error at all.
    """
    rs = gis.synthetic_socal(nrows=120, ncols=150, seed=7)
    assert (rs.fuel == HOUSE).any()
    yy, xx = np.mgrid[0:120, 0:150]
    ocean = (xx / 150 + yy / 120) < 0.18
    assert not ((rs.fuel == HOUSE) & ocean).any()
    assert set(np.unique(rs.fuel)) <= _DEM_BUILDER_CLASSES


def test_raster_builders_record_their_provenance():
    """A stored run must be able to say which fuel map it actually used."""
    assert gis.synthetic_socal(nrows=20, ncols=25, seed=7).source == "synthetic"


def test_fuel_map_is_byte_stable_for_a_fixed_seed_and_shape():
    """Guards RNG *consumption order*, not just the seed.

    Any new draw taken from ``rng`` before the urban scatter silently relabels
    every house cell while leaving the seed and the class histogram unchanged;
    plain equality against a freshly built raster would not notice.
    """
    import hashlib

    a = gis.synthetic_socal(nrows=64, ncols=80, seed=7).fuel
    b = gis.synthetic_socal(nrows=64, ncols=80, seed=7).fuel
    assert np.array_equal(a, b)
    digest = hashlib.sha256(np.ascontiguousarray(a, dtype=np.int16).tobytes())
    assert digest.hexdigest() == FUEL_MAP_SHA256_64x80_SEED7


def test_house_scatter_depends_on_grid_shape():
    """Documents a real reproducibility constraint.

    The scatter is drawn as ``rng.random(dem.shape)``, so house placement is a
    function of (seed, shape) -- changing the raster resolution relabels every
    structure. Pinned so the coupling is a stated property rather than a
    surprise when a scenario is re-gridded.
    """
    small = gis.synthetic_socal(nrows=40, ncols=40, seed=7).fuel
    large = gis.synthetic_socal(nrows=80, ncols=80, seed=7).fuel
    frac_small = float((small == HOUSE).mean())
    frac_large = float((large == HOUSE).mean())
    # similar density, different cells
    assert abs(frac_small - frac_large) < 0.02
    assert not np.array_equal(
        gis._resample_nearest(small, (80, 80)) == HOUSE, large == HOUSE
    )


def test_containment_preserves_houses_outside_the_footprint():
    """``contain_burnable_footprint`` must not erase the house layer.

    It zeroes all fuel outside the real perimeter to hold the calibrated
    extent. Since fuel class doubles as the settlement layer, a blanket zero
    would delete exactly the houses that did NOT burn -- the denominator of any
    "fraction of houses lost" metric.
    """
    pytest.importorskip("scipy")
    from wildfire_cma.wind_field import contain_burnable_footprint

    fuel = np.full((20, 20), 3, dtype=np.int16)
    fuel[2, 2] = HOUSE       # house far outside the perimeter
    fuel[10, 10] = HOUSE     # house inside the perimeter
    real_mask = np.zeros((20, 20), dtype=bool)
    real_mask[9:12, 9:12] = True

    out = contain_burnable_footprint(fuel, real_mask, margin_cells=2)

    assert out[2, 2] == HOUSE, "house outside the footprint was erased"
    assert out[10, 10] == HOUSE, "house inside the footprint was erased"
    # every non-house cell outside the dilated footprint is still zeroed
    assert out[0, 0] == 0
    assert out[19, 19] == 0
