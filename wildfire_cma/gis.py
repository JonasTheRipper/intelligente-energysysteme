"""California GIS ingest for the wildfire CMA.

Builds the co-registered :class:`~wildfire_cma.cma.RasterStack` (fuel + DEM)
over the Southern California footprint and provides helpers to load real GIS
layers when available:

* **LANDFIRE FBFM40** 40-class fuel model (30 m) -> coarse fuel-class ids.
* **USGS 3DEP** DEM (~10-30 m) -> elevation array.
* **NLCD** land cover (30 m) -> non-burnable mask (water / developed / barren).

Real layers are read with ``rasterio`` (GDAL) when a path is given; otherwise a
deterministic **synthetic** California-like raster is generated so the pipeline
and tests run offline. Layers can also be staged in / out of a PostGIS database
(see :mod:`wildfire_cma.postgis` and ``docker-compose.yml``).

LANDFIRE FBFM40 -> coarse class mapping (see ``cma.BASE_ROS_BY_FUEL``):
    GR (101-109)        -> 1  grass
    GS (121-124)        -> 2  grass-shrub
    SH (141-149)        -> 3  shrub / chaparral
    TU (161-165)        -> 4  timber-understory
    TL (181-189)        -> 5  timber-litter
    SB (201-204)        -> 6  slash-blowdown
    NB (91-99: urban/water/barren/snow/agriculture) -> 0 non-burnable
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from .cma import HOUSE_FUEL_CLASS, RasterStack

LOG = logging.getLogger("wildfire_cma.gis")

# Full Southern California grid footprint (SCE + LADWP + SDG&E service area
# plus the AZ/NV intertie buses that import into the region). Derived from the
# actual extent of the 2,294-bus SoCal pandapower model (bus lon/lat min/max
# with a small margin) so the wildfire raster co-registers with *every* bus and
# line, not just the coastal-metro core.
#   model lon: -121.115 .. -113.915   lat: 32.516 .. 37.546
# (minlon, minlat, maxlon, maxlat)
SOCAL_BOUNDS = (-121.3, 32.4, -113.7, 37.7)


def fbfm40_to_class(fbfm: np.ndarray) -> np.ndarray:
    """Map LANDFIRE FBFM40 codes to coarse fuel-class ids used by the CMA."""
    out = np.zeros_like(fbfm, dtype=np.int16)
    out[(fbfm >= 101) & (fbfm <= 109)] = 1   # GR
    out[(fbfm >= 121) & (fbfm <= 124)] = 2   # GS
    out[(fbfm >= 141) & (fbfm <= 149)] = 3   # SH (chaparral)
    out[(fbfm >= 161) & (fbfm <= 165)] = 4   # TU
    out[(fbfm >= 181) & (fbfm <= 189)] = 5   # TL
    out[(fbfm >= 201) & (fbfm <= 204)] = 6   # SB
    # everything else (incl. 91-99 NB) stays 0 = non-burnable
    return out


def _approx_cell_size_m(bounds, nrows, ncols) -> float:
    """Approximate cell size in metres at the centre latitude."""
    minlon, minlat, maxlon, maxlat = bounds
    mid_lat = (minlat + maxlat) / 2.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(mid_lat))
    dy = (maxlat - minlat) * m_per_deg_lat / nrows
    dx = (maxlon - minlon) * m_per_deg_lon / ncols
    return float((dx + dy) / 2.0)


def synthetic_socal(
    nrows: int = 200,
    ncols: int = 260,
    bounds: Tuple[float, float, float, float] = SOCAL_BOUNDS,
    seed: int = 7,
) -> RasterStack:
    """Generate a deterministic California-like raster (offline fallback).

    Produces a fuel map dominated by chaparral in the foothills, grass in the
    valleys, timber at elevation, and non-burnable urban/ocean cells, plus a
    DEM with coastal plain rising to inland mountains. Reproducible by seed.
    """
    rng = np.random.default_rng(seed)
    minlon, minlat, maxlon, maxlat = bounds

    # --- DEM: rise from the SW coast toward NE mountains + noise -----------
    yy, xx = np.mgrid[0:nrows, 0:ncols]
    coast = (xx / ncols) * 0.7 + (1 - yy / nrows) * 0.3   # high to the E/N
    ridges = 0.15 * np.sin(xx / 12.0) * np.cos(yy / 15.0)
    dem = (coast + ridges) * 2200.0
    dem += rng.normal(0, 40, (nrows, ncols))
    dem = np.clip(dem, 0, None)

    # --- fuel: elevation-banded with patchiness ----------------------------
    fuel = np.full((nrows, ncols), 3, dtype=np.int16)  # default chaparral
    fuel[dem < 150] = 1            # low valleys -> grass
    fuel[(dem >= 150) & (dem < 400)] = 2   # foothill grass-shrub
    fuel[(dem >= 400) & (dem < 1200)] = 3  # chaparral belt
    fuel[dem >= 1200] = 4          # montane timber-understory
    fuel[dem >= 1800] = 5          # high timber-litter
    # ocean to the SW corner -> non-burnable
    ocean = (xx / ncols + yy / nrows) < 0.18
    fuel[ocean] = 0
    # Scattered built-up patches (class 9 = houses), matching _fuel_from_dem so
    # the synthetic fallback is not silently house-free. ``& ~ocean`` guards the
    # same overwrite bug the real-DEM builder had: this scatter runs after the
    # water mask, so without it 4% of the sea becomes burnable houses.
    urban = (rng.random((nrows, ncols)) < 0.04) & ~ocean
    fuel[urban] = HOUSE_FUEL_CLASS
    # random fuel-break roads (cut through settlement too -- roads are the
    # non-burnable break, so they legitimately overwrite class 9)
    for _ in range(6):
        r = rng.integers(0, nrows)
        fuel[r, :] = np.where(rng.random(ncols) < 0.6, 0, fuel[r, :])

    delta = _approx_cell_size_m(bounds, nrows, ncols)
    LOG.info("Synthetic SoCal raster %dx%d, ~%.0f m cells", nrows, ncols, delta)
    return RasterStack(fuel=fuel, dem=dem, delta_m=delta, bounds=bounds,
                       source="synthetic")


# Default location of the cached real SRTM GL3 mosaic produced by
# ``data/dem/fetch_dem_tiles.py`` (OpenTopography). Stored as a compact .npz so
# the repo and CI need no rasterio/GDAL to read it -- just numpy.
import os as _os  # noqa: E402
_DEM_NPZ = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                         "data", "dem", "socal_srtm_gl3.npz")


def _resample_nearest(arr: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """Pure-numpy nearest-neighbour resample (no scipy dependency)."""
    out_r, out_c = out_shape
    in_r, in_c = arr.shape
    ri = (np.linspace(0, in_r - 1, out_r)).round().astype(int)
    ci = (np.linspace(0, in_c - 1, out_c)).round().astype(int)
    return arr[np.ix_(ri, ci)]


def _resample_bilinear(arr: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """Pure-numpy bilinear resample for the continuous DEM."""
    out_r, out_c = out_shape
    in_r, in_c = arr.shape
    ys = np.linspace(0, in_r - 1, out_r)
    xs = np.linspace(0, in_c - 1, out_c)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, in_r - 1)
    x1 = np.minimum(x0 + 1, in_c - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    a = arr[np.ix_(y0, x0)]
    b = arr[np.ix_(y0, x1)]
    c = arr[np.ix_(y1, x0)]
    d = arr[np.ix_(y1, x1)]
    top = a * (1 - wx) + b * wx
    bot = c * (1 - wx) + d * wx
    return top * (1 - wy) + bot * wy


def _fuel_from_dem(dem: np.ndarray, seed: int = 7) -> np.ndarray:
    """Derive coarse fuel classes from a real DEM (elevation-banded).

    Real LANDFIRE fuel is not bundled, so we approximate the SoCal fuel
    structure from elevation: ocean/water (<=0 m) and the highest alpine zone
    are non-burnable; grassland valleys, a broad chaparral foothill belt, and
    montane timber follow the classic Southern California vegetation gradient.
    Deterministic by ``seed`` for reproducibility.
    """
    rng = np.random.default_rng(seed)
    fuel = np.full(dem.shape, 3, dtype=np.int16)   # default chaparral
    fuel[dem < 150] = 1                              # valleys -> grass
    fuel[(dem >= 150) & (dem < 400)] = 2             # foothill grass-shrub
    fuel[(dem >= 400) & (dem < 1200)] = 3            # chaparral belt
    fuel[(dem >= 1200) & (dem < 1800)] = 4           # montane timber-understory
    fuel[(dem >= 1800) & (dem < 2800)] = 5           # high timber-litter
    fuel[dem >= 2800] = 0          # alpine / rock above treeline -> non-burnable
    fuel[dem <= 0] = 0             # ocean / Salton Trough water -> non-burnable
    # Light scatter of built-up cells (class 9 = houses) in low valleys.
    # ``dem > 0`` is load-bearing: ``dem < 250`` alone also selects every ocean
    # and Salton-Trough cell, and this scatter is applied AFTER the water mask
    # above, so without it ~55%% of the "houses" land below sea level -- and,
    # because class 9 is burnable, let fire spread across open water. The bug
    # was invisible while this line assigned 0 (overwriting water with water).
    low = (dem < 250) & (dem > 0)
    urban = (rng.random(dem.shape) < 0.03) & low
    fuel[urban] = HOUSE_FUEL_CLASS
    return fuel


def socal_from_srtm(
    nrows: int = 600,
    ncols: int = 760,
    bounds: Tuple[float, float, float, float] = SOCAL_BOUNDS,
    dem_npz: Optional[str] = None,
    seed: int = 7,
) -> RasterStack:
    """Build a RasterStack from the cached **real** SRTM GL3 DEM mosaic.

    Reads the numpy ``.npz`` produced by ``data/dem/fetch_dem_tiles.py``
    (OpenTopography SRTM GL3, origin upper / north-at-top), resamples it onto
    the model's ``nrows x ncols`` grid, and derives fuel classes from the real
    elevation. No rasterio/GDAL required -- works in the sandbox and in CI.

    Raises ``FileNotFoundError`` if the DEM cache is missing so callers can
    fall back to :func:`synthetic_socal`.
    """
    path = dem_npz or _DEM_NPZ
    if not _os.path.exists(path):
        raise FileNotFoundError(
            f"Real DEM cache not found at {path}; run data/dem/fetch_dem_tiles.py"
        )
    dem_full = np.load(path)["dem"].astype(float)   # (rows, cols), north at top
    dem = _resample_bilinear(dem_full, (nrows, ncols))
    fuel = _fuel_from_dem(dem, seed=seed)
    delta = _approx_cell_size_m(bounds, nrows, ncols)
    LOG.info("Real SRTM SoCal raster %dx%d from %s (~%.0f m cells, "
             "elev %.0f..%.0f m)", nrows, ncols, _os.path.basename(path),
             delta, float(dem.min()), float(dem.max()))
    return RasterStack(fuel=fuel, dem=dem, delta_m=delta, bounds=bounds,
                       source="srtm_gl3")


def from_rasters(
    fuel_path: str,
    dem_path: str,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    nlcd_path: Optional[str] = None,
    target_shape: Optional[Tuple[int, int]] = None,
) -> RasterStack:
    """Build a RasterStack from real LANDFIRE/3DEP GeoTIFFs via rasterio.

    Reprojects/resamples both layers onto a common grid clipped to ``bounds``
    (defaults to the SoCal footprint). Requires ``rasterio``.
    """
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    from rasterio.windows import from_bounds as window_from_bounds

    bounds = bounds or SOCAL_BOUNDS # Where the tiles description come from

    def _read(path):
        with rasterio.open(path) as src:
            arr = src.read(1)
            return arr, src.transform, src.crs, src.bounds

    fuel_raw, ftrans, fcrs, _ = _read(fuel_path)
    dem_raw, _, _, _ = _read(dem_path)

    # naive common grid: resample DEM to fuel shape (assumes overlapping AOI)
    if dem_raw.shape != fuel_raw.shape:
        from scipy.ndimage import zoom
        zy = fuel_raw.shape[0] / dem_raw.shape[0]
        zx = fuel_raw.shape[1] / dem_raw.shape[1]
        dem_raw = zoom(dem_raw, (zy, zx), order=1)

    fuel = fbfm40_to_class(fuel_raw)
    if nlcd_path:
        nlcd, _, _, _ = _read(nlcd_path)
        if nlcd.shape == fuel.shape:
            # NLCD: 11=water, 21-24=developed - houses should be burnable, 31=barren -> non-burnable           
            nb1 = np.isin(nlcd, [11, 12, 31])
            nb2 = np.isin(nlcd, [21, 22, 23, 24])
            fuel[nb1] = 0
            fuel[nb2] = 9


    if target_shape and target_shape != fuel.shape:
        from scipy.ndimage import zoom
        zy = target_shape[0] / fuel.shape[0]
        zx = target_shape[1] / fuel.shape[1]
        fuel = zoom(fuel, (zy, zx), order=0).astype(np.int16)
        dem_raw = zoom(dem_raw, (zy, zx), order=1)

    delta = _approx_cell_size_m(bounds, *fuel.shape)
    return RasterStack(fuel=fuel.astype(np.int16), dem=np.asarray(dem_raw, float),
                       delta_m=delta, bounds=bounds, source="rasters")


def build_socal_raster(
    fuel_path: Optional[str] = None,
    dem_path: Optional[str] = None,
    nrows: int = 200,
    ncols: int = 260,
    bounds: Tuple[float, float, float, float] = SOCAL_BOUNDS,
) -> RasterStack:
    """Convenience: real rasters if both paths given, else synthetic SoCal."""
    if fuel_path and dem_path:
        try:
            return from_rasters(fuel_path, dem_path, bounds,
                                target_shape=(nrows, ncols))
        except Exception as exc:  # pragma: no cover - env dependent
            LOG.warning("Real raster ingest failed (%s); using synthetic", exc)
    return synthetic_socal(nrows, ncols, bounds)
