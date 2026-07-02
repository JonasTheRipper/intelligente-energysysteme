"""Perimeter validation harness (v0.5).

Ingests an official CAL FIRE / NIFC fire perimeter (GeoJSON, EPSG:4326),
rasterises it onto a :class:`wildfire_cma.gis.RasterStack`, and scores a
simulated final fire footprint against it with set-overlap metrics:

    Dice  = 2|A n B| / (|A| + |B|)
    Jaccard (IoU) = |A n B| / |A u B|
    area ratio = sim_area / real_area  (both in acres)

The rasteriser is pure-numpy point-in-polygon (ray casting) over MultiPolygon
rings with hole support -- no shapely/GDAL dependency, so it runs in the pinned
numpy-only environment. Cell area is derived from the raster's ``delta_m``
(metres per cell) so acreage is physically meaningful.

This module is import-safe (no side effects) and is exercised by
``tests/test_perimeter_validation.py``.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

ACRES_PER_SQ_M = 0.000247105


# --------------------------------------------------------------------------- #
# GeoJSON ingest
# --------------------------------------------------------------------------- #
def load_perimeter_polygons(path: str) -> List[List[np.ndarray]]:
    """Load a GeoJSON perimeter -> list of polygons.

    Each polygon is a list of rings; ring[0] is the exterior, ring[1:] are
    holes. Each ring is an (N, 2) float array of (lon, lat) vertices.
    Supports Polygon and MultiPolygon (FeatureCollection or bare geometry).
    """
    with open(path) as f:
        gj = json.load(f)

    feats = []
    if gj.get("type") == "FeatureCollection":
        feats = [ft["geometry"] for ft in gj["features"] if ft.get("geometry")]
    elif gj.get("type") == "Feature":
        feats = [gj["geometry"]]
    else:  # bare geometry
        feats = [gj]

    polys: List[List[np.ndarray]] = []
    for geom in feats:
        gtype = geom["type"]
        coords = geom["coordinates"]
        if gtype == "Polygon":
            polys.append([np.asarray(ring, dtype=np.float64) for ring in coords])
        elif gtype == "MultiPolygon":
            for poly in coords:
                polys.append([np.asarray(ring, dtype=np.float64) for ring in poly])
        else:  # pragma: no cover - defensive
            raise ValueError(f"unsupported geometry type: {gtype}")
    return polys


def perimeter_bbox(polys: Sequence[Sequence[np.ndarray]]) -> Tuple[float, float, float, float]:
    """Return (minlon, minlat, maxlon, maxlat) over all rings."""
    xs: List[float] = []
    ys: List[float] = []
    for poly in polys:
        for ring in poly:
            xs.extend(ring[:, 0].tolist())
            ys.extend(ring[:, 1].tolist())
    return (min(xs), min(ys), max(xs), max(ys))


# --------------------------------------------------------------------------- #
# Rasterisation (ray-casting point-in-polygon, vectorised per ring)
# --------------------------------------------------------------------------- #
def _points_in_ring(lon: np.ndarray, lat: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting for a flat array of points vs one ring.

    ``lon``/``lat`` are 1-D arrays of equal length; ``ring`` is (M, 2). Returns
    a boolean array (True = inside the ring).
    """
    x = lon
    y = lat
    inside = np.zeros(x.shape, dtype=bool)
    xr = ring[:, 0]
    yr = ring[:, 1]
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = xr[i], yr[i]
        xj, yj = xr[j], yr[j]
        # does the horizontal ray from the point cross edge (j -> i)?
        cond = ((yi > y) != (yj > y))
        # x coordinate of the edge at the point's latitude
        denom = (yj - yi)
        denom = np.where(denom == 0.0, 1e-30, denom)
        xint = (xj - xi) * (y - yi) / denom + xi
        cross = cond & (x < xint)
        inside ^= cross
        j = i
    return inside


def rasterize_perimeter(polys, bounds, nrows: int, ncols: int) -> np.ndarray:
    """Rasterise perimeter polygons to a boolean (nrows, ncols) mask.

    ``bounds`` is (minlon, minlat, maxlon, maxlat). Cell (row, col) centre is
    sampled; row 0 is the northern edge (origin upper), matching
    :class:`wildfire_cma.gis.RasterStack`. A cell is burned iff its centre is
    inside an exterior ring and not inside any hole of that polygon.
    """
    minlon, minlat, maxlon, maxlat = bounds
    # cell-centre coordinates
    lons = minlon + (np.arange(ncols) + 0.5) / ncols * (maxlon - minlon)
    lats = maxlat - (np.arange(nrows) + 0.5) / nrows * (maxlat - minlat)  # upper origin
    LON, LAT = np.meshgrid(lons, lats)
    flon = LON.ravel()
    flat = LAT.ravel()

    mask = np.zeros(flon.shape, dtype=bool)
    for poly in polys:
        if not poly:
            continue
        ext = _points_in_ring(flon, flat, poly[0])
        for hole in poly[1:]:
            ext &= ~_points_in_ring(flon, flat, hole)
        mask |= ext
    return mask.reshape(nrows, ncols)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int(np.count_nonzero(a & b))
    sa = int(a.sum())
    sb = int(b.sum())
    if sa + sb == 0:
        return 1.0
    return 2.0 * inter / (sa + sb)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return inter / union


def cell_area_acres(delta_m: float) -> float:
    return (delta_m * delta_m) * ACRES_PER_SQ_M


def score(sim_mask: np.ndarray, real_mask: np.ndarray, delta_m: float) -> Dict[str, float]:
    """Full metric bundle for a simulated vs real burned mask."""
    ca = cell_area_acres(delta_m)
    sim_cells = int(sim_mask.astype(bool).sum())
    real_cells = int(real_mask.astype(bool).sum())
    sim_acres = sim_cells * ca
    real_acres = real_cells * ca
    d = dice(sim_mask, real_mask)
    j = jaccard(sim_mask, real_mask)
    area_ratio = (sim_acres / real_acres) if real_acres > 0 else float("nan")
    return {
        "dice": d,
        "jaccard": j,
        "sim_cells": sim_cells,
        "real_cells": real_cells,
        "sim_acres": sim_acres,
        "real_acres": real_acres,
        "area_ratio": area_ratio,
        "area_pct_err": (area_ratio - 1.0) * 100.0 if real_acres > 0 else float("nan"),
        "cell_area_acres": ca,
    }


def meets_bar(metrics: Dict[str, float], dice_min: float = 0.8,
              area_tol_pct: float = 10.0) -> bool:
    """The hard accuracy bar: Dice >= dice_min AND |area error| <= area_tol_pct."""
    return (metrics["dice"] >= dice_min
            and abs(metrics["area_pct_err"]) <= area_tol_pct)
