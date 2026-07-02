"""Satellite / aerial basemap tiles for the timelapse map rows (v0.6).

v0.5 drew a synthetic hypsometric-tinted hillshade for the left-hand map. Two
problems reported by the user:

  * everything at/below sea level was painted flat blue, so the (dry, urban) LA
    basin looked like open Pacific ocean; and
  * with no roads, street grids, or place context the terrain was hard to orient
    on unless you already knew the area.

This module fetches real **Esri World Imagery** web-map tiles (public, no API
key) covering a fire's lon/lat extent, stitches them into one RGB image, and
crops it to the exact extent so it registers pixel-for-pixel with the fire
raster's ``imshow(extent=...)``. The result is a true satellite backdrop that
shows the coastline, urban fabric, canyons and ridgelines directly.

Tiles are fetched through the sandbox's normal HTTPS path (``requests``) and
cached under ``data/basemap_cache/`` so repeated renders don't refetch (and we
stay well under any tile-server rate limit). If the network is unavailable the
caller can fall back to the synthetic basemap.

Public tile service (Esri, "World Imagery"), standard XYZ scheme:
  https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
Attribution: Esri, Maxar, Earthstar Geographics, and the GIS User Community.
"""
from __future__ import annotations

import io
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_CACHE = os.path.join(_ROOT, "data", "basemap_cache")

_ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
         "World_Imagery/MapServer/tile/{z}/{y}/{x}")
# Optional reference overlay with roads + place labels (transparent PNG).
_ESRI_REF = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
             "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}")

_TILE = 256
_UA = {"User-Agent": "Mozilla/5.0 (socal-wildfires basemap)"}
ATTRIBUTION = "Imagery: Esri, Maxar, Earthstar Geographics"


# --- Web-Mercator <-> tile math (standard slippy-map) ---------------------
def _lon_to_x(lon: float, z: int) -> float:
    return (lon + 180.0) / 360.0 * (2 ** z)


def _lat_to_y(lat: float, z: int) -> float:
    lat_r = math.radians(lat)
    return (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * (2 ** z)


def _x_to_lon(x: float, z: int) -> float:
    return x / (2 ** z) * 360.0 - 180.0


def _y_to_lat(y: float, z: int) -> float:
    n = math.pi - 2.0 * math.pi * y / (2 ** z)
    return math.degrees(math.atan(math.sinh(n)))


def _pick_zoom(extent, px_target=1600):
    """Choose the smallest zoom whose stitched width >= px_target across the
    extent's longitude span (so the crop is at least ~px_target px wide)."""
    minlon, maxlon, minlat, maxlat = extent
    for z in range(6, 17):
        span_tiles = _lon_to_x(maxlon, z) - _lon_to_x(minlon, z)
        if span_tiles * _TILE >= px_target:
            return z
    return 16


def _fetch_tile(session, template, z, x, y):
    import requests  # local import so module import never hard-fails
    n = 2 ** z
    if not (0 <= x < n and 0 <= y < n):
        return None
    key = template.split("/services/")[1].split("/MapServer")[0]
    key = key.replace("/", "_")
    cdir = os.path.join(_CACHE, key, str(z))
    os.makedirs(cdir, exist_ok=True)
    cpath = os.path.join(cdir, f"{x}_{y}.img")
    if os.path.exists(cpath):
        with open(cpath, "rb") as fh:
            return fh.read()
    url = template.format(z=z, x=x, y=y)
    for _ in range(3):
        try:
            r = session.get(url, timeout=25, headers=_UA)
            if r.status_code == 200 and r.content:
                with open(cpath, "wb") as fh:
                    fh.write(r.content)
                return r.content
        except Exception:
            pass
    return None


def _stitch(extent, z, template, with_alpha=False):
    """Fetch + stitch all tiles covering ``extent`` at zoom ``z``.

    Returns (rgb_or_rgba float array in [0,1], tile_extent) where tile_extent
    is the [minlon,maxlon,minlat,maxlat] actually covered by the tile grid
    (slightly larger than the requested extent).
    """
    from PIL import Image
    import requests
    minlon, maxlon, minlat, maxlat = extent
    x0 = int(math.floor(_lon_to_x(minlon, z)))
    x1 = int(math.floor(_lon_to_x(maxlon, z)))
    y0 = int(math.floor(_lat_to_y(maxlat, z)))  # north -> smaller y
    y1 = int(math.floor(_lat_to_y(minlat, z)))
    nx = x1 - x0 + 1
    ny = y1 - y0 + 1
    mode = "RGBA" if with_alpha else "RGB"
    canvas = Image.new(mode, (nx * _TILE, ny * _TILE))
    session = requests.Session()
    got = 0
    for xi, x in enumerate(range(x0, x1 + 1)):
        for yi, y in enumerate(range(y0, y1 + 1)):
            raw = _fetch_tile(session, template, z, x, y)
            if raw is None:
                continue
            try:
                tile = Image.open(io.BytesIO(raw)).convert(mode)
                canvas.paste(tile, (xi * _TILE, yi * _TILE))
                got += 1
            except Exception:
                continue
    if got == 0:
        raise RuntimeError("no basemap tiles fetched")
    tile_extent = [
        _x_to_lon(x0, z), _x_to_lon(x1 + 1, z),
        _y_to_lat(y1 + 1, z), _y_to_lat(y0, z),
    ]
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr, tile_extent


def _crop_to_extent(arr, tile_extent, extent):
    """Crop the stitched tile mosaic to the requested lon/lat extent.

    The mosaic is in Web-Mercator pixel space (linear in lon, non-linear in
    lat). We resample onto a regular lon/lat grid so the returned image lines up
    exactly with a matplotlib ``imshow(extent=extent, origin='upper')`` call
    that shares the fire raster's PlateCarree extent. A tiny latitude warp over
    a ~0.1-0.2 deg span is visually negligible but we handle it correctly via
    per-row Mercator-y interpolation.
    """
    minlon, maxlon, minlat, maxlat = extent
    tminlon, tmaxlon, tminlat, tmaxlat = tile_extent
    H, W = arr.shape[:2]
    out_h, out_w = 900, 1600  # oversampled; imshow will scale to the axes

    # longitude is linear in mercator-x -> linear pixel mapping
    lons = np.linspace(minlon, maxlon, out_w)
    fx = (lons - tminlon) / (tmaxlon - tminlon) * (W - 1)
    fx = np.clip(fx, 0, W - 1).astype(np.int32)

    # latitude: map each output lat (linear top->bottom) through mercator-y
    lats = np.linspace(maxlat, minlat, out_h)  # top row = north
    # tile row 0 = tmaxlat (north); row H-1 = tminlat (south), linear in
    # mercator-y. Convert lat -> mercator-y fraction within the tile extent.
    def _merc(lat):
        return math.asinh(math.tan(math.radians(lat)))
    ty_top, ty_bot = _merc(tmaxlat), _merc(tminlat)
    my = np.array([_merc(la) for la in lats])
    fy = (ty_top - my) / (ty_top - ty_bot) * (H - 1)
    fy = np.clip(fy, 0, H - 1).astype(np.int32)

    out = arr[np.ix_(fy, fx)]
    return out


def satellite_rgb(extent, px_target=1600, with_labels=True):
    """Return an RGB float image (H,W,3) for ``extent`` = [minlon,maxlon,
    minlat,maxlat], resampled to PlateCarree so it aligns with the fire raster.

    When ``with_labels`` is True the Esri roads+places reference layer is
    alpha-composited on top so street grids and city names are visible. Raises
    on total network failure so the caller can fall back to the synthetic map.
    """
    z = _pick_zoom(extent, px_target=px_target)
    base, tile_extent = _stitch(extent, z, _ESRI, with_alpha=False)
    rgb = _crop_to_extent(base, tile_extent, extent)

    if with_labels:
        try:
            ref, ref_te = _stitch(extent, z, _ESRI_REF, with_alpha=True)
            ref_rgba = _crop_to_extent(ref, ref_te, extent)
            a = ref_rgba[:, :, 3:4]
            rgb = rgb * (1.0 - a) + ref_rgba[:, :, :3] * a
        except Exception:
            pass  # imagery alone is fine if the reference layer is unavailable
    return np.clip(rgb, 0.0, 1.0)


if __name__ == "__main__":
    # quick smoke test / cache warm for the Eaton extent
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [-118.1771, -117.9981, 34.1469, 34.2528]
    img = satellite_rgb(ext, with_labels=True)
    print("satellite_rgb shape", img.shape, "zoom", _pick_zoom(ext))
    plt.imshow(img, extent=ext, origin="upper")
    plt.title("Esri World Imagery — Eaton extent")
    plt.savefig("/tmp/sat_smoke.png", dpi=110, bbox_inches="tight")
    print("wrote /tmp/sat_smoke.png")
