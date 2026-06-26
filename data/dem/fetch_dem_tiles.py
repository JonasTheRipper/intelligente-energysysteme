"""Fetch the SoCal SRTM GL3 DEM from OpenTopography in tiles and stitch.

The full-footprint ASCII request is too large for the proxied transfer
(connection closed abruptly), so we request a grid of smaller bounding boxes
and mosaic them into one numpy array, saved as a compact .npz plus a small
header .json. Pure-numpy ASCII parsing -- no rasterio/GDAL needed.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import numpy as np

BOUNDS = (-121.3, 32.4, -113.7, 37.7)  # minlon, minlat, maxlon, maxlat
DEMTYPE = "SRTMGL3"
HERE = os.path.dirname(os.path.abspath(__file__))
NX_TILES = 4   # along longitude
NY_TILES = 3   # along latitude


def _parse_asc(path):
    """Parse an ESRI ASCII grid -> (array, header dict)."""
    hdr = {}
    with open(path, "r") as f:
        # header: 6 lines of "key value"
        keys = ["ncols", "nrows", "xllcorner", "yllcorner", "cellsize",
                "nodata_value"]
        for _ in range(6):
            line = f.readline().split()
            hdr[line[0].lower()] = float(line[1])
        data = np.loadtxt(f)
    ncols = int(hdr["ncols"])
    nrows = int(hdr["nrows"])
    data = data.reshape(nrows, ncols)
    return data, hdr


def fetch_tile(s, n, w, e, idx):
    out = os.path.join(HERE, f"_tile_{idx}.asc")
    url = (f"https://portal.opentopography.org/API/globaldem?demtype={DEMTYPE}"
           f"&south={s}&north={n}&west={w}&east={e}&outputFormat=AAIGrid")
    # OpenTopography requires an API key. Append it from the environment when
    # present; if the request is routed through an auth proxy that injects the
    # key (e.g. as a query parameter), leaving this unset is also fine.
    api_key = os.environ.get("OPENTOPOGRAPHY_API_KEY")
    if api_key:
        url += f"&API_Key={api_key}"
    # retry up to 3x; the proxied transfer can drop on larger payloads
    for attempt in range(3):
        rc = subprocess.run(
            ["curl", "-sS", "--max-time", "180", "-o", out, url],
            capture_output=True, text=True,
        )
        if rc.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1000:
            arr, hdr = _parse_asc(out)
            print(f"  tile {idx}: ok {arr.shape} "
                  f"({os.path.getsize(out)//1024} KiB) attempt {attempt+1}")
            return arr, hdr
        print(f"  tile {idx}: attempt {attempt+1} failed rc={rc.returncode} "
              f"size={os.path.getsize(out) if os.path.exists(out) else 0}")
    raise RuntimeError(f"tile {idx} failed after retries")


def main():
    minlon, minlat, maxlon, maxlat = BOUNDS
    lon_edges = np.linspace(minlon, maxlon, NX_TILES + 1)
    lat_edges = np.linspace(minlat, maxlat, NY_TILES + 1)

    # collect tiles row by row (top = north). We'll assemble by lat descending.
    rows = []  # each entry list of column arrays, north->south
    cell = None
    idx = 0
    # iterate latitude bands from north (top) to south (bottom)
    for j in range(NY_TILES - 1, -1, -1):
        s, n = lat_edges[j], lat_edges[j + 1]
        cols = []
        for i in range(NX_TILES):
            w, e = lon_edges[i], lon_edges[i + 1]
            arr, hdr = fetch_tile(round(s, 5), round(n, 5),
                                  round(w, 5), round(e, 5), idx)
            arr = np.where(arr == hdr["nodata_value"], 0.0, arr)
            cols.append(arr)
            cell = hdr["cellsize"]
            idx += 1
        # align row heights (tiles may differ by 1 px); crop to min nrows
        minr = min(a.shape[0] for a in cols)
        cols = [a[:minr, :] for a in cols]
        rows.append(np.hstack(cols))

    # align row widths; crop to min ncols
    minc = min(r.shape[1] for r in rows)
    rows = [r[:, :minc] for r in rows]
    dem = np.vstack(rows)  # north (top) -> south (bottom), origin upper
    print(f"[dem] mosaic shape = {dem.shape}, cellsize={cell:.6f} deg, "
          f"min={dem.min():.0f} max={dem.max():.0f} m")

    np.savez_compressed(os.path.join(HERE, "socal_srtm_gl3.npz"),
                        dem=dem.astype(np.float32))
    meta = {
        "bounds": list(BOUNDS), "demtype": DEMTYPE,
        "cellsize_deg": cell, "shape": list(dem.shape),
        "origin": "upper", "source": "OpenTopography SRTM GL3 (90m)",
        "nx_tiles": NX_TILES, "ny_tiles": NY_TILES,
    }
    json.dump(meta, open(os.path.join(HERE, "socal_srtm_gl3.json"), "w"),
              indent=2)
    # cleanup tile files
    for k in range(idx):
        p = os.path.join(HERE, f"_tile_{k}.asc")
        if os.path.exists(p):
            os.remove(p)
    print("[dem] saved socal_srtm_gl3.npz + .json")


if __name__ == "__main__":
    main()
