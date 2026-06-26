"""Unit tests for the PostGIS staging layer (no live server required).

These tests use a fake DB-API connection/cursor to verify that the SQL and the
WKT geometry strings the module emits are well-formed, without needing a real
PostGIS server (which is provided by docker-compose for integration use).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from wildfire_cma import postgis as pg_mod  # noqa: E402
from wildfire_cma.cma import BURNED_OUT, BURNING, RasterStack, Theta, WildfireCMA  # noqa: E402
from wildfire_cma.gis import synthetic_socal  # noqa: E402


class FakeCursor:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.store["executed"].append((sql, params))

    def executemany(self, sql, rows):
        self.store["executemany"].append((sql, list(rows)))

    def fetchone(self):
        return self.store.get("fetchone")

    def fetchall(self):
        return self.store.get("fetchall", [])


class FakeConn:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return FakeCursor(self.store)

    def commit(self):
        self.store["commits"] += 1


def _patch(monkey_store):
    pg_mod._connect = lambda dsn: FakeConn(monkey_store)  # type: ignore


def _fresh_store():
    return {"executed": [], "executemany": [], "commits": 0}


def test_default_dsn_has_expected_fields():
    dsn = pg_mod.default_dsn()
    for token in ["dbname=", "user=", "host=", "port=", "password="]:
        assert token in dsn
    print("DSN:", dsn)


def test_init_schema_emits_postgis_and_tables():
    store = _fresh_store()
    _patch(store)
    pg = pg_mod.PostGIS("fake")
    pg.init_schema()
    sql = " ".join(s for s, _ in store["executed"])
    assert "CREATE EXTENSION IF NOT EXISTS postgis" in sql
    for tbl in ["raster_meta", "fuel_cells", "grid_bus", "grid_line",
                "fire_perimeter"]:
        assert tbl in sql, f"missing table {tbl}"
    assert store["commits"] >= 1


def test_write_raster_bulk_inserts_cells():
    store = _fresh_store()
    _patch(store)
    raster = synthetic_socal(nrows=10, ncols=12, seed=3)
    pg = pg_mod.PostGIS("fake")
    pg.write_raster(raster, name="t")
    # one executemany with nrows*ncols rows
    assert store["executemany"], "expected bulk cell insert"
    _, rows = store["executemany"][0]
    assert len(rows) == 10 * 12
    # each row: (name,row,col,fuel,elev,lon,lat)
    name, r, c, fuel, elev, lon, lat = rows[0]
    assert name == "t"
    assert -180 <= lon <= 180 and -90 <= lat <= 90
    print("first cell:", rows[0])


def test_write_fire_perimeter_emits_valid_multipolygon():
    store = _fresh_store()
    _patch(store)
    raster = synthetic_socal(nrows=40, ncols=50, seed=3)
    theta = Theta(ignition_rc=[(20, 25)], wind_speed=12.0, kappa=2.0,
                  dead_fuel_moisture=0.04)
    cma = WildfireCMA(raster, theta, dt_cma_min=5, t_burn_steps=6, seed=1)
    cma.advance(minutes=60)
    n_fire = int(((cma.state == BURNING) | (cma.state == BURNED_OUT)).sum())
    assert n_fire > 0, "need some fire cells to write a perimeter"

    pg = pg_mod.PostGIS("fake")
    pg.write_fire_perimeter("run1", step=1, sim_min=60.0, cma=cma)
    # find the insert with a MULTIPOLYGON wkt
    wkts = [
        params[-1]
        for sql, params in store["executed"]
        if params and isinstance(params[-1], str) and "MULTIPOLYGON" in params[-1]
    ]
    assert wkts, "expected a MULTIPOLYGON WKT insert"
    wkt = wkts[0]
    assert wkt.startswith("MULTIPOLYGON(")
    assert wkt.count("((") == n_fire  # one square polygon per fire cell
    print(f"MULTIPOLYGON with {n_fire} cell polygons, len={len(wkt)}")


def test_read_raster_reconstructs_grid():
    store = _fresh_store()
    # meta row + cell rows
    store["fetchone"] = (3, 4, 947.0, -121.3, 32.4, -113.7, 37.7)
    cells = []
    for r in range(3):
        for c in range(4):
            cells.append((r, c, (r + c) % 6, 100.0 * (r + 1)))
    store["fetchall"] = cells
    _patch(store)
    pg = pg_mod.PostGIS("fake")
    raster = pg.read_raster("socal")
    assert raster.shape == (3, 4)
    assert raster.fuel[2, 3] == (2 + 3) % 6
    assert raster.dem[2, 0] == 300.0
    print("reconstructed raster", raster.shape, "bounds", raster.bounds)


if __name__ == "__main__":
    for fn in [
        test_default_dsn_has_expected_fields,
        test_init_schema_emits_postgis_and_tables,
        test_write_raster_bulk_inserts_cells,
        test_write_fire_perimeter_emits_valid_multipolygon,
        test_read_raster_reconstructs_grid,
    ]:
        print(f"\n=== {fn.__name__} ===")
        fn()
        print(f"PASS {fn.__name__}")
    print("\nALL POSTGIS TESTS PASSED")
