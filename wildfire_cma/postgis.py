"""PostGIS staging layer for the SoCal wildfire CMA.

Optional persistence backend that houses the California GIS layers used to
build the cellular structure and to co-register the grid:

* ``fuel_cells``  -- the rasterised fuel-class grid (one row per cell, with a
  cell polygon and centroid in EPSG:4326),
* ``dem_cells``   -- elevation per cell,
* ``grid_bus`` / ``grid_line`` -- the pandapower bus/line geometries, so the
  damage mapper's co-registration can be reproduced from the database,
* ``fire_perimeter`` -- per-timestep fire front polygons written back during a
  simulation run (handy for QGIS / web-map visualisation).

The module is intentionally dependency-light: it uses ``psycopg`` (v3) or
``psycopg2`` if available, and degrades gracefully (raising a clear error) when
neither the driver nor a server is present. The synthetic / rasterio paths in
``gis.py`` do **not** require PostGIS -- this is purely a staging convenience,
spun up via the bundled ``docker-compose.yml``.

Typical use::

    from wildfire_cma.gis import synthetic_socal
    from wildfire_cma.postgis import PostGIS, default_dsn

    raster = synthetic_socal()
    pg = PostGIS(default_dsn())
    pg.init_schema()
    pg.write_raster(raster)
    raster2 = pg.read_raster()          # round-trips the fuel/DEM grid
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from .cma import RasterStack
from .gis import SOCAL_BOUNDS

LOG = logging.getLogger("wildfire_cma.postgis")


def default_dsn() -> str:
    """DSN matching the bundled docker-compose PostGIS service."""
    return os.environ.get(
        "WILDFIRE_PG_DSN",
        "host={h} port={p} dbname={db} user={u} password={pw}".format(
            h=os.environ.get("PGHOST", "localhost"),
            p=os.environ.get("PGPORT", "5432"),
            db=os.environ.get("PGDATABASE", "wildfire"),
            u=os.environ.get("PGUSER", "wildfire"),
            pw=os.environ.get("PGPASSWORD", "wildfire"),
        ),
    )


def _connect(dsn: str):
    """Return a DB-API connection using psycopg(3) or psycopg2."""
    try:
        import psycopg  # type: ignore

        return psycopg.connect(dsn)
    except Exception:
        pass
    try:
        import psycopg2  # type: ignore

        return psycopg2.connect(dsn)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "No PostgreSQL driver available (install 'psycopg' or 'psycopg2'), "
            f"and/or no server reachable at the DSN. Original error: {e}"
        )


class PostGIS:
    """Thin PostGIS staging helper for the wildfire GIS layers."""

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or default_dsn()

    # -- schema -------------------------------------------------------------
    def init_schema(self) -> None:
        """Create the PostGIS extension and the staging tables (idempotent)."""
        ddl = """
        CREATE EXTENSION IF NOT EXISTS postgis;

        CREATE TABLE IF NOT EXISTS raster_meta (
            id          SERIAL PRIMARY KEY,
            name        TEXT UNIQUE NOT NULL,
            nrows       INTEGER NOT NULL,
            ncols       INTEGER NOT NULL,
            delta_m     DOUBLE PRECISION NOT NULL,
            minlon      DOUBLE PRECISION NOT NULL,
            minlat      DOUBLE PRECISION NOT NULL,
            maxlon      DOUBLE PRECISION NOT NULL,
            maxlat      DOUBLE PRECISION NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS fuel_cells (
            raster_name TEXT NOT NULL,
            row_idx     INTEGER NOT NULL,
            col_idx     INTEGER NOT NULL,
            fuel_class  SMALLINT NOT NULL,
            elevation_m DOUBLE PRECISION,
            centroid    geometry(Point, 4326),
            PRIMARY KEY (raster_name, row_idx, col_idx)
        );
        CREATE INDEX IF NOT EXISTS fuel_cells_gix
            ON fuel_cells USING GIST (centroid);

        CREATE TABLE IF NOT EXISTS grid_bus (
            bus_id   INTEGER PRIMARY KEY,
            vn_kv    DOUBLE PRECISION,
            geom     geometry(Point, 4326)
        );
        CREATE INDEX IF NOT EXISTS grid_bus_gix ON grid_bus USING GIST (geom);

        CREATE TABLE IF NOT EXISTS grid_line (
            line_id  INTEGER PRIMARY KEY,
            from_bus INTEGER,
            to_bus   INTEGER,
            geom     geometry(LineString, 4326)
        );
        CREATE INDEX IF NOT EXISTS grid_line_gix ON grid_line USING GIST (geom);

        CREATE TABLE IF NOT EXISTS fire_perimeter (
            run_uid   TEXT NOT NULL,
            step      INTEGER NOT NULL,
            sim_min   DOUBLE PRECISION,
            geom      geometry(MultiPolygon, 4326),
            PRIMARY KEY (run_uid, step)
        );
        """
        with _connect(self.dsn) as con:
            with con.cursor() as cur:
                cur.execute(ddl)
            con.commit()
        LOG.info("PostGIS schema initialised")

    # -- raster round-trip --------------------------------------------------
    def write_raster(self, raster: RasterStack, name: str = "socal") -> None:
        """Persist a RasterStack (fuel + DEM + per-cell centroids)."""
        nrows, ncols = raster.shape
        minlon, minlat, maxlon, maxlat = raster.bounds
        dlon = (maxlon - minlon) / ncols
        dlat = (maxlat - minlat) / nrows

        with _connect(self.dsn) as con:
            with con.cursor() as cur:
                cur.execute("DELETE FROM fuel_cells WHERE raster_name = %s", (name,))
                cur.execute("DELETE FROM raster_meta WHERE name = %s", (name,))
                cur.execute(
                    """INSERT INTO raster_meta
                       (name, nrows, ncols, delta_m, minlon, minlat, maxlon, maxlat)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (name, nrows, ncols, raster.delta_m,
                     minlon, minlat, maxlon, maxlat),
                )
                # bulk insert cells
                rows = []
                for r in range(nrows):
                    lat = maxlat - (r + 0.5) * dlat
                    for c in range(ncols):
                        lon = minlon + (c + 0.5) * dlon
                        rows.append(
                            (name, r, c, int(raster.fuel[r, c]),
                             float(raster.dem[r, c]), lon, lat)
                        )
                # executemany with WKT centroid
                cur.executemany(
                    """INSERT INTO fuel_cells
                       (raster_name,row_idx,col_idx,fuel_class,elevation_m,centroid)
                       VALUES (%s,%s,%s,%s,%s,
                               ST_SetSRID(ST_MakePoint(%s,%s),4326))""",
                    rows,
                )
            con.commit()
        LOG.info("wrote raster '%s' (%dx%d, %d cells) to PostGIS",
                 name, nrows, ncols, nrows * ncols)

    def read_raster(self, name: str = "socal") -> RasterStack:
        """Reconstruct a RasterStack from PostGIS."""
        with _connect(self.dsn) as con:
            with con.cursor() as cur:
                cur.execute(
                    """SELECT nrows,ncols,delta_m,minlon,minlat,maxlon,maxlat
                       FROM raster_meta WHERE name=%s""",
                    (name,),
                )
                meta = cur.fetchone()
                if meta is None:
                    raise KeyError(f"no raster '{name}' in PostGIS")
                nrows, ncols, delta_m, minlon, minlat, maxlon, maxlat = meta
                fuel = np.zeros((nrows, ncols), dtype=np.int16)
                dem = np.zeros((nrows, ncols), dtype=float)
                cur.execute(
                    """SELECT row_idx,col_idx,fuel_class,elevation_m
                       FROM fuel_cells WHERE raster_name=%s""",
                    (name,),
                )
                for r, c, fc, elev in cur.fetchall():
                    fuel[r, c] = fc
                    dem[r, c] = elev if elev is not None else 0.0
        return RasterStack(
            fuel=fuel, dem=dem, delta_m=float(delta_m),
            bounds=(minlon, minlat, maxlon, maxlat),
        )

    # -- grid assets --------------------------------------------------------
    def write_grid(self, net) -> None:
        """Persist pandapower bus/line geometries (GeoJSON columns)."""
        import json

        with _connect(self.dsn) as con:
            with con.cursor() as cur:
                cur.execute("DELETE FROM grid_bus")
                cur.execute("DELETE FROM grid_line")
                for b in net.bus.index:
                    geo = net.bus.at[b, "geo"] if "geo" in net.bus.columns else None
                    if not isinstance(geo, str):
                        continue
                    g = json.loads(geo)
                    lon, lat = g["coordinates"][0], g["coordinates"][1]
                    cur.execute(
                        """INSERT INTO grid_bus (bus_id,vn_kv,geom)
                           VALUES (%s,%s,ST_SetSRID(ST_MakePoint(%s,%s),4326))
                           ON CONFLICT (bus_id) DO UPDATE
                           SET vn_kv=EXCLUDED.vn_kv, geom=EXCLUDED.geom""",
                        (int(b), float(net.bus.at[b, "vn_kv"]), lon, lat),
                    )
                for ln in net.line.index:
                    geo = net.line.at[ln, "geo"] if "geo" in net.line.columns else None
                    if not isinstance(geo, str):
                        continue
                    g = json.loads(geo)
                    wkt = "LINESTRING(" + ",".join(
                        f"{x} {y}" for x, y in g["coordinates"]
                    ) + ")"
                    cur.execute(
                        """INSERT INTO grid_line (line_id,from_bus,to_bus,geom)
                           VALUES (%s,%s,%s,ST_SetSRID(ST_GeomFromText(%s),4326))
                           ON CONFLICT (line_id) DO UPDATE SET geom=EXCLUDED.geom""",
                        (int(ln), int(net.line.at[ln, "from_bus"]),
                         int(net.line.at[ln, "to_bus"]), wkt),
                    )
            con.commit()
        LOG.info("wrote %d buses / %d lines to PostGIS", len(net.bus), len(net.line))

    def write_fire_perimeter(self, run_uid: str, step: int, sim_min: float,
                             cma, name: str = "socal") -> None:
        """Write the current fire footprint as a MultiPolygon (cell squares)."""
        from .cma import BURNED_OUT, BURNING

        raster = cma.raster
        nrows, ncols = raster.shape
        minlon, minlat, maxlon, maxlat = raster.bounds
        dlon = (maxlon - minlon) / ncols
        dlat = (maxlat - minlat) / nrows
        fire = (cma.state == BURNING) | (cma.state == BURNED_OUT)
        polys = []
        for (r, c) in np.argwhere(fire):
            lon0 = minlon + c * dlon
            lat1 = maxlat - r * dlat
            lon1 = lon0 + dlon
            lat0 = lat1 - dlat
            polys.append(
                f"(({lon0} {lat0},{lon1} {lat0},{lon1} {lat1},"
                f"{lon0} {lat1},{lon0} {lat0}))"
            )
        if not polys:
            return
        wkt = "MULTIPOLYGON(" + ",".join(polys) + ")"
        with _connect(self.dsn) as con:
            with con.cursor() as cur:
                cur.execute(
                    """INSERT INTO fire_perimeter (run_uid,step,sim_min,geom)
                       VALUES (%s,%s,%s,ST_SetSRID(ST_GeomFromText(%s),4326))
                       ON CONFLICT (run_uid,step) DO UPDATE SET geom=EXCLUDED.geom""",
                    (run_uid, step, sim_min, wkt),
                )
            con.commit()
