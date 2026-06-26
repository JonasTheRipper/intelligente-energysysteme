"""CLI: build the SoCal raster + load the grid geometry into PostGIS.

Used by the ``gis-loader`` service in ``docker-compose.yml`` (and runnable
standalone once a PostGIS server is reachable)::

    python -m wildfire_cma.postgis_load --raster-rows 600 --raster-cols 760

It is a thin orchestration wrapper around :mod:`wildfire_cma.postgis`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOG = logging.getLogger("wildfire_cma.postgis_load")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raster-rows", type=int, default=600)
    ap.add_argument("--raster-cols", type=int, default=760)
    ap.add_argument("--raster-name", default="socal")
    ap.add_argument(
        "--grid-json",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "socal_grid", "socal_grid.json",
        ),
    )
    ap.add_argument("--dsn", default=None, help="override the PostGIS DSN")
    ap.add_argument("--skip-grid", action="store_true",
                    help="only stage the raster, not the grid geometry")
    a = ap.parse_args(argv)

    from wildfire_cma.gis import synthetic_socal
    from wildfire_cma.postgis import PostGIS, default_dsn

    pg = PostGIS(a.dsn or default_dsn())
    LOG.info("initialising PostGIS schema ...")
    pg.init_schema()

    LOG.info("building synthetic SoCal raster %dx%d ...",
             a.raster_rows, a.raster_cols)
    raster = synthetic_socal(nrows=a.raster_rows, ncols=a.raster_cols)
    pg.write_raster(raster, name=a.raster_name)

    if not a.skip_grid:
        if os.path.exists(a.grid_json):
            LOG.info("loading grid geometry from %s ...", a.grid_json)
            import pandapower as pp

            net = pp.from_json(a.grid_json)
            pg.write_grid(net)
        else:
            LOG.warning("grid json not found at %s; skipping grid load",
                        a.grid_json)

    LOG.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
