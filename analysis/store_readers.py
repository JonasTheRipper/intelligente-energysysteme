"""Read a palaestrAI v0.2 run back out of the store for offline analysis.

The v0.2 timelapse is **store-only**: instead of re-running the environment
(v0.1's ``make_timelapse.capture``), it reconstructs every frame from the
``world_states`` rows the two-environment run wrote to the palaestrAI sqlite
store. This module is the reader half; :mod:`analysis.make_timelapse` feeds the
result straight into the unchanged ``render()``.

What the store actually holds
-----------------------------
palaestrAI serialises each environment's ``sensors_available`` list (jsonpickle)
into ``world_states.state_dump`` -- one row per environment per step. Every
sensor is a ``{"py/object": "...SensorInformation", "py/state": {"uid",
"value", ...}}`` dict whose ``value`` is a plain JSON list, so we can decode the
whole thing with the standard ``json`` module (no palaestrai / jsonpickle
import needed -- this module is numpy-only).

* ``gis_world`` rows carry the spatial substrate: ``gis.grid_shape``,
  ``gis.bounds``, ``gis.cell_size_m``, ``gis.fuel_class``, ``gis.elevation_m``,
  ``gis.cell_state`` (the per-step hazard grid) and ``gis.wind_field``.
* ``socal_grid`` rows carry every MIDAS powergrid sensor; we sum the load
  ``p_mw`` sensors to recover the served MW each step (the DamageMapperAgent
  sheds load on fire-affected buses, so served MW drops as the fire grows).

KPIs (served MW shortfall -> customers out -> cumulative SAIDI) are derived here
exactly as the v0.1 environment did (``CUSTOMERS_PER_MW = 200``), so the
reconstructed ``snaps`` are drop-in compatible with the v0.1 ``render()``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np

# cell-state codes (mirror palaestrai_socal.spaces)
UNBURNED, BURNING, BURNED_OUT = 0, 1, 2

# same planning figure the v0.1 environment used to turn MW into customers.
CUSTOMERS_PER_MW = 200.0


def _db_path(store_uri: str) -> str:
    """Accept either a bare path or a ``sqlite:///...`` SQLAlchemy URI."""
    if store_uri.startswith("sqlite:///"):
        return store_uri[len("sqlite:///"):]
    if store_uri.startswith("sqlite://"):
        return store_uri[len("sqlite://"):]
    return store_uri


def _is_postgres(store_uri: str) -> bool:
    return store_uri.startswith("postgresql://") or store_uri.startswith(
        "postgres://"
    )


def _connect(store_uri: str):
    """Open the store for reading.

    Returns ``(connection, paramstyle)`` where ``paramstyle`` is the SQL
    placeholder the driver expects (``?`` for sqlite, ``%s`` for psycopg2). The
    v0.2 store moved from a sqlite file to PostgreSQL+TimescaleDB (see
    ``runtime_pg.conf.yaml`` / V02_STORE_EATON_RESULT.md); both back the same
    ``world_states``/``environments`` schema, so only the connection differs.
    """
    if _is_postgres(store_uri):
        import psycopg2  # local import: only needed for the PG store

        return psycopg2.connect(store_uri), "%s"
    path = _db_path(store_uri)
    if not os.path.exists(path):
        raise FileNotFoundError(f"store db not found: {path}")
    return sqlite3.connect(path), "?"


def _suffix(uid: str) -> str:
    """Drop the ``<env_uid>.`` prefix palaestrAI prepends to sensor uids."""
    return uid.split(".", 1)[1] if "." in uid else uid


def _sensors_by_suffix(state_dump) -> Dict[str, np.ndarray]:
    """Decode a ``world_states.state_dump`` into ``{sensor_suffix: ndarray}``.

    Accepts both the sqlite form (a JSON/jsonpickle *string*) and the
    PostgreSQL form (a ``jsonb`` column, which psycopg2 returns already parsed
    into a Python list/dict).
    """
    payload = json.loads(state_dump) if isinstance(state_dump, str) else state_dump
    out: Dict[str, np.ndarray] = {}
    for s in payload:
        st = s["py/state"] if isinstance(s, dict) and "py/state" in s else s
        uid = st.get("uid")
        val = st.get("value")
        if uid is None or val is None:
            continue
        # gis.* uids keep their full ``gis.<name>`` tail; strip only the env id.
        suf = uid
        if uid.count(".") >= 2 and uid.split(".", 1)[0] in ("gis_world",):
            suf = _suffix(uid)
        out[suf] = np.asarray(val, dtype=np.float64)
    return out


def _fetch_env_rows(con, env_uid: str, ph: str = "?"
                    ) -> List[Tuple[int, str]]:
    """Return ``[(id, state_dump), ...]`` for one environment, in step order.

    ``ph`` is the driver placeholder (``?`` sqlite, ``%s`` psycopg2).
    """
    q = (
        "SELECT ws.id, ws.state_dump FROM world_states ws "
        "JOIN environments e ON e.id = ws.environment_id "
        f"WHERE e.uid = {ph} ORDER BY ws.simtime_ticks, ws.id"
    )
    cur = con.cursor()
    cur.execute(q, (env_uid,))
    return list(cur.fetchall())


def _grid_served_mw(sensors: Dict[str, np.ndarray]) -> float:
    """Sum every load real-power sensor -> total served MW for the step."""
    total = 0.0
    for uid, v in sensors.items():
        if uid.endswith(".p_mw") and "-load-" in uid:
            try:
                total += float(np.asarray(v).ravel()[0])
            except (IndexError, ValueError):
                pass
    return total


def read_run(
    store_uri: str,
    gis_uid: str = "gis_world",
    grid_uid: str = "socal_grid",
    env_step_min: float = 60.0,
) -> Tuple[List[dict], dict]:
    """Reconstruct ``(snaps, meta)`` for ``render()`` from the store.

    ``snaps`` and ``meta`` match the structures produced by
    :func:`analysis.make_timelapse.capture`, so the v0.1 ``render()`` consumes
    them unchanged.
    """
    con, ph = _connect(store_uri)
    try:
        gis_rows = _fetch_env_rows(con, gis_uid, ph)
        grid_rows = _fetch_env_rows(con, grid_uid, ph)
    finally:
        con.close()
    if not gis_rows:
        raise ValueError(f"no world_states for environment uid={gis_uid!r}")

    # --- static geometry from the first gis_world row ---------------------
    first = _sensors_by_suffix(gis_rows[0][1])
    nr, nc = (int(x) for x in first["gis.grid_shape"].ravel()[:2])
    bounds = tuple(float(x) for x in first["gis.bounds"].ravel()[:4])
    delta_m = float(first["gis.cell_size_m"].ravel()[0])
    fuel = first["gis.fuel_class"].reshape(nr, nc)
    dem = first["gis.elevation_m"].reshape(nr, nc)
    minlon, minlat, maxlon, maxlat = bounds
    extent = [minlon, maxlon, minlat, maxlat]

    # --- per-step served MW from the grid env (if present) ----------------
    served_by_step: List[float] = []
    for (_id, dump) in grid_rows:
        served_by_step.append(_grid_served_mw(_sensors_by_suffix(dump)))
    base_served = served_by_step[0] if served_by_step else 0.0
    total_customers = max(1.0, base_served) * CUSTOMERS_PER_MW

    # --- per-step spatial frames + derived KPIs ---------------------------
    snaps: List[dict] = []
    cum_customer_min = 0.0
    for i, (_id, dump) in enumerate(gis_rows):
        sm = _sensors_by_suffix(dump)
        S = sm["gis.cell_state"].reshape(nr, nc)
        fire_code = np.zeros((nr, nc), dtype=np.int8)
        fire_code[S == BURNING] = 1
        fire_code[S == BURNED_OUT] = 2
        wind = sm.get("gis.wind_field", np.array([0.0, 0.0]))
        wind_speed = float(np.asarray(wind).ravel()[0])

        served = served_by_step[i] if i < len(served_by_step) else base_served
        disconnected = float(np.clip(base_served - served, 0.0, base_served))
        cust_disc = disconnected * CUSTOMERS_PER_MW
        cum_customer_min += cust_disc * env_step_min
        saidi = cum_customer_min / total_customers if total_customers > 0 else 0.0

        step = i + 1
        snaps.append({
            "step": step,
            "hour": step * env_step_min / 60.0,
            "day": (step * env_step_min / 60.0) / 24.0,
            "fire_code": fire_code,
            # v0.2 sheds LOAD, not lines: no line-trip set is reconstructed.
            "failed_lines": set(),
            "wind_speed": wind_speed,
            "saidi": saidi,
            "served_mw": served,
            "failed_bus_n": 0,
            "failed_line_n": 0,
            "cust_disc": cust_disc,
        })

    meta = {
        "extent": extent,
        "burnable": (fuel != 0).astype(float),
        "fuel": fuel,
        "dem": dem,
        "delta_m": delta_m,
        "all_lines": {},          # line geo not stored; left panel needs none
        "base_served": base_served,
    }
    return snaps, meta
