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
# v0.3 firefighter retardant line; surfaced as fire_code==3 so the timelapse can
# tint it. Absent from any v0.2 run (no firefighter), so older runs are unchanged.
SUPPRESSED = 3
# v0.4 ground containment line / point protection; surfaced as fire_code==5.
CONTAINED = 5

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


def latest_instance_id(con, ph: str = "?") -> Optional[int]:
    """Newest ``experiment_run_instances.id`` in the store, or None if empty.

    Re-running the same experiment file writes a SECOND set of phases with the
    SAME uids into the same database. Every reader here filters on
    ``experiment_run_phases.uid`` / ``.number`` only, so without an instance
    filter two runs come back merged -- and because env rows are ordered by
    ``simtime_ticks`` first, their steps interleave rather than concatenate.
    Nothing errors; the data is just silently wrong.

    Callers default to this newest instance, which is almost always the run you
    just finished. Pass an explicit ``instance_id`` to pin an older one, or
    ``instance_id=0`` to opt out and read across every run (the pre-existing
    behaviour).
    """
    try:
        cur = con.cursor()
        cur.execute("SELECT MAX(id) FROM experiment_run_instances")
        row = cur.fetchone()
    except Exception:
        # Partial / hand-built stores (and the test fixtures) may not carry the
        # instance table at all. Nothing to disambiguate then -- fall through to
        # the unfiltered behaviour rather than failing the read.
        return None
    return int(row[0]) if row and row[0] is not None else None


def _instance_clause(con, ph: str, instance_id: Optional[int]):
    """Return ``(sql_fragment, params)`` restricting rows to one run instance."""
    if instance_id == 0:
        return "", []
    resolved = instance_id if instance_id else latest_instance_id(con, ph)
    if resolved is None:
        return "", []
    return f" AND p.experiment_run_instance_id = {ph}", [int(resolved)]


def list_phases(store_uri: str) -> List[dict]:
    """List the palaestrAI phases present in a store, in run order.

    Returns ``[{"number": int, "uid": str}, ...]`` sorted by ``number``. A
    single-phase v0.2 run yields one entry; the v0.3 A/B run yields two
    (``phase_0_no_ff`` then ``phase_1_with_ff``). Callers use this to discover
    the phase uids/indices to pass to :func:`read_run`.

    palaestrAI maps a stored ``world_states`` row to its phase via
    ``environments.experiment_run_phase_id -> experiment_run_phases`` (see
    ``palaestrai/store/database_model.py``); this is a plain SQL join, so no
    palaestrai import is required.
    """
    con, _ph = _connect(store_uri)
    try:
        cur = con.cursor()
        # Only phases that actually have stored world_states rows.
        cur.execute(
            "SELECT DISTINCT p.number, p.uid "
            "FROM experiment_run_phases p "
            "JOIN environments e ON e.experiment_run_phase_id = p.id "
            "JOIN world_states ws ON ws.environment_id = e.id "
            "ORDER BY p.number"
        )
        rows = cur.fetchall()
    finally:
        con.close()
    return [{"number": int(n), "uid": str(u)} for (n, u) in rows]


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


def _fetch_env_rows(
    con,
    env_uid: str,
    ph: str = "?",
    phase_uid: Optional[str] = None,
    phase_index: Optional[int] = None,
    instance_id: Optional[int] = None,
) -> List[Tuple[int, str]]:
    """Return ``[(id, state_dump), ...]`` for one environment, in step order.

    ``ph`` is the driver placeholder (``?`` sqlite, ``%s`` psycopg2).

    When ``phase_uid`` or ``phase_index`` is given, the rows are restricted to
    that single palaestrAI phase by joining
    ``environments.experiment_run_phase_id -> experiment_run_phases`` and
    filtering on ``experiment_run_phases.uid`` (``phase_uid``) or
    ``experiment_run_phases.number`` (``phase_index``). With neither set the
    behaviour is unchanged -- every row for ``env_uid`` across all phases (a
    single-phase v0.2 run has exactly one), preserving back-compat.
    """
    params: List = [env_uid]
    join = ""
    where = f"e.uid = {ph}"
    # Resolve the instance filter first: it is empty for stores with no
    # experiment_run_instances table, and the phase JOIN must only be added when
    # something needs it (hand-built fixtures carry neither table).
    clause, extra = _instance_clause(con, ph, instance_id)
    if phase_uid is not None or phase_index is not None or clause:
        join = "JOIN experiment_run_phases p ON p.id = e.experiment_run_phase_id "
        if phase_uid is not None:
            where += f" AND p.uid = {ph}"
            params.append(phase_uid)
        elif phase_index is not None:
            where += f" AND p.number = {ph}"
            params.append(int(phase_index))
        where += clause
        params.extend(extra)
    q = (
        "SELECT ws.id, ws.state_dump FROM world_states ws "
        "JOIN environments e ON e.id = ws.environment_id "
        f"{join}"
        f"WHERE {where} ORDER BY ws.simtime_ticks, ws.id"
    )
    cur = con.cursor()
    cur.execute(q, tuple(params))
    return list(cur.fetchall())


def _bus_voltages(sensors: Dict[str, np.ndarray]) -> np.ndarray:
    """Collect every per-bus ``*-bus-*.vm_pu`` scalar this step -> 1-D array."""
    vals: List[float] = []
    for uid, v in sensors.items():
        if uid.endswith(".vm_pu") and "-bus-" in uid:
            try:
                vals.append(float(np.asarray(v).ravel()[0]))
            except (IndexError, ValueError):
                pass
    return np.asarray(vals, dtype=np.float64)


def _line_flow_mw(sensors: Dict[str, np.ndarray]) -> Optional[float]:
    """Sum |p_from_mw| over every ``*-line-*`` sensor -> real intertie metric.

    Returns ``None`` when the run stored no line-flow sensors (then the caller
    falls back to a documented proxy). When present this is the total real-power
    throughput on the modelled lines; for this load-pocket scenario it tracks
    the power imported across the network to serve local load.
    """
    found = False
    total = 0.0
    for uid, v in sensors.items():
        if "-line-" in uid and uid.endswith(".p_from_mw"):
            try:
                total += abs(float(np.asarray(v).ravel()[0]))
                found = True
            except (IndexError, ValueError):
                pass
    return total if found else None


def _grid_served_mw(
    sensors: Dict[str, np.ndarray],
    load_uids: Optional[set] = None,
) -> float:
    """Sum load real-power sensors -> served MW for the step.

    With ``load_uids`` given, only those sensors are summed. That matters when
    the number is to be compared against an AGENT's view: SaidiObjective sums
    the ``*-load-*.p_mw`` sensors that agent subscribes to, which on the Eaton
    scenario is 14 loads totalling ~232 MW, whereas the stored dump carries
    every load in the grid (~27,100 MW). Baselining against the wrong set makes
    the offline SAIDI ~117x smaller than the online one -- silently, since both
    are plausible numbers. Keys are matched on the env-stripped suffix, so
    either form of uid may be passed.
    """
    # Match on the element tail, because the two sides carry the environment
    # prefix inconsistently: the stored dump keys keep it
    # ("socal_grid.Powergrid-0.0-load-7-9.p_mw") while a caller may pass either
    # form. Comparing raw strings silently intersects to nothing.
    wanted = None
    if load_uids is not None:
        wanted = {
            u.rsplit(".", 2)[-2] + "." + u.rsplit(".", 1)[-1]
            if u.count(".") >= 2 else u
            for u in load_uids
        }
    total = 0.0
    for uid, v in sensors.items():
        if wanted is not None:
            tail = (
                uid.rsplit(".", 2)[-2] + "." + uid.rsplit(".", 1)[-1]
                if uid.count(".") >= 2 else uid
            )
            if tail not in wanted:
                continue
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
    phase_uid: Optional[str] = None,
    phase_index: Optional[int] = None,
    instance_id: Optional[int] = None,
    load_uids: Optional[set] = None,
) -> Tuple[List[dict], dict]:
    """Reconstruct ``(snaps, meta)`` for ``render()`` from the store.

    ``snaps`` and ``meta`` match the structures produced by
    :func:`analysis.make_timelapse.capture`, so the v0.1 ``render()`` consumes
    them unchanged.

    Phase selection
    ---------------
    A single store may hold MORE than one palaestrAI phase (the v0.3 A/B run
    writes ``phase_0_no_ff`` and ``phase_1_with_ff`` to one DB). Pass
    ``phase_uid`` (e.g. ``"phase_0_no_ff"``) or ``phase_index`` (0/1) to read a
    SINGLE phase. With neither set, behaviour is unchanged: every row for the
    env across all phases (a v0.2 single-phase run has exactly one), so existing
    callers (``firefighter_report.py``, the v0.2 timelapse) keep working. Use
    :func:`list_phases` to discover the phases present.

    New grid metrics (per snap)
    ---------------------------
    * ``vmin_pu`` / ``vmean_pu`` -- min & mean across every stored
      ``*-bus-*.vm_pu`` sensor that step (NaN if none stored).
    * ``intertie_mw`` -- sum of ``|p_from_mw|`` over stored ``*-line-*`` sensors
      when present (a REAL flow metric); otherwise a documented proxy equal to
      the served load MW (power imported to serve local load). ``meta`` carries
      ``intertie_is_proxy: bool`` so plots can label it honestly.
    """
    con, ph = _connect(store_uri)
    try:
        gis_rows = _fetch_env_rows(
            con, gis_uid, ph, phase_uid, phase_index, instance_id
        )
        grid_rows = _fetch_env_rows(
            con, grid_uid, ph, phase_uid, phase_index, instance_id
        )
    finally:
        con.close()
    if not gis_rows:
        raise ValueError(
            f"no world_states for environment uid={gis_uid!r}"
            + (f" phase_uid={phase_uid!r}" if phase_uid is not None else "")
            + (f" phase_index={phase_index!r}" if phase_index is not None else "")
        )
    first = _sensors_by_suffix(gis_rows[0][1])
    print("AVAILABLE KEYS:")
    for key, value in first.items():
        try:
            print(f"  {key}: shape={np.asarray(value).shape}")
        except Exception:
            print(f"  {key}: {type(value)}")

    nr, nc = (int(x) for x in first["gis.grid_shape"].ravel()[:2])
    bounds = tuple(float(x) for x in first["gis.bounds"].ravel()[:4])
    delta_m = float(first["gis.cell_size_m"].ravel()[0])
    fuel = first["gis.fuel_class"].reshape(nr, nc)
    dem = first["gis.elevation_m"].reshape(nr, nc)

    minlon, minlat, maxlon, maxlat = bounds
    extent = [minlon, maxlon, minlat, maxlat]
    nr, nc = (int(x) for x in first["gis.grid_shape"].ravel()[:2])
    bounds = tuple(float(x) for x in first["gis.bounds"].ravel()[:4])
    delta_m = float(first["gis.cell_size_m"].ravel()[0])
    fuel = first["gis.fuel_class"].reshape(nr, nc)
    dem = first["gis.elevation_m"].reshape(nr, nc)
    minlon, minlat, maxlon, maxlat = bounds
    extent = [minlon, maxlon, minlat, maxlat]

    # --- per-step grid metrics from the grid env (if present) -------------
    served_by_step: List[float] = []
    vmin_by_step: List[float] = []
    vmean_by_step: List[float] = []
    lineflow_by_step: List[Optional[float]] = []
    for (_id, dump) in grid_rows:
        gsen = _sensors_by_suffix(dump)
        served_by_step.append(_grid_served_mw(gsen, load_uids))
        volts = _bus_voltages(gsen)
        if volts.size:
            vmin_by_step.append(float(volts.min()))
            vmean_by_step.append(float(volts.mean()))
        else:
            vmin_by_step.append(float("nan"))
            vmean_by_step.append(float("nan"))
        lineflow_by_step.append(_line_flow_mw(gsen))
    base_served = served_by_step[0] if served_by_step else 0.0
    total_customers = max(1.0, base_served) * CUSTOMERS_PER_MW
    # Real line-flow intertie iff ANY grid step stored line sensors; else proxy.
    intertie_is_proxy = not any(x is not None for x in lineflow_by_step)

    # --- per-step spatial frames + derived KPIs ---------------------------
    snaps: List[dict] = []
    cum_customer_min = 0.0
    for i, (_id, dump) in enumerate(gis_rows):
        sm = _sensors_by_suffix(dump)
        S = sm["gis.cell_state"].reshape(nr, nc)
        fire_code = np.zeros((nr, nc), dtype=np.int8)
        fire_code[S == BURNING] = 1
        fire_code[S == BURNED_OUT] = 2
        # v0.3: firefighter retardant lines (absent in v0.2 runs -> stays 0).
        fire_code[S == SUPPRESSED] = 3
        # v0.4: CONTAINED ground line / point protection (code 5; absent in
        # v0.2/v0.3 runs -> stays 0). Kept distinct so renderers colour
        # air-retardant vs ground-containment tactics apart.
        fire_code[S == CONTAINED] = 5
        suppressed_n = int((S == SUPPRESSED).sum())
        contained_n = int((S == CONTAINED).sum())
        wind = sm.get("gis.wind_field", np.array([0.0, 0.0]))
        wind_speed = float(np.asarray(wind).ravel()[0])

        # v0.8 structural telemetry. Absent in pre-v0.8 stores (and in any run
        # whose StoreDumpTrimmer does not keep the gis.houses_* suffixes), in
        # which case these stay 0 and any houses-based metric reads as "no
        # settlement" rather than crashing.
        def _scalar(key: str) -> float:
            v = sm.get(key)
            return float(np.asarray(v).ravel()[0]) if v is not None and np.asarray(v).size else 0.0

        houses_total = _scalar("gis.houses_total")
        houses_burned_step = _scalar("gis.houses_burned_this_step")
        houses_burned_total = _scalar("gis.houses_burned_total")

        served = served_by_step[i] if i < len(served_by_step) else base_served
        disconnected = float(np.clip(base_served - served, 0.0, base_served))
        cust_disc = disconnected * CUSTOMERS_PER_MW
        cum_customer_min += cust_disc * env_step_min
        saidi = cum_customer_min / total_customers if total_customers > 0 else 0.0

        vmin_pu = vmin_by_step[i] if i < len(vmin_by_step) else float("nan")
        vmean_pu = vmean_by_step[i] if i < len(vmean_by_step) else float("nan")
        raw_flow = lineflow_by_step[i] if i < len(lineflow_by_step) else None
        # Real line-flow when stored; else documented proxy = served load MW
        # (power imported across the network to serve the still-energised load).
        intertie_mw = served if raw_flow is None else raw_flow

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
            "houses_total": houses_total,
            "houses_burned_this_step": houses_burned_step,
            "houses_burned_total": houses_burned_total,
            "served_mw": served,
            # alias for callers that prefer the "consumed/served load" name.
            "load_mw": served,
            "failed_bus_n": 0,
            "failed_line_n": 0,
            "cust_disc": cust_disc,
            # v0.3: active retardant-line cells this step (0 for v0.2 runs).
            "suppressed_n": suppressed_n,
            # v0.4: active CONTAINED ground-line / protected-asset cells.
            "contained_n": contained_n,
            # v0.3 grid metrics for the comparison plots.
            "vmin_pu": vmin_pu,
            "vmean_pu": vmean_pu,
            "intertie_mw": intertie_mw,
        })

    meta = {
        "extent": extent,
        "burnable": (fuel != 0).astype(float),
        "fuel": fuel,
        "dem": dem,
        "delta_m": delta_m,
        "all_lines": {},          # line geo not stored; left panel needs none
        "base_served": base_served,
        # True when intertie_mw is the served-load proxy (no line sensors
        # stored); False when it is a real summed line-flow metric.
        "intertie_is_proxy": intertie_is_proxy,
    }
    return snaps, meta


def read_agent_objectives(
    store_uri: str,
    agent_name: str,
    phase_uid: Optional[str] = None,
    phase_index: Optional[int] = None,
    instance_id: Optional[int] = None,
) -> List[dict]:
    """Read an agent's per-decision objective (reward) trace from the store.

    Returns ``[{"episode": int, "objective": float, "done": bool,
    "grid_tick": Optional[int]}, ...]`` in decision order for the named agent,
    optionally restricted to a single phase (via ``phase_uid`` or
    ``phase_index``). Used by :mod:`analysis.drl_firefighter_report` to plot the
    DRL firefighter's SAIDI-reward learning curve across training/eval episodes.

    palaestrAI stores one ``muscle_actions`` row per agent decision; the
    ``objective`` column is the scalar the agent's Objective returned that step
    (for the DRL firefighter, ``-delta_saidi / SAIDI_SCALE``). Phase filtering
    joins ``agents.experiment_run_phase_id -> experiment_run_phases`` -- a real
    filter, so multi-phase stores return only the requested phase's rows.
    """
    con, ph = _connect(store_uri)
    try:
        params: List = [agent_name]
        where = f"a.name = {ph}"
        join = ""
        clause, extra = _instance_clause(con, ph, instance_id)
        if phase_uid is not None or phase_index is not None or clause:
            join = (
                "JOIN experiment_run_phases p "
                "ON p.id = a.experiment_run_phase_id "
            )
            if phase_uid is not None:
                where += f" AND p.uid = {ph}"
                params.append(phase_uid)
            elif phase_index is not None:
                where += f" AND p.number = {ph}"
                params.append(int(phase_index))
            where += clause
            params.extend(extra)
        q = (
            "SELECT ma.episode, ma.objective, ma.done, ma.simtimes "
            "FROM muscle_actions ma "
            "JOIN agents a ON a.id = ma.agent_id "
            f"{join}"
            f"WHERE {where} "
            "ORDER BY ma.id"
        )
        cur = con.cursor()
        cur.execute(q, tuple(params))
        rows = cur.fetchall()
    finally:
        con.close()

    out: List[dict] = []
    for episode, objective, done, simtimes in rows:
        grid_tick: Optional[int] = None
        payload = (
            json.loads(simtimes) if isinstance(simtimes, str) else simtimes
        )
        if isinstance(payload, dict):
            grid = payload.get("socal_grid") or {}
            t = grid.get("simtime_ticks")
            if t is not None:
                grid_tick = int(t)
        out.append(
            {
                "episode": int(episode) if episode is not None else 0,
                "objective": (
                    float(objective) if objective is not None else float("nan")
                ),
                "done": bool(done),
                "grid_tick": grid_tick,
            }
        )
    return out
