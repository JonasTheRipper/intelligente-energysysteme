"""System test: the v0.2 store-only timelapse pipeline.

Builds a *tiny self-contained* palaestrAI-style sqlite store (no full run, no
MIDAS), then verifies that:

  1. ``analysis.store_readers.read_run`` reconstructs the per-step ``snaps`` +
     ``meta`` from the ``world_states`` rows -- fire grid from the ``gis_world``
     ``gis.cell_state`` sensor and served-MW / SAIDI from the ``socal_grid``
     load ``p_mw`` sensors;
  2. the unchanged v0.1 ``analysis.make_timelapse.render`` consumes them and
     writes a GIF.

Marked *slow* only because ``render`` pulls in matplotlib's Agg animation
writer; the store decode itself is numpy-only.
"""

import json
import os
import sqlite3

import numpy as np
import pytest


def _sensor(uid, value):
    """A jsonpickle-style SensorInformation dict (value as a plain list)."""
    return {
        "py/object": "palaestrai.agent.sensor_information.SensorInformation",
        "py/state": {"uid": uid, "value": list(np.asarray(value, float).ravel()),
                     "space": None, "value_ids": None},
    }


def _gis_dump(nr, nc, fuel, dem, cell_state, wind=(12.0, 225.0)):
    sensors = [
        _sensor("gis_world.gis.grid_shape", [nr, nc]),
        _sensor("gis_world.gis.bounds", [-120.0, 33.0, -118.0, 35.0]),
        _sensor("gis_world.gis.cell_size_m", [100.0]),
        _sensor("gis_world.gis.fuel_class", fuel),
        _sensor("gis_world.gis.elevation_m", dem),
        _sensor("gis_world.gis.cell_state", cell_state),
        _sensor("gis_world.gis.wind_field", list(wind)),
    ]
    return json.dumps(sensors)


def _grid_dump(load_p_mws):
    sensors = [
        _sensor(f"socal_grid.Powergrid-0.0-load-{i}-{i}.p_mw", [p])
        for i, p in enumerate(load_p_mws)
    ]
    return json.dumps(sensors)


def _build_store(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE environments (id INTEGER PRIMARY KEY, uid TEXT)")
    con.execute(
        "CREATE TABLE world_states (id INTEGER PRIMARY KEY, walltime TEXT, "
        "simtime_ticks INTEGER, simtime_timestamp TEXT, episode INTEGER, "
        "state_dump TEXT, setpoints TEXT, done INTEGER, environment_id INTEGER)"
    )
    con.execute("INSERT INTO environments VALUES (1, 'gis_world')")
    con.execute("INSERT INTO environments VALUES (2, 'socal_grid')")

    nr, nc = 10, 12
    fuel = np.full(nr * nc, 3.0)            # all burnable
    dem = np.zeros(nr * nc)

    # step 1: nothing burning. step 2: one cell burning, one burned out.
    s1 = np.zeros(nr * nc); s2 = np.zeros(nr * nc)
    s2[nc * 5 + 6] = 1.0                     # BURNING
    s2[nc * 5 + 7] = 2.0                     # BURNED_OUT

    rows = [
        (1, 0, s1), (2, 1, s2),              # gis_world (env 1)
    ]
    wid = 1
    for env1_id, tick, S in rows:
        con.execute(
            "INSERT INTO world_states (id, simtime_ticks, episode, state_dump, "
            "done, environment_id) VALUES (?,?,?,?,?,?)",
            (wid, tick, 0, _gis_dump(nr, nc, fuel, dem, S), 0, 1),
        )
        wid += 1
    # socal_grid (env 2): served MW drops as load is shed in step 2.
    for tick, loads in [(0, [100.0, 100.0, 100.0]), (1, [100.0, 0.0, 100.0])]:
        con.execute(
            "INSERT INTO world_states (id, simtime_ticks, episode, state_dump, "
            "done, environment_id) VALUES (?,?,?,?,?,?)",
            (wid, tick, 0, _grid_dump(loads), 0, 2),
        )
        wid += 1
    con.commit()
    con.close()


def test_read_run_reconstructs_snaps(tmp_path):
    from analysis.store_readers import read_run

    db = str(tmp_path / "store.db")
    _build_store(db)
    snaps, meta = read_run(f"sqlite:///{db}")

    assert len(snaps) == 2
    assert meta["fuel"].shape == (10, 12)
    assert meta["dem"].shape == (10, 12)
    assert meta["extent"] == [-120.0, -118.0, 33.0, 35.0]
    assert meta["base_served"] == pytest.approx(300.0)

    # step 1: no fire; step 2: one burning + one burned-out cell.
    assert int((snaps[0]["fire_code"] != 0).sum()) == 0
    assert int((snaps[1]["fire_code"] == 1).sum()) == 1
    assert int((snaps[1]["fire_code"] == 2).sum()) == 1

    # served MW dropped 300 -> 200 in step 2 -> 100 MW shed -> customers out.
    assert snaps[1]["served_mw"] == pytest.approx(200.0)
    assert snaps[1]["cust_disc"] == pytest.approx(100.0 * 200.0)
    assert snaps[1]["saidi"] > 0.0


def test_render_writes_gif(tmp_path):
    from analysis.store_readers import read_run
    from analysis.make_timelapse import render

    db = str(tmp_path / "store.db")
    _build_store(db)
    snaps, meta = read_run(f"sqlite:///{db}")

    outdir = str(tmp_path / "out")
    gif_path, _mp4 = render(snaps, meta, outdir=outdir, stride=1, fps=4)
    assert os.path.exists(gif_path)
    assert os.path.getsize(gif_path) > 0
