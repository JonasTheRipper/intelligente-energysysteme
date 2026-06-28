"""Unit tests for the v0.3 phase-aware store reader + plane-icon helper.

These are numpy-only: they build a tiny self-contained sqlite store that mimics
the palaestrAI schema relationship a 2-phase A/B run produces
(``world_states.environment_id -> environments.experiment_run_phase_id ->
experiment_run_phases``; see ``palaestrai/store/database_model.py``) and assert
that :func:`analysis.store_readers.read_run` can pull a SINGLE phase, that
omitting the phase preserves the v0.2 back-compat behaviour, and that the new
grid-metric snap keys are present. No palaestrAI / MIDAS import, no live sim.
"""

import json
import sqlite3

import numpy as np
import pytest


# cell-state codes (mirror palaestrai_socal.spaces / store_readers)
UNBURNED, BURNING, BURNED_OUT, SUPPRESSED = 0, 1, 2, 3


def _sensor(uid, value):
    return {
        "py/object": "palaestrai.agent.sensor_information.SensorInformation",
        "py/state": {"uid": uid, "value": list(np.asarray(value, float).ravel()),
                     "space": None, "value_ids": None},
    }


def _gis_dump(nr, nc, fuel, dem, cell_state, wind=(8.0, 45.0)):
    return json.dumps([
        _sensor("gis_world.gis.grid_shape", [nr, nc]),
        _sensor("gis_world.gis.bounds", [-118.14, 34.13, -118.04, 34.23]),
        _sensor("gis_world.gis.cell_size_m", [50.0]),
        _sensor("gis_world.gis.fuel_class", fuel),
        _sensor("gis_world.gis.elevation_m", dem),
        _sensor("gis_world.gis.cell_state", cell_state),
        _sensor("gis_world.gis.wind_field", list(wind)),
    ])


def _grid_dump(load_p_mws, vm_pus=None, line_flows=None):
    sensors = [
        _sensor(f"socal_grid.Powergrid-0.0-load-{i}-{i}.p_mw", [p])
        for i, p in enumerate(load_p_mws)
    ]
    if vm_pus is not None:
        sensors += [
            _sensor(f"socal_grid.Powergrid-0.0-bus-{i}.vm_pu", [v])
            for i, v in enumerate(vm_pus)
        ]
    if line_flows is not None:
        sensors += [
            _sensor(f"socal_grid.Powergrid-0.0-line-{i}-{i + 1}.p_from_mw", [f])
            for i, f in enumerate(line_flows)
        ]
    return sensors and json.dumps(sensors) or json.dumps([])


def _create_schema(con):
    con.execute("CREATE TABLE experiment_run_phases "
                "(id INTEGER PRIMARY KEY, uid TEXT, number INTEGER)")
    con.execute("CREATE TABLE environments (id INTEGER PRIMARY KEY, uid TEXT, "
                "experiment_run_phase_id INTEGER)")
    con.execute(
        "CREATE TABLE world_states (id INTEGER PRIMARY KEY, walltime TEXT, "
        "simtime_ticks INTEGER, simtime_timestamp TEXT, episode INTEGER, "
        "state_dump TEXT, setpoints TEXT, done INTEGER, environment_id INTEGER)")


def _build_ab_store(path, with_grid_sensors=True):
    """Two-phase store: phase A (no suppression), phase B (a retardant line)."""
    con = sqlite3.connect(path)
    _create_schema(con)
    con.execute("INSERT INTO experiment_run_phases VALUES (1,'phase_0_no_ff',0)")
    con.execute("INSERT INTO experiment_run_phases VALUES (2,'phase_1_with_ff',1)")
    # env ids: 1=A.gis 2=A.grid 3=B.gis 4=B.grid
    con.execute("INSERT INTO environments VALUES (1,'gis_world',1)")
    con.execute("INSERT INTO environments VALUES (2,'socal_grid',1)")
    con.execute("INSERT INTO environments VALUES (3,'gis_world',2)")
    con.execute("INSERT INTO environments VALUES (4,'socal_grid',2)")

    nr, nc = 8, 10
    fuel = np.full(nr * nc, 3.0)
    dem = np.zeros(nr * nc)

    # phase A frames: no fire, then a burning + burned-out cell. NO suppression.
    a1 = np.zeros(nr * nc)
    a2 = np.zeros(nr * nc); a2[nc * 4 + 5] = BURNING; a2[nc * 4 + 6] = BURNED_OUT
    # phase B frames: same fire but with a retardant line (SUPPRESSED) in frame 2.
    b1 = np.zeros(nr * nc)
    b2 = np.zeros(nr * nc); b2[nc * 4 + 5] = BURNING
    b2[nc * 3 + 5] = SUPPRESSED; b2[nc * 3 + 6] = SUPPRESSED

    wid = 1

    def add(env_id, tick, dump):
        nonlocal wid
        con.execute(
            "INSERT INTO world_states (id, simtime_ticks, episode, state_dump, "
            "done, environment_id) VALUES (?,?,?,?,?,?)",
            (wid, tick, 0, dump, 0, env_id))
        wid += 1

    vmA = [1.00, 0.99, 0.98] if with_grid_sensors else None
    vmB = [1.00, 0.995, 0.99] if with_grid_sensors else None
    flA = [10.0, 5.0] if with_grid_sensors else None
    flB = [12.0, 6.0] if with_grid_sensors else None

    # phase A: gis (env1) + grid (env2). Grid sheds load in step 2.
    add(1, 0, _gis_dump(nr, nc, fuel, dem, a1))
    add(1, 1, _gis_dump(nr, nc, fuel, dem, a2))
    add(2, 0, _grid_dump([100.0, 100.0, 100.0], vmA, flA))
    add(2, 1, _grid_dump([100.0, 0.0, 100.0], vmA, flA))
    # phase B: gis (env3) + grid (env4). Less load shed (firefighters help).
    add(3, 0, _gis_dump(nr, nc, fuel, dem, b1))
    add(3, 1, _gis_dump(nr, nc, fuel, dem, b2))
    add(4, 0, _grid_dump([100.0, 100.0, 100.0], vmB, flB))
    add(4, 1, _grid_dump([100.0, 100.0, 100.0], vmB, flB))

    con.commit()
    con.close()


def test_list_phases(tmp_path):
    from analysis.store_readers import list_phases
    db = str(tmp_path / "ab.db")
    _build_ab_store(db)
    phases = list_phases(f"sqlite:///{db}")
    assert [p["uid"] for p in phases] == ["phase_0_no_ff", "phase_1_with_ff"]
    assert [p["number"] for p in phases] == [0, 1]


def test_read_single_phase_by_uid(tmp_path):
    from analysis.store_readers import read_run
    db = str(tmp_path / "ab.db")
    _build_ab_store(db)

    snaps_a, _ = read_run(f"sqlite:///{db}", phase_uid="phase_0_no_ff")
    snaps_b, _ = read_run(f"sqlite:///{db}", phase_uid="phase_1_with_ff")

    assert len(snaps_a) == 2 and len(snaps_b) == 2
    # phase A has NO suppressed cells; phase B has a 2-cell retardant line.
    assert all(s["suppressed_n"] == 0 for s in snaps_a)
    assert snaps_b[1]["suppressed_n"] == 2
    # firefighters preserved load in B -> higher served MW in the last step.
    assert snaps_b[1]["served_mw"] > snaps_a[1]["served_mw"]


def test_read_single_phase_by_index_matches_uid(tmp_path):
    from analysis.store_readers import read_run
    db = str(tmp_path / "ab.db")
    _build_ab_store(db)
    by_idx, _ = read_run(f"sqlite:///{db}", phase_index=1)
    by_uid, _ = read_run(f"sqlite:///{db}", phase_uid="phase_1_with_ff")
    assert [s["suppressed_n"] for s in by_idx] == \
           [s["suppressed_n"] for s in by_uid]


def test_no_phase_is_backcompat_concatenation(tmp_path):
    """Omitting the phase reads every row for the env uid (v0.2 behaviour)."""
    from analysis.store_readers import read_run
    db = str(tmp_path / "ab.db")
    _build_ab_store(db)
    snaps_all, _ = read_run(f"sqlite:///{db}")
    # both phases' gis rows (2 + 2) are returned, vs 2 for a single phase.
    snaps_a, _ = read_run(f"sqlite:///{db}", phase_uid="phase_0_no_ff")
    assert len(snaps_all) == 4
    assert len(snaps_a) == 2


def test_new_grid_metric_keys_present(tmp_path):
    from analysis.store_readers import read_run
    db = str(tmp_path / "ab.db")
    _build_ab_store(db, with_grid_sensors=True)
    snaps, meta = read_run(f"sqlite:///{db}", phase_uid="phase_1_with_ff")
    s = snaps[0]
    for key in ("vmin_pu", "vmean_pu", "intertie_mw", "load_mw", "served_mw"):
        assert key in s
    # min <= mean across the three stored bus voltages
    assert s["vmin_pu"] <= s["vmean_pu"]
    # line sensors present -> real intertie metric (NOT a proxy)
    assert meta["intertie_is_proxy"] is False
    assert s["intertie_mw"] == pytest.approx(18.0)  # |12| + |6|


def test_intertie_proxy_when_no_line_sensors(tmp_path):
    from analysis.store_readers import read_run
    db = str(tmp_path / "noline.db")
    _build_ab_store(db, with_grid_sensors=False)
    snaps, meta = read_run(f"sqlite:///{db}", phase_uid="phase_0_no_ff")
    assert meta["intertie_is_proxy"] is True
    # proxy intertie == served MW
    for s in snaps:
        assert s["intertie_mw"] == pytest.approx(s["served_mw"])
    # no bus sensors stored -> vmin/vmean are NaN
    assert np.isnan(snaps[0]["vmin_pu"])


# ----------------------------- plane-icon helper -----------------------------

def test_plane_positions_empty_when_no_new_cells():
    from analysis.plane_icons import plane_positions
    g = np.zeros((6, 6), dtype=bool)
    assert plane_positions(g, g.copy(), 3) == []


def test_plane_positions_zero_planes():
    from analysis.plane_icons import plane_positions
    prev = np.zeros((6, 6), dtype=bool)
    curr = prev.copy(); curr[2, 1:5] = True
    assert plane_positions(prev, curr, 0) == []


def test_plane_positions_caps_at_n_planes_and_is_deterministic():
    from analysis.plane_icons import plane_positions
    prev = np.zeros((6, 8), dtype=bool)
    curr = prev.copy()
    curr[3, 1:7] = True            # 6 new cells along a row
    pos1 = plane_positions(prev, curr, 3)
    pos2 = plane_positions(prev, curr, 3)
    assert pos1 == pos2            # deterministic
    assert len(pos1) == 3          # capped at n_planes
    assert all(r == 3 for (r, c) in pos1)
    # spread across the line: includes both ends
    cols = [c for (_r, c) in pos1]
    assert min(cols) == 1 and max(cols) == 6


def test_plane_positions_returns_all_when_fewer_cells_than_planes():
    from analysis.plane_icons import plane_positions
    prev = np.zeros((5, 5), dtype=bool)
    curr = prev.copy(); curr[1, 1] = True; curr[2, 2] = True
    pos = plane_positions(prev, curr, 5)
    assert sorted(pos) == [(1, 1), (2, 2)]


# ---------------------- comparison renderers (matplotlib) --------------------

def test_grid_metrics_report_builds_figure(tmp_path):
    """Static A/B PNG builder runs end-to-end on the synthetic store."""
    from analysis.store_readers import read_run
    from analysis.grid_metrics_report import build_figure
    db = str(tmp_path / "ab.db")
    _build_ab_store(db)
    sa, ma = read_run(f"sqlite:///{db}", phase_uid="phase_0_no_ff")
    sb, mb = read_run(f"sqlite:///{db}", phase_uid="phase_1_with_ff")
    fig, deltas = build_figure(sa, ma, sb, mb)
    out = str(tmp_path / "metrics.png")
    fig.savefig(out)
    import os
    assert os.path.exists(out) and os.path.getsize(out) > 0
    # firefighters save area in this fixture (phase B burns fewer cells).
    assert deltas["acres_saved"] >= 0.0
    assert deltas["intertie_is_proxy"] is False


def test_comparison_timelapse_writes_gif(tmp_path):
    from analysis.store_readers import read_run
    from analysis.make_comparison_timelapse import render_comparison
    import os
    db = str(tmp_path / "ab.db")
    _build_ab_store(db)
    sa, ma = read_run(f"sqlite:///{db}", phase_uid="phase_0_no_ff")
    sb, mb = read_run(f"sqlite:///{db}", phase_uid="phase_1_with_ff")
    gif, _mp4 = render_comparison(sa, ma, sb, mb, n_planes=3,
                                  outdir=str(tmp_path / "out"), stride=1, fps=4)
    assert os.path.exists(gif) and os.path.getsize(gif) > 0
