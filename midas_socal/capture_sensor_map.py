"""Capture the native MIDAS powergrid sensor/actuator uids at build time.

The DESIGN forbids hard-coding the ``Powergrid-0.0-<element>.<attr>`` uids;
instead we ask the real ``midas_palaestrai.descriptor.Descriptor`` to describe
the SoCal scenario and dump every uid (with its space) to
``midas_socal/socal_sensor_map.json``. Downstream code (grid_kpis reducer,
DamageMapperAgent) reads this map to discover bus/line/load uids rather than
guessing them.

Run:  python midas_socal/capture_sensor_map.py
"""

from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO = os.path.join(_HERE, "socal.yml")
OUT = os.path.join(_HERE, "socal_sensor_map.json")


def _space_str(s) -> str:
    try:
        return s.to_string() if hasattr(s, "to_string") else str(s)
    except Exception:
        return str(s)


def main() -> None:
    from midas_palaestrai.descriptor import Descriptor

    d = Descriptor()
    sensors, actuators, _world_state = d.describe(
        {"name": "socal", "config": [SCENARIO], "silent": True}
    )

    def _uid(x):
        if isinstance(x, dict):
            return x.get("uid", str(x))
        return getattr(x, "uid", str(x))

    def _space_of(x):
        if isinstance(x, dict):
            return _space_str(x.get("space"))
        return _space_str(getattr(x, "space", None))

    sensor_map = {_uid(s): _space_of(s) for s in sensors}
    actuator_map = {_uid(a): _space_of(a) for a in actuators}

    # categorise by element type for convenience
    def _by_kind(uids):
        kinds = {"bus": [], "line": [], "load": [], "sgen": [],
                 "trafo": [], "other": []}
        for u in uids:
            for k in ("bus", "line", "load", "sgen", "trafo"):
                if f"-{k}-" in u:
                    kinds[k].append(u)
                    break
            else:
                kinds["other"].append(u)
        return {k: sorted(v) for k, v in kinds.items()}

    out = {
        "scenario": "socal",
        "n_sensors": len(sensor_map),
        "n_actuators": len(actuator_map),
        "sensor_uids_by_kind": _by_kind(sensor_map.keys()),
        "actuator_uids_by_kind": _by_kind(actuator_map.keys()),
        "sensors": sensor_map,
        "actuators": actuator_map,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {OUT}: {len(sensor_map)} sensors, {len(actuator_map)} actuators")
    for k, v in out["actuator_uids_by_kind"].items():
        print(f"  actuators[{k}]: {len(v)}")
    for k, v in out["sensor_uids_by_kind"].items():
        print(f"  sensors[{k}]: {len(v)}")


if __name__ == "__main__":
    main()
