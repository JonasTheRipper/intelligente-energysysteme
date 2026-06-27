"""Reducer: native MIDAS ``Powergrid`` sensors -> the v0.1 grid KPI set.

In v0.1 the KPIs were computed *inside* the monolithic environment from a
hand-rolled pandapower ``_runpp()``. In v0.2 the grid is a real
``palaestrai_mosaik.MosaikEnvironment`` stepping a MIDAS ``midas-powergrid``
simulator, which auto-exposes per-element sensors with uids like::

    Powergrid-0.0-bus-<i>.vm_pu
    Powergrid-0.0-bus-<i>.in_service
    Powergrid-0.0-line-<j>.in_service
    Powergrid-0.0-line-<j>.loading_percent
    Powergrid-0.0-load-<eid>-<ppidx>.p_mw
    Powergrid-0.0-<id>.health            (grid-level convergence health)

This module maps that native telemetry back onto the exact v0.1 KPI vector --
``min_bus_vm_pu``, ``mean_bus_vm_pu``, ``grid_served_mw``,
``customers_connected``, ``customers_disconnected``, ``saidi_minutes``,
``failed_buses``, ``failed_lines``, ``pf_converged`` -- so the reward and the
timelapse are byte-for-byte comparable across the v0.1 -> v0.2 refactor. Only
the *source* of the numbers changed (real MIDAS power flow), not their meaning.

Pure numpy: no pandapower / palaestrai import, so it is a fast unit test.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

# Same planning figure as v0.1 environment.py: ~1 customer per 5 kW peak.
CUSTOMERS_PER_MW = 200.0

KPI_NAMES = (
    "min_bus_vm_pu",
    "mean_bus_vm_pu",
    "grid_served_mw",
    "customers_connected",
    "customers_disconnected",
    "saidi_minutes",
    "failed_buses",
    "failed_lines",
    "pf_converged",
)


def _as_float(v) -> float:
    try:
        return float(np.asarray(v).ravel()[0])
    except Exception:
        try:
            return float(v)
        except Exception:
            return 0.0


def _is(uid: str, etype: str, attr: str) -> bool:
    """True if ``uid`` is a ``Powergrid`` sensor of ``etype`` with ``attr``."""
    return (f"-{etype}-" in uid) and uid.endswith("." + attr)


def split_native_sensors(readings: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
    """Group a flat ``{uid: value}`` dict into element buckets.

    Returns ``{"bus_vm": {uid: vm_pu}, "bus_in": {uid: 0/1},
    "line_in": {uid: 0/1}, "load_p": {uid: p_mw}, "health": <float|None>}``.
    """
    bus_vm: Dict[str, float] = {}
    bus_in: Dict[str, float] = {}
    line_in: Dict[str, float] = {}
    load_p: Dict[str, float] = {}
    health: Optional[float] = None
    for uid, val in readings.items():
        if not isinstance(uid, str):
            continue
        if _is(uid, "bus", "vm_pu"):
            bus_vm[uid] = _as_float(val)
        elif _is(uid, "bus", "in_service"):
            bus_in[uid] = _as_float(val)
        elif _is(uid, "line", "in_service"):
            line_in[uid] = _as_float(val)
        elif _is(uid, "load", "p_mw"):
            load_p[uid] = _as_float(val)
        elif uid.endswith(".health"):
            health = _as_float(val)
    return {
        "bus_vm": bus_vm,
        "bus_in": bus_in,
        "line_in": line_in,
        "load_p": load_p,
        "health": health,
    }


class GridKpiReducer:
    """Stateful native-sensor -> v0.1-KPI reducer (carries SAIDI accrual).

    Parameters
    ----------
    base_served_mw:
        the baseline (all-customers-connected) served load in MW. Disconnected
        load is measured as the shortfall from this reference, exactly as v0.1.
    total_customers:
        denominator for SAIDI; defaults to ``base_served_mw * CUSTOMERS_PER_MW``.
    """

    def __init__(
        self,
        base_served_mw: float,
        total_customers: Optional[float] = None,
    ):
        self.base_served_mw = float(base_served_mw)
        self.total_customers = float(
            total_customers
            if total_customers is not None
            else max(1.0, self.base_served_mw) * CUSTOMERS_PER_MW
        )
        self.cum_customer_minutes = 0.0

    def reset(self) -> None:
        self.cum_customer_minutes = 0.0

    def reduce(
        self,
        readings: Mapping[str, object],
        dt_min: float,
        accrue: bool = True,
    ) -> Dict[str, float]:
        """Map one native-sensor snapshot to the v0.1 KPI dict.

        ``dt_min`` is the environment step length in minutes (for SAIDI accrual).
        Set ``accrue=False`` to compute KPIs without advancing cumulative SAIDI
        (e.g. for the baseline step).
        """
        g = split_native_sensors(readings)

        # --- voltages (in-service buses only when in_service is reported) ---
        bus_vm = g["bus_vm"]
        bus_in = g["bus_in"]
        vms = []
        for uid, vm in bus_vm.items():
            in_svc = bus_in.get(uid.rsplit(".", 1)[0] + ".in_service")
            if in_svc is not None and in_svc < 0.5:
                continue
            if vm is None or np.isnan(vm) or vm <= 0.0:
                continue
            vms.append(vm)
        vms_arr = np.asarray(vms, dtype=float)
        min_vm = float(vms_arr.min()) if vms_arr.size else 0.0
        mean_vm = float(vms_arr.mean()) if vms_arr.size else 0.0

        # --- served load = sum of (current) load p_mw -----------------------
        load_vals = np.asarray(list(g["load_p"].values()), dtype=float)
        served_mw = float(load_vals[load_vals > 0].sum()) if load_vals.size else 0.0

        # --- failed elements (in_service == 0) ------------------------------
        failed_buses = int(sum(1 for v in bus_in.values() if v < 0.5))
        failed_lines = int(sum(1 for v in g["line_in"].values() if v < 0.5))

        # --- convergence ----------------------------------------------------
        if g["health"] is not None:
            pf_converged = 1.0 if g["health"] >= 0.5 else 0.0
        else:
            pf_converged = 1.0 if vms_arr.size else 0.0

        # --- customers + SAIDI (identical accrual to v0.1) ------------------
        disconnected_mw = float(np.clip(
            self.base_served_mw - served_mw, 0.0, self.base_served_mw))
        customers_disconnected = disconnected_mw * CUSTOMERS_PER_MW
        customers_connected = max(
            0.0, self.total_customers - customers_disconnected)
        if accrue:
            self.cum_customer_minutes += customers_disconnected * float(dt_min)
        saidi_minutes = (
            self.cum_customer_minutes / self.total_customers
            if self.total_customers > 0 else 0.0
        )

        return {
            "min_bus_vm_pu": min_vm,
            "mean_bus_vm_pu": mean_vm,
            "grid_served_mw": served_mw,
            "customers_connected": customers_connected,
            "customers_disconnected": customers_disconnected,
            "saidi_minutes": saidi_minutes,
            "failed_buses": float(failed_buses),
            "failed_lines": float(failed_lines),
            "pf_converged": pf_converged,
        }
