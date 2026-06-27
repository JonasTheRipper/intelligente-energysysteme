"""Unit tests for the midas_socal.grid_kpis reducer (pure numpy)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "midas_socal"))

import grid_kpis  # noqa: E402
from grid_kpis import GridKpiReducer, split_native_sensors, CUSTOMERS_PER_MW  # noqa: E402


def _readings(n_bus=4, served_loads=(10.0, 20.0, 30.0), vm=1.0,
              bus_oos=(), line_oos=(), health=1.0):
    r = {}
    for i in range(n_bus):
        r[f"Powergrid-0.0-bus-{i}.vm_pu"] = vm
        r[f"Powergrid-0.0-bus-{i}.in_service"] = 0.0 if i in bus_oos else 1.0
    for j, p in enumerate(served_loads):
        r[f"Powergrid-0.0-load-{j}-{j}.p_mw"] = p
    for k in range(3):
        r[f"Powergrid-0.0-line-{k}.in_service"] = 0.0 if k in line_oos else 1.0
    r["Powergrid-0.0-0.health"] = health
    return r


def test_split_groups_by_element():
    g = split_native_sensors(_readings())
    assert len(g["bus_vm"]) == 4
    assert len(g["bus_in"]) == 4
    assert len(g["line_in"]) == 3
    assert len(g["load_p"]) == 3
    assert g["health"] == 1.0


def test_baseline_kpis():
    red = GridKpiReducer(base_served_mw=60.0)
    k = red.reduce(_readings(served_loads=(10, 20, 30)), dt_min=60.0)
    assert abs(k["grid_served_mw"] - 60.0) < 1e-9
    assert abs(k["customers_disconnected"]) < 1e-9
    assert abs(k["min_bus_vm_pu"] - 1.0) < 1e-9
    assert k["failed_buses"] == 0
    assert k["failed_lines"] == 0
    assert k["pf_converged"] == 1.0


def test_load_shed_drives_disconnect_and_saidi():
    red = GridKpiReducer(base_served_mw=60.0)
    # one 30 MW load shed -> 30 MW short -> 6000 customers out
    k = red.reduce(_readings(served_loads=(10, 20, 0)), dt_min=60.0)
    assert abs(k["grid_served_mw"] - 30.0) < 1e-9
    assert abs(k["customers_disconnected"] - 30.0 * CUSTOMERS_PER_MW) < 1e-6
    # SAIDI after one 60-min step = cust_min/total_customers
    expected_saidi = (30.0 * CUSTOMERS_PER_MW * 60.0) / red.total_customers
    assert abs(k["saidi_minutes"] - expected_saidi) < 1e-6


def test_saidi_accrues_across_steps():
    red = GridKpiReducer(base_served_mw=60.0)
    red.reduce(_readings(served_loads=(10, 20, 0)), dt_min=60.0)
    k2 = red.reduce(_readings(served_loads=(10, 20, 0)), dt_min=60.0)
    # two steps of identical outage -> SAIDI doubles
    one = (30.0 * CUSTOMERS_PER_MW * 60.0) / red.total_customers
    assert abs(k2["saidi_minutes"] - 2 * one) < 1e-6


def test_failed_elements_counted():
    red = GridKpiReducer(base_served_mw=60.0)
    k = red.reduce(_readings(bus_oos=(1,), line_oos=(0, 2)), dt_min=60.0)
    assert k["failed_buses"] == 1
    assert k["failed_lines"] == 2


def test_oos_bus_voltage_excluded():
    red = GridKpiReducer(base_served_mw=60.0)
    r = _readings(vm=1.0, bus_oos=(0,))
    r["Powergrid-0.0-bus-0.vm_pu"] = 0.0  # dead bus reports 0
    k = red.reduce(r, dt_min=60.0)
    # the dead bus must not drag min_vm to 0
    assert abs(k["min_bus_vm_pu"] - 1.0) < 1e-9


def test_health_zero_means_not_converged():
    red = GridKpiReducer(base_served_mw=60.0)
    k = red.reduce(_readings(health=0.0), dt_min=60.0)
    assert k["pf_converged"] == 0.0
