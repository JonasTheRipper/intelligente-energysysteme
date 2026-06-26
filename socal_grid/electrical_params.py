#!/usr/bin/env python3
"""Standard per-voltage electrical parameters for overhead AC transmission/sub-transmission
lines, used to build pandapower lines from geometry + voltage class.

Values are typical/representative per-km positive-sequence parameters for ACSR overhead
conductors at each voltage class. Sources: typical utility planning values and
pandapower/PyPSA standard ranges. These are approximations suitable for a non-exact model.

  r_ohm_per_km : series resistance
  x_ohm_per_km : series reactance
  c_nf_per_km  : shunt capacitance
  max_i_ka     : thermal current rating (per circuit)

We map every CEC voltage value to the nearest standard class.
"""

# Representative parameters by nominal voltage class (kV)
# Higher voltages -> bundled conductors -> lower x, higher capacitance, higher current rating.
LINE_PARAMS = {
    500: dict(r=0.018, x=0.27, c=13.0, imax=3.6),   # bundled 500 kV
    287: dict(r=0.035, x=0.32, c=11.0, imax=2.2),
    230: dict(r=0.050, x=0.40, c=9.5,  imax=1.6),
    220: dict(r=0.050, x=0.40, c=9.5,  imax=1.6),
    161: dict(r=0.075, x=0.42, c=9.0,  imax=1.1),
    138: dict(r=0.090, x=0.43, c=8.8,  imax=0.95),
    115: dict(r=0.100, x=0.40, c=9.0,  imax=0.80),
    92:  dict(r=0.130, x=0.41, c=8.5,  imax=0.62),
    70:  dict(r=0.170, x=0.40, c=8.5,  imax=0.50),
    69:  dict(r=0.170, x=0.40, c=8.5,  imax=0.50),
    66:  dict(r=0.180, x=0.40, c=8.5,  imax=0.48),
    60:  dict(r=0.200, x=0.39, c=8.5,  imax=0.44),
    55:  dict(r=0.220, x=0.39, c=8.4,  imax=0.40),
    34.5:dict(r=0.340, x=0.37, c=8.6,  imax=0.30),
    34:  dict(r=0.340, x=0.37, c=8.6,  imax=0.30),
    33:  dict(r=0.350, x=0.37, c=8.6,  imax=0.29),
}

# Underground cables: higher capacitance, lower reactance, lower current rating per circuit.
LINE_PARAMS_UG = {
    230: dict(r=0.030, x=0.18, c=180.0, imax=1.0),
    220: dict(r=0.030, x=0.18, c=180.0, imax=1.0),
    138: dict(r=0.045, x=0.16, c=200.0, imax=0.75),
    115: dict(r=0.050, x=0.15, c=210.0, imax=0.65),
    69:  dict(r=0.080, x=0.13, c=230.0, imax=0.45),
    66:  dict(r=0.085, x=0.13, c=230.0, imax=0.43),
    60:  dict(r=0.095, x=0.13, c=230.0, imax=0.40),
    34.5:dict(r=0.160, x=0.11, c=260.0, imax=0.28),
    34:  dict(r=0.160, x=0.11, c=260.0, imax=0.28),
    33:  dict(r=0.165, x=0.11, c=260.0, imax=0.27),
}

# Standard voltage classes we collapse the raw CEC voltages onto.
STD_VOLTAGES = sorted(LINE_PARAMS.keys())

def nearest_voltage(kv):
    """Map a raw kV value to the nearest standard class."""
    if kv is None:
        return None
    try:
        kv = float(kv)
    except (TypeError, ValueError):
        return None
    return min(STD_VOLTAGES, key=lambda v: abs(v - kv))

def get_line_params(kv_std, underground=False):
    table = LINE_PARAMS_UG if underground else LINE_PARAMS
    if kv_std in table:
        return table[kv_std]
    # fallback to nearest available in the chosen table
    keys = sorted(table.keys())
    k = min(keys, key=lambda v: abs(v - kv_std))
    return table[k]


# Transformer ratings (MVA) chosen by the higher voltage side of a co-located bus pair.
# We create transformers between adjacent voltage levels present at the same site.
def transformer_rating_mva(hv_kv, lv_kv):
    if hv_kv >= 500:
        return 1000.0
    if hv_kv >= 220:
        return 500.0
    if hv_kv >= 115:
        return 200.0
    if hv_kv >= 60:
        return 100.0
    return 40.0
