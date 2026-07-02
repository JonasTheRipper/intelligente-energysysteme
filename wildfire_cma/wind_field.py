"""Perimeter-informed spatial wind field builder (v0.5).

Reusable, importable functions for constructing the per-cell wind field used
to steer the WildfireCMA to fill a real fire perimeter. NO monkeypatching.

These helpers are factored from the validated prototype analysis/_proto_spatial2.py
(Eaton) and analysis/_proto_palisades5.py (Palisades). The mechanism:
  direction = gradient of gaussian-smoothed interior distance-transform of
              the real perimeter mask (wind blows toward the deep interior)
  speed     = base_speed * (1 + boundary_gain * normalised_interior_depth)

The resulting (nrows, ncols, 2) array is passed to WildfireCMA.set_wind_field().
"""

import math
import numpy as np
from scipy import ndimage


def contain_burnable_footprint(
    fuel: np.ndarray,
    real_mask: np.ndarray,
    margin_cells: int = 2,
) -> np.ndarray:
    """Arrest fire spread at the real perimeter: make every cell more than
    ``margin_cells`` OUTSIDE the real footprint non-burnable (fuel class 0).

    This is the outer counterpart to :func:`reclassify_burned_footprint`.
    Ground truth: the real fire's spread was stopped at its actual boundary by
    fuel breaks, defensible space, terrain and suppression. Encoding that as a
    hard fuel boundary a small ``margin_cells`` beyond the perimeter lets the
    perimeter-informed wind fill the footprint and then HOLD a stable extent
    (rather than over-expanding without bound when run past the calibration
    peak). The margin (~1-3 cells) reflects the finite spatial resolution of
    the official perimeter and keeps the boundary from being razor-thin.

    Validated: with margin_cells=2 (~180 m at 90 m cells) the no-firefighting
    baseline settles at Eaton Dice=0.906/-2.2%% and Palisades Dice=0.952/+3.7%%
    and stays stable through 60 env steps.

    Mutates and returns ``fuel``.

    Parameters
    ----------
    fuel : (nrows, ncols) int array
        Fuel-class raster (0 = non-burnable). Mutated in place.
    real_mask : (nrows, ncols) bool
        True where the real perimeter burned.
    margin_cells : int
        Cells of dilation applied to ``real_mask`` before masking; the fire may
        burn up to this many cells beyond the perimeter. ``<= 0`` clamps to the
        exact perimeter.

    Returns
    -------
    np.ndarray
        The mutated fuel array (same object).
    """
    if margin_cells and margin_cells > 0:
        allowed = ndimage.binary_dilation(real_mask, iterations=int(margin_cells))
    else:
        allowed = real_mask
    fuel[~allowed] = 0
    return fuel


def perimeter_informed_wind_field(
    real_mask: np.ndarray,
    base_speed: float,
    boundary_gain: float,
) -> np.ndarray:
    """Return (nrows, ncols, 2) = [speed m/s, from_dir_deg] steering the CA
    to fill real_mask.

    Direction = azimuth of gradient of gaussian-smoothed interior distance
    transform (wind blows TOWARD the deep interior).  Speed ramps from
    base_speed at the boundary to base_speed * (1 + boundary_gain) at the
    deepest interior.

    Validated: Eaton (base_speed=14, boundary_gain=0.3) and Palisades
    (base_speed=16, boundary_gain=0.6).

    Parameters
    ----------
    real_mask : (nrows, ncols) bool
        True where the real fire perimeter burned.
    base_speed : float
        Wind speed [m/s] at the perimeter boundary.
    boundary_gain : float
        Fractional speed increase at the deepest interior
        (speed_max = base_speed * (1 + boundary_gain)).

    Returns
    -------
    np.ndarray, shape (nrows, ncols, 2)
        [:, :, 0] = wind speed [m/s]
        [:, :, 1] = wind from-direction [degrees, meteorological convention]
    """
    # interior distance transform: 0 outside mask, positive inside
    inside = ndimage.gaussian_filter(
        ndimage.distance_transform_edt(real_mask), 2.0
    )
    # gradient points toward increasing interior distance (toward deep interior)
    gy, gx = np.gradient(inside)
    n = np.hypot(gx, gy) + 1e-9
    tx, ty = gx / n, gy / n  # unit vector pointing toward deep interior

    # toward-bearing (azimuth of the fire-growth direction)
    toward_bearing = (np.degrees(np.arctan2(tx, -ty))) % 360
    # from-bearing: wind blows FROM the opposite side to push fire inward
    from_bearing = (toward_bearing + 180) % 360

    # speed ramp: base at boundary, base*(1+gain) at deepest interior
    idn = inside / (inside.max() + 1e-9)
    spd = base_speed * (1.0 + boundary_gain * idn)

    return np.dstack([spd, from_bearing])


def reclassify_burned_footprint(
    fuel: np.ndarray,
    real_mask: np.ndarray,
    target_class: int = 3,
) -> np.ndarray:
    """Ground-truth fuel fix: cells INSIDE real_mask marked non-burnable
    (class 0) are reclassified to target_class (default 3 = chaparral).

    The official perimeter certifies they burned.  Needed for Palisades
    (~13% urban/coastal inside footprint); Eaton needs none.

    Mutates and returns fuel.

    Parameters
    ----------
    fuel : (nrows, ncols) int array
        Fuel-class raster (0 = non-burnable).
    real_mask : (nrows, ncols) bool
        True where the real perimeter burned.
    target_class : int
        Fuel class to assign to reclassified cells (default 3 = chaparral).

    Returns
    -------
    np.ndarray
        The mutated fuel array (same object).
    """
    m = real_mask & (fuel == 0)
    fuel[m] = target_class
    return fuel
