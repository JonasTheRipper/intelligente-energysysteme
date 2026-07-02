"""Tests for the v0.5 per-cell wind field hook in WildfireCMA (wildfire_cma/cma.py).

Verifies:
1. _phi_wind with _wind_field=None returns the SAME value as pre-v0.5 (bit-for-bit).
2. Setting a non-None wind_field changes _phi_wind output vs scalar theta.
3. set_wind_field() shape guard raises AssertionError on wrong shape.
4. ros() uses the per-cell wind when _wind_field is set.
"""
import math
import numpy as np
import pytest

from wildfire_cma.cma import WildfireCMA, RasterStack, Theta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raster(nrows: int = 10, ncols: int = 12) -> RasterStack:
    """Minimal synthetic raster (flat, uniform chaparral fuel)."""
    fuel = np.full((nrows, ncols), 3, dtype=np.int16)   # chaparral everywhere
    dem = np.zeros((nrows, ncols), dtype=float)
    return RasterStack(
        fuel=fuel,
        dem=dem,
        delta_m=90.0,
        bounds=(-118.5, 34.0, -118.4, 34.1),
    )


def _make_cma(raster: RasterStack, wind_speed: float = 10.0,
              wind_dir_deg: float = 45.0, seed: int = 0) -> WildfireCMA:
    theta = Theta(
        ignition_rc=[(5, 6)],
        wind_speed=wind_speed,
        wind_dir_deg=wind_dir_deg,
        dead_fuel_moisture=0.05,
        kappa=1.5,
    )
    return WildfireCMA(raster, theta, seed=seed)


# ---------------------------------------------------------------------------
# Tests: bit-for-bit identity when _wind_field is None
# ---------------------------------------------------------------------------

class TestPhiWindNullField:
    """_phi_wind with _wind_field=None must be bit-identical to the old scalar path."""

    def _ref_phi_wind(self, cma: WildfireCMA, dr: int, dc: int) -> float:
        """Reference implementation of the old (pre-v0.5) scalar _phi_wind."""
        u = cma.theta.wind_speed
        toward = math.radians((cma.theta.wind_dir_deg + 180.0) % 360.0)
        spread_bearing = math.atan2(dc, -dr)
        cos_align = math.cos(toward - spread_bearing)
        c = 0.25
        return float(math.exp(c * u * max(cos_align, -0.5)))

    def test_null_field_identity_various_directions(self):
        """For all 8 Moore neighbours, _phi_wind(dr,dc) == reference with no field."""
        raster = _make_raster()
        cma = _make_cma(raster, wind_speed=15.0, wind_dir_deg=270.0)
        assert cma._wind_field is None

        moore = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        for dr, dc in moore:
            got = cma._phi_wind(dr, dc)            # no row/col => scalar path
            ref = self._ref_phi_wind(cma, dr, dc)
            assert got == ref, (
                f"dr={dr} dc={dc}: got {got} != ref {ref} with no wind_field"
            )

    def test_null_field_identity_with_row_col(self):
        """Passing row/col with _wind_field=None still gives scalar result."""
        raster = _make_raster()
        cma = _make_cma(raster, wind_speed=8.0, wind_dir_deg=180.0)

        moore = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for dr, dc in moore:
            got_with_rc = cma._phi_wind(dr, dc, row=3, col=4)
            got_no_rc   = cma._phi_wind(dr, dc)
            assert got_with_rc == got_no_rc, (
                f"dr={dr} dc={dc}: passing row/col with null field changes result"
            )


# ---------------------------------------------------------------------------
# Tests: wind_field changes _phi_wind output
# ---------------------------------------------------------------------------

class TestPhiWindWithField:
    def test_set_field_changes_phi_wind(self):
        """With a wind_field set, _phi_wind(dr,dc,row,col) reads per-cell values."""
        raster = _make_raster(nrows=8, ncols=8)
        cma = _make_cma(raster, wind_speed=10.0, wind_dir_deg=45.0)

        # Build a field: spread southward (dr=+1, dc=0), wind from north (dir=0 deg)
        # => toward = (0+180)%360 = 180 deg = southward => aligned with spread
        field = np.zeros((8, 8, 2), dtype=float)
        field[:, :, 0] = 5.0       # speed everywhere: 5 m/s
        field[:, :, 1] = 0.0       # from north (toward south)
        field[3, 3, 0] = 50.0      # extreme speed at cell (3,3), same direction

        cma.set_wind_field(field)

        dr, dc = 1, 0   # southward spread (aligned with wind toward=180 deg)
        # Normal cell (2,2): 5 m/s, toward south => some boost
        val_normal = cma._phi_wind(dr, dc, row=2, col=2)
        # Extreme cell (3,3): 50 m/s, same direction => much larger boost
        val_extreme = cma._phi_wind(dr, dc, row=3, col=3)
        assert val_extreme > val_normal, (
            f"Extreme speed at (3,3) ({val_extreme:.4f}) should be > normal 5 m/s ({val_normal:.4f})"
        )

    def test_set_field_direction_effect(self):
        """Wind direction from the field steers phi_wind differently than theta."""
        raster = _make_raster()
        cma = _make_cma(raster, wind_speed=10.0, wind_dir_deg=0.0)  # N wind

        # Field: south wind (from 180 deg) everywhere
        field = np.full((10, 12, 2), fill_value=0.0)
        field[:, :, 0] = 10.0    # speed
        field[:, :, 1] = 180.0  # from south (toward north)
        cma.set_wind_field(field)

        # Spreading north (dr=-1, dc=0): field wind FROM south blows TOWARD north
        # => strong alignment => high phi_wind
        phi_north = cma._phi_wind(-1, 0, row=5, col=5)
        # Spreading south (dr=+1, dc=0): against the wind
        phi_south = cma._phi_wind(1, 0, row=5, col=5)
        assert phi_north > phi_south, (
            "Wind from south should boost northward spread more than southward"
        )

    def test_set_field_then_none_restores_scalar(self):
        """After set_wind_field(None), _phi_wind reverts to scalar theta path."""
        raster = _make_raster()
        cma = _make_cma(raster, wind_speed=12.0, wind_dir_deg=90.0)

        field = np.zeros((10, 12, 2))
        field[:, :, 0] = 99.0
        field[:, :, 1] = 0.0
        cma.set_wind_field(field)
        assert cma._wind_field is not None

        cma.set_wind_field(None)
        assert cma._wind_field is None

        # Now _phi_wind should use scalar theta again
        got = cma._phi_wind(-1, 0, row=3, col=3)
        # Expected: scalar with wind_speed=12, wind_dir_deg=90
        toward = math.radians((90.0 + 180.0) % 360.0)
        sb = math.atan2(0, -(-1))
        cos_align = math.cos(toward - sb)
        expected = float(math.exp(0.25 * 12.0 * max(cos_align, -0.5)))
        assert got == expected, f"After set_wind_field(None): got {got} != expected {expected}"


# ---------------------------------------------------------------------------
# Tests: set_wind_field shape guard
# ---------------------------------------------------------------------------

class TestSetWindFieldShapeGuard:
    def test_wrong_shape_raises(self):
        """set_wind_field raises AssertionError when shape is wrong."""
        raster = _make_raster(nrows=10, ncols=12)
        cma = _make_cma(raster)

        bad_field = np.zeros((5, 5, 2))   # wrong nrows/ncols
        with pytest.raises(AssertionError):
            cma.set_wind_field(bad_field)

    def test_wrong_last_dim_raises(self):
        """set_wind_field raises AssertionError when last dim != 2."""
        raster = _make_raster(nrows=10, ncols=12)
        cma = _make_cma(raster)

        bad_field = np.zeros((10, 12, 3))  # last dim should be 2
        with pytest.raises(AssertionError):
            cma.set_wind_field(bad_field)

    def test_correct_shape_accepted(self):
        """set_wind_field accepts array of correct shape without error."""
        raster = _make_raster(nrows=10, ncols=12)
        cma = _make_cma(raster)

        good_field = np.zeros((10, 12, 2))
        good_field[:, :, 0] = 5.0
        good_field[:, :, 1] = 90.0
        cma.set_wind_field(good_field)   # should not raise
        assert cma._wind_field is not None

    def test_none_always_accepted(self):
        """set_wind_field(None) always succeeds."""
        raster = _make_raster()
        cma = _make_cma(raster)
        cma.set_wind_field(None)   # no exception
        assert cma._wind_field is None


# ---------------------------------------------------------------------------
# Tests: ros() uses per-cell wind when field is set
# ---------------------------------------------------------------------------

class TestRosWithWindField:
    def test_ros_uses_wind_field(self):
        """ros() should give different values with vs without a wind_field set."""
        raster = _make_raster()
        cma_scalar = _make_cma(raster, wind_speed=10.0, wind_dir_deg=45.0, seed=1)
        cma_field  = _make_cma(raster, wind_speed=10.0, wind_dir_deg=45.0, seed=1)

        # Field with a very different direction at cell (5,5)
        field = np.zeros((10, 12, 2))
        field[:, :, 0] = 10.0
        field[:, :, 1] = 225.0   # 180 deg opposite to 45 deg scalar wind
        cma_field.set_wind_field(field)

        dr, dc = -1, 0
        ros_scalar = cma_scalar.ros(5, 5, dr, dc)
        ros_field  = cma_field.ros(5, 5, dr, dc)
        # With opposite wind direction, ROS should differ from scalar
        assert ros_scalar != ros_field, (
            "ros() should give different values when wind_field overrides scalar theta"
        )
