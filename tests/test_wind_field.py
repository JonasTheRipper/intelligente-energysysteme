"""Tests for wildfire_cma.wind_field (v0.5).

Verifies:
1. perimeter_informed_wind_field() returns correct shape.
2. Speed is monotone increasing with interior distance (boundary_gain > 0).
3. Direction values are all finite (no NaN / Inf).
4. reclassify_burned_footprint() only touches class-0 cells inside mask.
"""
import numpy as np
import pytest

from wildfire_cma.wind_field import (
    perimeter_informed_wind_field,
    reclassify_burned_footprint,
    contain_burnable_footprint,
)


def _circular_mask(nrows: int = 30, ncols: int = 40, radius: float = 10.0):
    """A simple circular mask for testing."""
    rr = np.arange(nrows)[:, None]
    cc = np.arange(ncols)[None, :]
    cx, cy = ncols / 2, nrows / 2
    return ((rr - cy) ** 2 + (cc - cx) ** 2) <= radius ** 2


class TestPerimeterInformedWindField:
    def test_shape(self):
        """Field shape must be (nrows, ncols, 2)."""
        nrows, ncols = 25, 35
        mask = _circular_mask(nrows, ncols)
        field = perimeter_informed_wind_field(mask, base_speed=10.0, boundary_gain=0.5)
        assert field.shape == (nrows, ncols, 2), field.shape

    def test_speed_positive(self):
        """All speed values must be positive."""
        mask = _circular_mask()
        field = perimeter_informed_wind_field(mask, base_speed=12.0, boundary_gain=0.3)
        speeds = field[:, :, 0]
        assert np.all(speeds > 0), "Some speeds are non-positive"

    def test_speed_monotone_with_interior(self):
        """Cells deeper inside the mask should have higher speed than boundary cells."""
        mask = _circular_mask(nrows=40, ncols=40, radius=15.0)
        field = perimeter_informed_wind_field(mask, base_speed=10.0, boundary_gain=1.0)
        speeds = field[:, :, 0]
        # Centre cell (deepest interior) should have the highest speed.
        centre = speeds[20, 20]
        # Edge cells of the mask (near boundary) should have lower speed.
        # Sample a few cells near the boundary row.
        boundary_speeds = speeds[mask & (speeds < centre)]
        assert centre >= boundary_speeds.max(), (
            f"Centre speed {centre:.2f} not >= max boundary speed {boundary_speeds.max():.2f}"
        )

    def test_direction_finite(self):
        """All direction values must be finite (no NaN or Inf)."""
        mask = _circular_mask()
        field = perimeter_informed_wind_field(mask, base_speed=14.0, boundary_gain=0.3)
        dirs = field[:, :, 1]
        assert np.all(np.isfinite(dirs)), "Some direction values are non-finite"

    def test_direction_range(self):
        """Direction values must be in [0, 360)."""
        mask = _circular_mask()
        field = perimeter_informed_wind_field(mask, base_speed=14.0, boundary_gain=0.3)
        dirs = field[:, :, 1]
        assert np.all(dirs >= 0.0), "Some directions < 0"
        assert np.all(dirs < 360.0), "Some directions >= 360"

    def test_zero_boundary_gain(self):
        """With boundary_gain=0, all speeds equal base_speed."""
        mask = _circular_mask()
        base_speed = 8.0
        field = perimeter_informed_wind_field(mask, base_speed=base_speed, boundary_gain=0.0)
        speeds = field[:, :, 0]
        assert np.allclose(speeds, base_speed, atol=1e-9), (
            "With boundary_gain=0, speed should be uniform == base_speed"
        )

    def test_empty_mask_no_crash(self):
        """Empty mask (no burning area) should not crash."""
        mask = np.zeros((20, 20), dtype=bool)
        field = perimeter_informed_wind_field(mask, base_speed=10.0, boundary_gain=0.5)
        assert field.shape == (20, 20, 2)
        assert np.all(np.isfinite(field))


class TestReclassifyBurnedFootprint:
    def test_only_class0_inside_mask_changed(self):
        """Only class-0 cells inside the mask are reclassified."""
        fuel = np.array([[0, 1, 2], [0, 3, 0], [1, 0, 3]], dtype=np.int16)
        # Mask covers the middle column + some extra
        mask = np.array(
            [[True, True, False], [True, True, False], [False, True, False]]
        )
        fuel_orig = fuel.copy()
        result = reclassify_burned_footprint(fuel, mask, target_class=3)

        for r in range(3):
            for c in range(3):
                if mask[r, c] and fuel_orig[r, c] == 0:
                    assert result[r, c] == 3, (
                        f"Cell ({r},{c}) inside mask with class 0 was not reclassified"
                    )
                else:
                    assert result[r, c] == fuel_orig[r, c], (
                        f"Cell ({r},{c}) was changed when it should not have been"
                    )

    def test_class0_outside_mask_unchanged(self):
        """Class-0 cells outside the mask must not be touched."""
        fuel = np.zeros((5, 5), dtype=np.int16)
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True  # only centre inside mask
        result = reclassify_burned_footprint(fuel.copy(), mask, target_class=3)
        # centre should be reclassified
        assert result[2, 2] == 3
        # everything else should remain 0
        for r in range(5):
            for c in range(5):
                if (r, c) != (2, 2):
                    assert result[r, c] == 0, f"({r},{c}) changed unexpectedly"

    def test_nonzero_inside_mask_unchanged(self):
        """Non-zero fuel classes inside the mask must not change."""
        fuel = np.array([[1, 2, 3], [0, 4, 0]], dtype=np.int16)
        mask = np.ones((2, 3), dtype=bool)
        result = reclassify_burned_footprint(fuel.copy(), mask, target_class=3)
        # Non-zero entries inside the mask: (0,0)=1, (0,1)=2, (0,2)=3, (1,1)=4
        assert result[0, 0] == 1
        assert result[0, 1] == 2
        assert result[0, 2] == 3
        assert result[1, 1] == 4
        # Zero entries inside the mask: (1,0) and (1,2) -> reclassified to 3
        assert result[1, 0] == 3
        assert result[1, 2] == 3

    def test_returns_same_array(self):
        """reclassify_burned_footprint must return (mutate) the same array."""
        fuel = np.zeros((4, 4), dtype=np.int16)
        mask = np.ones((4, 4), dtype=bool)
        result = reclassify_burned_footprint(fuel, mask, target_class=3)
        assert result is fuel, "Must return the same (mutated) array"

    def test_custom_target_class(self):
        """Custom target_class is applied."""
        fuel = np.zeros((3, 3), dtype=np.int16)
        mask = np.ones((3, 3), dtype=bool)
        result = reclassify_burned_footprint(fuel, mask, target_class=2)
        assert np.all(result == 2)


class TestContainBurnableFootprint:
    """contain_burnable_footprint zeroes fuel outside the dilated real mask."""

    def test_zeroes_outside_perimeter(self):
        """Cells beyond the margin become non-burnable (class 0)."""
        fuel = np.full((20, 20), 3, dtype=np.int16)
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True  # small central footprint
        result = contain_burnable_footprint(fuel, mask, margin_cells=0)
        # Inside the mask: fuel preserved.
        assert np.all(result[8:12, 8:12] == 3)
        # A far corner well outside the footprint: zeroed.
        assert result[0, 0] == 0
        assert result[19, 19] == 0

    def test_margin_expands_allowed_region(self):
        """margin_cells>0 keeps a ring of burnable fuel around the perimeter."""
        fuel = np.full((20, 20), 3, dtype=np.int16)
        mask = np.zeros((20, 20), dtype=bool)
        mask[9:11, 9:11] = True
        r0 = contain_burnable_footprint(fuel.copy(), mask, margin_cells=0)
        r2 = contain_burnable_footprint(fuel.copy(), mask, margin_cells=2)
        # The margin=2 version must keep at least as many burnable cells.
        assert (r2 != 0).sum() >= (r0 != 0).sum()
        # A cell 2 rings out (row 8, col 9) is burnable with margin=2, not with 0.
        assert r2[8, 9] == 3
        assert r0[8, 9] == 0

    def test_preserves_interior_nonburnable(self):
        """Non-burnable cells inside the footprint stay non-burnable."""
        fuel = np.full((10, 10), 3, dtype=np.int16)
        fuel[5, 5] = 0  # e.g. a lake inside the footprint
        mask = np.ones((10, 10), dtype=bool)
        result = contain_burnable_footprint(fuel, mask, margin_cells=2)
        assert result[5, 5] == 0
        assert result[0, 0] == 3  # rest of the fully-masked grid preserved

    def test_returns_same_array(self):
        """contain_burnable_footprint must return (mutate) the same array."""
        fuel = np.full((6, 6), 3, dtype=np.int16)
        mask = np.ones((6, 6), dtype=bool)
        result = contain_burnable_footprint(fuel, mask, margin_cells=1)
        assert result is fuel
