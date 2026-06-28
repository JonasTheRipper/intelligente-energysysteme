"""Unit tests for deterministic mutation arbitration (spaces.arbitrate_mutations).

v0.3 replaces the v0.2 last-writer-wins ``_apply_mutations`` with a fixed-priority
resolution so two agents (wildfire + firefighter) writing the SAME
``gis.cell_mutations`` actuator in one env step produce a turn-order-independent
result: BURNED_OUT (terminal) > SUPPRESSED > FLOODED > BURNING > UNBURNED. These
are numpy-only (no palaestrai).
"""

import itertools

import numpy as np

from palaestrai_socal import spaces
from palaestrai_socal.spaces import (
    UNBURNED, BURNING, BURNED_OUT, SUPPRESSED, arbitrate_mutations,
)


def _blank(n=5):
    return np.full((n, n), UNBURNED, dtype=np.int8)


def test_suppressed_beats_burning_same_cell():
    S = _blank()
    out = arbitrate_mutations(S, [(2, 2, BURNING, spaces.LAYER_FIRE),
                                  (2, 2, SUPPRESSED, spaces.LAYER_SUPPRESSION)])
    assert out[2, 2] == SUPPRESSED


def test_order_independent():
    S = _blank()
    muts = [(1, 1, BURNING, spaces.LAYER_FIRE),
            (1, 1, SUPPRESSED, spaces.LAYER_SUPPRESSION),
            (2, 3, BURNING, spaces.LAYER_FIRE),
            (0, 4, SUPPRESSED, spaces.LAYER_SUPPRESSION)]
    base = arbitrate_mutations(S, muts)
    # every permutation of the mutation list yields the identical grid.
    for perm in itertools.permutations(muts):
        assert np.array_equal(arbitrate_mutations(S, list(perm)), base)


def test_burned_out_is_terminal():
    S = _blank()
    S[2, 2] = BURNED_OUT
    out = arbitrate_mutations(S, [(2, 2, SUPPRESSED, spaces.LAYER_SUPPRESSION),
                                  (2, 2, BURNING, spaces.LAYER_FIRE)])
    assert out[2, 2] == BURNED_OUT            # never overwritten


def test_burned_out_wins_among_proposals():
    S = _blank()
    out = arbitrate_mutations(S, [(0, 0, SUPPRESSED, spaces.LAYER_SUPPRESSION),
                                  (0, 0, BURNED_OUT, spaces.LAYER_FIRE)])
    assert out[0, 0] == BURNED_OUT


def test_out_of_bounds_dropped():
    S = _blank()
    out = arbitrate_mutations(S, [(-1, 0, BURNING, spaces.LAYER_FIRE),
                                  (99, 99, BURNING, spaces.LAYER_FIRE)])
    assert np.array_equal(out, S)


def test_invalid_state_dropped():
    S = _blank()
    out = arbitrate_mutations(S, [(1, 1, 99, spaces.LAYER_FIRE)])
    assert out[1, 1] == UNBURNED


def test_empty_mutations_is_noop_copy():
    S = _blank()
    S[1, 1] = BURNING
    out = arbitrate_mutations(S, [])
    assert np.array_equal(out, S)
    assert out is not S                        # returns a fresh array


def test_does_not_mutate_input():
    S = _blank()
    _ = arbitrate_mutations(S, [(2, 2, BURNING, spaces.LAYER_FIRE)])
    assert S[2, 2] == UNBURNED                 # original untouched
