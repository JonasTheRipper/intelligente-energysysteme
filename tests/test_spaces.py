"""Unit tests for palaestrai_socal.spaces (pure-numpy codecs)."""

import numpy as np

from palaestrai_socal import spaces


def test_mutation_round_trip_exact():
    muts = [(0, 0, spaces.BURNING, spaces.LAYER_FIRE),
            (5, 9, spaces.BURNED_OUT, spaces.LAYER_FIRE),
            (123, 456, spaces.SUPPRESSED, spaces.LAYER_SUPPRESSION)]
    vec = spaces.encode_mutations(muts)
    assert vec.shape == (spaces.mutation_vector_size(),)
    out = spaces.decode_mutations(vec)
    assert out == muts


def test_mutation_empty():
    vec = spaces.encode_mutations([])
    assert spaces.decode_mutations(vec) == []
    assert vec[0] == 0.0


def test_mutation_truncates_at_cap():
    cap = 16
    muts = [(i, i, spaces.BURNING, spaces.LAYER_FIRE) for i in range(cap + 50)]
    vec = spaces.encode_mutations(muts, cap=cap)
    out = spaces.decode_mutations(vec, cap=cap)
    assert len(out) == cap
    assert out[0] == (0, 0, spaces.BURNING, spaces.LAYER_FIRE)
    assert out[-1] == (cap - 1, cap - 1, spaces.BURNING, spaces.LAYER_FIRE)


def test_decode_clamps_bad_count():
    vec = spaces.encode_mutations([(1, 2, 1, 0)], cap=4)
    vec[0] = 9999  # corrupt the count header
    out = spaces.decode_mutations(vec, cap=4)
    assert len(out) <= 4


def test_grid_codec_round_trip():
    rng = np.random.default_rng(0)
    grid = rng.integers(0, 5, size=(40, 53)).astype(np.int8)
    obj = spaces.encode_grid(grid)
    # JSON-serialisable: only list/str/int leaves
    assert isinstance(obj["data"], str)
    assert obj["shape"] == [40, 53]
    back = spaces.decode_grid(obj)
    assert back.shape == grid.shape
    assert np.array_equal(back, grid)
    assert back.dtype == np.int8


def test_grid_codec_json_safe():
    import json
    grid = np.zeros((8, 8), dtype=np.int8)
    grid[2, 3] = spaces.BURNING
    s = json.dumps(spaces.encode_grid(grid))
    obj = json.loads(s)
    assert np.array_equal(spaces.decode_grid(obj), grid)
