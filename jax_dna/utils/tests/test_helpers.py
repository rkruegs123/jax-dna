import pytest

import jax_dna.utils.helpers as jdh


@pytest.mark.parametrize(
    ("in_iter", "n", "out_iter"),
    [
        (
            "ABCDEFG",
            3,
            [("A", "B", "C"), ("D", "E", "F"), ("G",)],
        ),
    ],
)
def test_batched(in_iter, n, out_iter):
    assert list(jdh.batched(in_iter, n)) == out_iter


def test_batched_raises_value_error():
    with pytest.raises(ValueError, match=jdh.ERR_BATCHED_N):
        list(jdh.batched("ABCDEFG", 0))
