import numpy as np
import pytest

import jax_dna.input.topology as jdt


@pytest.mark.parametrize(
    ("n_nucleotides", "strand_counts", "bonded_neighbors", "unbonded_neighbors", "seq", "seq_one_hot", "expected_error"),
    # invalid number of nucleotides
    [
        (
            0,
            [8, 8],
            [[1, 2, 3, 4, 5, 6, 7, 8], [0, 9, 10, 11, 12, 13, 14, 15]],
            [[], [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
            "ATCGATCGATCGATCG",
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] * 4,
        ),
    ],
)
def test_topology_class_validation_raises_value_error(
    n_nucleotides: int,
    strand_counts: list[int],
    bonded_neighbors: list[list[int]],
    unbonded_neighbors: list[list[int]],
    seq: str,
    seq_one_hot: list[list[int]],
    expected_error: str,
):
    with pytest.raises(ValueError):
        jdt.Topology(
            n_nucleotides=n_nucleotides,
            strand_counts=strand_counts,
            bonded_neighbors=bonded_neighbors,
            unbonded_neighbors=unbonded_neighbors,
            seq=seq,
            seq_one_hot=seq_one_hot,
        )
