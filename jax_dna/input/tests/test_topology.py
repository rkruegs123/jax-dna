from typing import Callable
import numpy as np
import pytest

import jax_dna.input.topology as jdt
import jax_dna.utils.types as typ


@pytest.mark.parametrize(
    (
        "n_nucleotides",
        "strand_counts",
        "bonded_neighbors",
        "unbonded_neighbors",
        "seq",
        "seq_one_hot",
        "expected_error",
    ),
    [
        # invalid number of nucleotides
        (
            0,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            "AAAAGGGGTTTTCCCC",
            np.concatenate(
                [
                    np.tile([[1, 0, 0, 0]], (4, 1)),
                    np.tile([[0, 1, 0, 0]], (4, 1)),
                    np.tile([[0, 0, 0, 1]], (4, 1)),
                    np.tile([[0, 0, 1, 0]], (4, 1)),
                ]
            ),
            jdt.ERR_TOPOLOGY_INVALID_NUMBER_NUCLEOTIDES,
        ),
        # number of nucleotides does not match sequence length
        (
            8,  # should be 16
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            "AAAAGGGGTTTTCCCC",
            np.concatenate(
                [
                    np.tile([[1, 0, 0, 0]], (4, 1)),
                    np.tile([[0, 1, 0, 0]], (4, 1)),
                    np.tile([[0, 0, 0, 1]], (4, 1)),
                    np.tile([[0, 0, 1, 0]], (4, 1)),
                ]
            ),
            jdt.ERR_TOPOLOGY_SEQ_NOT_MATCH_NUCLEOTIDES,
        ),
        # number of nucleotides does not match sum of strand counts
        (
            8,
            np.array([8, 8]),  # should be 8
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            "AAAAGGGG",
            np.concatenate(
                [
                    np.tile([[1, 0, 0, 0]], (4, 1)),
                    np.tile([[0, 1, 0, 0]], (4, 1)),
                    np.tile([[0, 0, 0, 1]], (4, 1)),
                    np.tile([[0, 0, 1, 0]], (4, 1)),
                ]
            ),
            jdt.ERR_TOPOLOGY_STRAND_COUNTS_NOT_MATCH,
        ),
        # strand counts is zeros
        (
            16,
            np.array([0, 0]),  # should be non-zero and 16
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            "AAAAGGGGTTTTCCCC",
            np.concatenate(
                [
                    np.tile([[1, 0, 0, 0]], (4, 1)),
                    np.tile([[0, 1, 0, 0]], (4, 1)),
                    np.tile([[0, 0, 0, 1]], (4, 1)),
                    np.tile([[0, 0, 1, 0]], (4, 1)),
                ]
            ),
            jdt.ERR_TOPOLOGY_INVALID_STRAND_COUNTS,
        ),
        # bonded neighbors is not 2d
        (
            16,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]).flatten(),  # not 2d
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            "AAAAGGGGTTTTCCCC",
            np.concatenate(
                [
                    np.tile([[1, 0, 0, 0]], (4, 1)),
                    np.tile([[0, 1, 0, 0]], (4, 1)),
                    np.tile([[0, 0, 0, 1]], (4, 1)),
                    np.tile([[0, 0, 1, 0]], (4, 1)),
                ]
            ),
            jdt.ERR_TOPOLOGY_BONDED_NEIGHBORS_INVALID_SHAPE,
        ),
        # bonded neighbors second dimension is not 2
        (
            16,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]).reshape(
                -1, 4
            ),  # not (n, 2)
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            "AAAAGGGGTTTTCCCC",
            np.concatenate(
                [
                    np.tile([[1, 0, 0, 0]], (4, 1)),
                    np.tile([[0, 1, 0, 0]], (4, 1)),
                    np.tile([[0, 0, 0, 1]], (4, 1)),
                    np.tile([[0, 0, 1, 0]], (4, 1)),
                ]
            ),
            jdt.ERR_TOPOLOGY_BONDED_NEIGHBORS_INVALID_SHAPE,
        ),
        # unbonded neighbors is not 2d
        (
            16,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]).flatten(),  # not 2d
            "AAAAGGGGTTTTCCCC",
            np.concatenate(
                [
                    np.tile([[1, 0, 0, 0]], (4, 1)),
                    np.tile([[0, 1, 0, 0]], (4, 1)),
                    np.tile([[0, 0, 0, 1]], (4, 1)),
                    np.tile([[0, 0, 1, 0]], (4, 1)),
                ]
            ),
            jdt.ERR_TOPOLOGY_UNBONDED_NEIGHBORS_INVALID_SHAPE,
        ),
        # unbonded neighbors second dimension is not 2
        (
            16,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]).reshape(
                -1, 4
            ),  # not (n, 2)
            "AAAAGGGGTTTTCCCC",
            np.concatenate(
                [
                    np.tile([[1, 0, 0, 0]], (4, 1)),
                    np.tile([[0, 1, 0, 0]], (4, 1)),
                    np.tile([[0, 0, 0, 1]], (4, 1)),
                    np.tile([[0, 0, 1, 0]], (4, 1)),
                ]
            ),
            jdt.ERR_TOPOLOGY_UNBONDED_NEIGHBORS_INVALID_SHAPE,
        ),
        # invalid sequence character
        (
            16,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            "AAAAGGGGDDDDCCCC",  # D is not allowed
            np.concatenate(
                [
                    np.tile([[1, 0, 0, 0]], (4, 1)),
                    np.tile([[0, 1, 0, 0]], (4, 1)),
                    np.tile([[0, 0, 0, 1]], (4, 1)),
                    np.tile([[0, 0, 1, 0]], (4, 1)),
                ]
            ),
            jdt.ERR_TOPOLOGY_INVALID_SEQUENCE_NUCLEOTIDES,
        ),
        # second dimension of one-hot sequence is not 4
        (
            16,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            "AAAAGGGGTTTTCCCC",
            np.concatenate(
                [
                    np.tile([[1, 0, 0, 0]], (4, 1)),
                    np.tile([[0, 1, 0, 0]], (4, 1)),
                    np.tile([[0, 0, 0, 1]], (4, 1)),
                    np.tile([[0, 0, 1, 0]], (4, 1)),
                ]
            ).reshape(-1, 2),  # not (n, 4)
            jdt.ERR_TOPOLOGY_INVALID_ONE_HOT_SEQUENCE_SHAPE,
        ),
        # number of one hot sequence values does not match number of nucleotides
        (
            16,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            "AAAAGGGGTTTTCCCC",
            np.concatenate(
                [
                    np.tile([[1, 0, 0, 0]], (3, 1)),
                    np.tile([[0, 1, 0, 0]], (4, 1)),
                    np.tile([[0, 0, 0, 1]], (4, 1)),
                    np.tile([[0, 0, 1, 0]], (4, 1)),
                ]
            ),
            jdt.ERR_TOPOLOGY_INVALID_ONE_HOT_SEQUENCE_SHAPE,
        ),
        # number of one hot sequence values does not match number of nucleotides
        (
            16,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            "AAAAGGGGTTTTCCCC",
            np.concatenate(
                [
                    np.tile([[2, 0, 0, 0]], (4, 1)),
                    np.tile([[0, 1, 0, 0]], (4, 1)),
                    np.tile([[0, 0, 0, 1]], (4, 1)),
                    np.tile([[0, 0, 1, 0]], (4, 1)),
                ]
            ),
            jdt.ERR_TOPOLOGY_INVALID_ONE_HOT_SEQUENCE_VALUES,
        ),
    ],
)
def test_topology_class_validation_raises_value_error(
    n_nucleotides: int,
    strand_counts: np.ndarray,
    bonded_neighbors: np.ndarray,
    unbonded_neighbors: np.ndarray,
    seq: str,
    seq_one_hot: np.ndarray,
    expected_error: str,
):
    with pytest.raises(ValueError, match=expected_error):
        jdt.Topology(
            n_nucleotides=n_nucleotides,
            strand_counts=strand_counts,
            bonded_neighbors=bonded_neighbors,
            unbonded_neighbors=unbonded_neighbors,
            seq=seq,
            seq_one_hot=seq_one_hot,
        )


@pytest.mark.parametrize(
    ("in_str", "expected_format", "expected_func"),
    [
        ("16 2", typ.OxdnaFormat.CLASSIC, jdt._from_file_oxdna_classic),
        ("16 2 5->3", typ.OxdnaFormat.NEW, jdt._from_file_oxdna_new),
    ],
)
def test_determine_oxdna_format(in_str: str, expected_format: typ.OxdnaFormat, expected_func: Callable):
    actual_format, actual_func = jdt._determine_oxdna_format(in_str)
    assert actual_format == expected_format
    assert actual_func == expected_func


@pytest.mark.parametrize(
    ("in_str"),
    [
        ("16"),
        ("16 2 5->3 4"),
    ],
)
def test_determine_oxdna_format_raises(in_str: str):
    with pytest.raises(ValueError, match=jdt.ERR_INVALID_OXDNA_FORMAT):
        jdt._determine_oxdna_format(in_str)
