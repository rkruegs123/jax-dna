import dataclasses as dc
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jax_dna.input.topology as jdt
import jax_dna.utils.constants as jd_const
import jax_dna.utils.types as typ

TEST_FILES_DIR = Path(__file__).parent / "test_files"


@pytest.mark.parametrize(
    (
        "n_nucleotides",
        "strand_counts",
        "bonded_neighbors",
        "unbonded_neighbors",
        "seq",
        "is_end",
        "nt_type",
        "expected_error",
    ),
    [
        # invalid number of nucleotides
        (
            0,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=jnp.int32),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 16),
            jdt.ERR_TOPOLOGY_INVALID_NUMBER_NUCLEOTIDES,
        ),
        # number of nucleotides does not match sequence length
        (
            16,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=jnp.int32),  # only has 15 nucleotides
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 16),
            jdt.ERR_TOPOLOGY_INVALID_DISCRETE_SEQUENCE_SHAPE,
        ),
        # number of nucleotides does not match sum of strand counts
        (
            8,
            np.array([8, 8]),  # should be 8
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            jnp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 16),
            jdt.ERR_TOPOLOGY_STRAND_COUNTS_NOT_MATCH,
        ),
        # strand counts is zeros
        (
            16,
            np.array([0, 0]),  # should be non-zero and 16
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=jnp.int32),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 16),
            jdt.ERR_TOPOLOGY_INVALID_STRAND_COUNTS,
        ),
        # bonded neighbors is not 2d
        (
            16,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]).flatten(),  # not 2d
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=jnp.int32),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 16),
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
            jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=jnp.int32),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 16),
            jdt.ERR_TOPOLOGY_BONDED_NEIGHBORS_INVALID_SHAPE,
        ),
        # unbonded neighbors is not 2d
        (
            16,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]).flatten(),  # not 2d
            jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=jnp.int32),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 16),
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
            jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=jnp.int32),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 16),
            jdt.ERR_TOPOLOGY_UNBONDED_NEIGHBORS_INVALID_SHAPE,
        ),
        # invalid sequence character
        (
            16,
            np.array([8, 8]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 9], [10, 11], [12, 13], [14, 15]]),
            jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4], dtype=jnp.int32),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 16),
            jdt.ERR_TOPOLOGY_INVALID_SEQUENCE_NUCLEOTIDES,
        ),
        # invalid unpaired probabilistic sequence shape (1)
        (
            8,
            np.array([4, 4]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            (
                np.array(
                    [  # unpaired pseq is (n_unpaired, 5)
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                    ]
                ),
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                    ]
                ),
            ),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 8),
            jdt.ERR_TOPOLOGY_INVALID_UNPAIRED_PSEQ_SHAPE,
        ),
        # invalid unpaired probabilistic sequence shape (2)
        (
            8,
            np.array([4, 4]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            (
                np.array([0, 0, 0]),  # unpaired pseq has only one dimension
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                    ]
                ),
            ),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 8),
            jdt.ERR_TOPOLOGY_INVALID_UNPAIRED_PSEQ_SHAPE,
        ),
        # invalid bp probabilistic sequence shape (1)
        (
            8,
            np.array([4, 4]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            (
                np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
                np.array(
                    [  # bp pseq is (n_bp, 5)
                        [1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                    ]
                ),
            ),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 8),
            jdt.ERR_TOPOLOGY_INVALID_BP_PSEQ_SHAPE,
        ),
        # invalid bp probabilistic sequence shape (2)
        (
            8,
            np.array([4, 4]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            (
                np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
                np.array([0, 0, 0]),  # bp pseq is one-dimensional
            ),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 8),
            jdt.ERR_TOPOLOGY_INVALID_BP_PSEQ_SHAPE,
        ),
        # invalid number of implicit nucleotides
        (
            8,
            np.array([4, 4]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            (
                np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],  # extra nucleotide
                    ]
                ),
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                    ]
                ),
            ),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 8),
            jdt.ERR_TOPOLOGY_MISMATCH_PSEQ_SHAPE_NUM_NUCLEOTIDES,
        ),
        # invalid probabilities
        (
            8,
            np.array([4, 4]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            (
                np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                    ]
                ),
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, -1],  # Invalid probability
                    ]
                ),
            ),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 8),
            jdt.ERR_TOPOLOGY_INVALID_PROBABILITIES,
        ),
        # probabilities not normalized
        (
            8,
            np.array([4, 4]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            np.array([[1, 2], [3, 4], [5, 6], [0, 7]]),
            (
                np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                    ]
                ),
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 1],  # Invalid probability
                    ]
                ),
            ),
            jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
            jnp.array([jdt.NucleotideType.UNSPECIFIED] * 8),
            jdt.ERR_TOPOLOGY_PSEQ_NOT_NORMALIZED,
        ),
    ],
)
def test_topology_class_validation_raises_value_error(
    n_nucleotides: int,
    strand_counts: np.ndarray,
    bonded_neighbors: np.ndarray,
    unbonded_neighbors: np.ndarray,
    seq: typ.Sequence,
    is_end: np.ndarray,
    nt_type: np.ndarray,
    expected_error: str,
):
    with pytest.raises(ValueError, match=expected_error):
        jdt.Topology(
            n_nucleotides=n_nucleotides,
            strand_counts=strand_counts,
            bonded_neighbors=bonded_neighbors,
            unbonded_neighbors=unbonded_neighbors,
            seq=seq,
            is_end=is_end,
            nt_type=nt_type,
        )


@pytest.mark.parametrize(
    ("in_str", "expected_format", "expected_func"),
    [
        ("16 2", typ.oxDNAFormat.CLASSIC, jdt._from_file_oxdna_classic),
        ("16 2 5->3", typ.oxDNAFormat.NEW, jdt._from_file_oxdna_new),
    ],
)
def test_determine_oxdna_format(in_str: str, expected_format: typ.oxDNAFormat, expected_func: Callable):
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


def test_get_bonded_neighbors_raises_value_error():
    with pytest.raises(ValueError, match=jdt.ERR_STRAND_COUNTS_CIRCULAR_MISMATCH):
        jdt._get_bonded_neighbors([8, 8], [True])


@pytest.mark.parametrize(
    ("strand_lengths", "is_circular", "expected"),
    [
        # two circular strands
        ([3, 3], [True, True], [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]),
        # two linear strands
        ([3, 3], [False, False], [(0, 1), (1, 2), (3, 4), (4, 5)]),
    ],
)
def test_get_bonded_neighbors(strand_lengths: list[int], is_circular: list[bool], expected: list[tuple[int, int]]):
    assert set(jdt._get_bonded_neighbors(strand_lengths, is_circular)) == set(expected)


@pytest.mark.parametrize(
    ("n_nucleotides", "bonded_neighbors", "expected"),
    [
        (4, [(0, 1), (2, 3)], [(0, 2), (0, 3), (1, 2), (1, 3)]),
    ],
)
def test_get_unbonded_neighbors(
    n_nucleotides: int, bonded_neighbors: list[tuple[int, int]], expected: list[tuple[int, int]]
):
    assert set(jdt._get_unbonded_neighbors(n_nucleotides, bonded_neighbors)) == set(expected)


def test_from_oxdna_file_raises_file_not_found_error():
    with pytest.raises(FileNotFoundError, match=jdt.ERR_FILE_NOT_FOUND):
        jdt.from_oxdna_file("does-not-exist.top")


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        (
            TEST_FILES_DIR / "simple-helix-topology-circular-oxdna-classic.top",
            jdt.Topology(
                n_nucleotides=6,
                strand_counts=np.array([3, 3]),
                bonded_neighbors=np.array([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]),
                unbonded_neighbors=np.array(
                    [
                        (0, 3),
                        (0, 4),
                        (0, 5),
                        (1, 3),
                        (1, 4),
                        (1, 5),
                        (2, 3),
                        (2, 4),
                        (2, 5),
                    ]
                ),
                seq=jnp.array([jd_const.NUCLEOTIDES_IDX[s] for s in "GCATGC"], dtype=jnp.int32),
                is_end=jnp.array([0, 0, 0, 0, 0, 0]),
                nt_type=jnp.array([jdt.NucleotideType.UNSPECIFIED] * 3 + [jdt.NucleotideType.DNA] * 3),
            ),
        ),
        (
            TEST_FILES_DIR / "simple-helix-topology-circular-oxdna-new.top",
            jdt.Topology(
                n_nucleotides=6,
                strand_counts=np.array([3, 3]),
                bonded_neighbors=np.array([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]),
                unbonded_neighbors=np.array(
                    [
                        (0, 3),
                        (0, 4),
                        (0, 5),
                        (1, 3),
                        (1, 4),
                        (1, 5),
                        (2, 3),
                        (2, 4),
                        (2, 5),
                    ]
                ),
                seq=jnp.array([jd_const.NUCLEOTIDES_IDX[s] for s in "GCATGC"], dtype=jnp.int32),
                is_end=jnp.array([0, 0, 0, 0, 0, 0]),
                nt_type=jnp.array([jdt.NucleotideType.UNSPECIFIED] * 6),
            ),
        ),
        (
            TEST_FILES_DIR / "simple-helix-topology-linear-oxdna-classic.top",
            jdt.Topology(
                n_nucleotides=6,
                strand_counts=np.array([3, 3]),
                bonded_neighbors=np.array([(0, 1), (1, 2), (3, 4), (4, 5)]),
                unbonded_neighbors=np.array(
                    [
                        (0, 2),
                        (0, 3),
                        (0, 4),
                        (0, 5),
                        (1, 3),
                        (1, 4),
                        (1, 5),
                        (2, 3),
                        (2, 4),
                        (2, 5),
                        (3, 5),
                    ]
                ),
                seq=jnp.array([jd_const.NUCLEOTIDES_IDX[s] for s in "GCATGC"], dtype=jnp.int32),
                is_end=jnp.array([1, 0, 1, 1, 0, 1]),
                nt_type=jnp.array([jdt.NucleotideType.UNSPECIFIED] * 3 + [jdt.NucleotideType.DNA] * 3),
            ),
        ),
        (
            TEST_FILES_DIR / "simple-helix-topology-linear-oxdna-new.top",
            jdt.Topology(
                n_nucleotides=6,
                strand_counts=np.array([3, 3]),
                bonded_neighbors=np.array([(0, 1), (1, 2), (3, 4), (4, 5)]),
                unbonded_neighbors=np.array(
                    [
                        (0, 2),
                        (0, 3),
                        (0, 4),
                        (0, 5),
                        (1, 3),
                        (1, 4),
                        (1, 5),
                        (2, 3),
                        (2, 4),
                        (2, 5),
                        (3, 5),
                    ]
                ),
                seq=jnp.array([jd_const.NUCLEOTIDES_IDX[s] for s in "GCATGC"], dtype=jnp.int32),
                is_end=jnp.array([1, 0, 1, 1, 0, 1]),
                nt_type=jnp.array([jdt.NucleotideType.UNSPECIFIED] * 6),
            ),
        ),
    ],
)
def test_from_oxdna_file(file_path: str, expected: jdt.Topology):
    actual = dc.asdict(jdt.from_oxdna_file(file_path))
    expected = dc.asdict(expected)

    for key in expected:
        if key in ["bonded_neighbors", "unbonded_neighbors"]:
            to_set = lambda x: {tuple(y) for y in x}
            assert to_set(actual[key].tolist()) == to_set(expected[key].tolist()), key
        elif key in ["strand_counts", "seq"]:
            np.testing.assert_allclose(actual[key], expected[key])
        elif key in ["is_end", "nt_type"]:
            assert (actual[key] == expected[key]).all()
        else:
            assert actual[key] == expected[key]
