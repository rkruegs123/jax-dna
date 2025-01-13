import dataclasses as dc
from pathlib import Path

import numpy as np
import pytest

import jax_dna.input.sequence_constraints as jsc
import jax_dna.utils.types as typ

TEST_FILES_DIR = Path(__file__).parent / "test_files"


@pytest.mark.parametrize(
    (
        "n_nucleotides",
        "n_unpaired",
        "n_bp",
        "is_unpaired",
        "unpaired",
        "bps",
        "idx_to_unpaired_idx",
        "idx_to_bp_idx",
        "expected_error",
    ),
    [
        # invalid number of nucleotides
        (
            0,
            2,
            2,
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([2, 4]),
            np.array([[0, 5], [2, 3]]),
            np.array([-1, -1, 0, 1, -1, -1]),
            np.array(
                [
                    [-1, -1],
                    [-1, -1],
                    [0, 0],
                    [0, 1],
                    [-1, -1],
                    [-1, -1],
                ]
            ),
            jsc.ERR_SEQ_CONSTRAINTS_INVALID_NUMBER_NUCLEOTIDES,
        ),
        # invalid unpaired shape
        (
            6,
            2,
            2,
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([2]),
            np.array([[0, 5], [2, 3]]),
            np.array([-1, -1, 0, 1, -1, -1]),
            np.array(
                [
                    [-1, -1],
                    [-1, -1],
                    [0, 0],
                    [0, 1],
                    [-1, -1],
                    [-1, -1],
                ]
            ),
            jsc.ERR_SEQ_CONSTRAINTS_INVALID_UNPAIRED_SHAPE,
        ),
        # invalid bp shape
        (
            6,
            2,
            2,
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([2, 4]),
            np.array(
                [
                    [0, 5],
                ]
            ),
            np.array([-1, -1, 0, 1, -1, -1]),
            np.array(
                [
                    [-1, -1],
                    [-1, -1],
                    [0, 0],
                    [0, 1],
                    [-1, -1],
                    [-1, -1],
                ]
            ),
            jsc.ERR_INVALID_BP_SHAPE,
        ),
        # invalid is_unpaired shape
        (
            6,
            2,
            2,
            np.array([0, 1, 0, 0, 1]),
            np.array([2, 4]),
            np.array([[0, 5], [2, 3]]),
            np.array([-1, -1, 0, 1, -1, -1]),
            np.array(
                [
                    [-1, -1],
                    [-1, -1],
                    [0, 0],
                    [0, 1],
                    [-1, -1],
                    [-1, -1],
                ]
            ),
            jsc.ERR_SEQ_CONSTRAINTS_INVALID_IS_UNPAIRED_SHAPE,
        ),
        # invalid unpaired mapper shape
        (
            6,
            2,
            2,
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([2, 4]),
            np.array([[0, 5], [2, 3]]),
            np.array([-1, -1, 0, 1, -1]),
            np.array(
                [
                    [-1, -1],
                    [-1, -1],
                    [0, 0],
                    [0, 1],
                    [-1, -1],
                    [-1, -1],
                ]
            ),
            jsc.ERR_SEQ_CONSTRAINTS_INVALID_UNPAIRED_MAPPER_SHAPE,
        ),
        # invalid bp mapper shape
        (
            6,
            2,
            2,
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([2, 4]),
            np.array([[0, 5], [2, 3]]),
            np.array([-1, -1, 0, 1, -1, -1]),
            np.array(
                [
                    [-1, -1],
                    [-1, -1],
                    [0, 0],
                    [0, 1],
                    [-1, -1],
                ]
            ),
            jsc.ERR_SEQ_CONSTRAINTS_INVALID_BP_MAPPER_SHAPE,
        ),
        # mismatched number of nucleotides, unpaired, and bps
        (
            6,
            2,
            1,
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([2, 4]),
            np.array([[2, 3]]),
            np.array([-1, -1, 0, 1, -1, -1]),
            np.array([[-1, -1], [-1, -1], [0, 0], [0, 1], [-1, -1], [-1, -1]]),
            jsc.ERR_SEQ_CONSTRAINTS_MISMATCH_NUM_TYPES,
        ),
        # unpaired and bp arrays dont cover range
        (
            6,
            2,
            2,
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([0, 4]),
            np.array([[0, 5], [2, 3]]),
            np.array([-1, -1, 0, 1, -1, -1]),
            np.array([[0, 0], [-1, -1], [1, 0], [1, 1], [-1, -1], [0, 1]]),
            jsc.ERR_SEQ_CONSTRAINTS_INVALID_COVER,
        ),
        # is_unpaired contains invalid values (i.e. is not one-hot)
        (
            6,
            2,
            2,
            np.array([0, 1, 0, 0, 1, 2]),
            np.array([1, 4]),
            np.array([[0, 5], [2, 3]]),
            np.array([-1, -1, 0, 1, -1, -1]),
            np.array([[0, 0], [-1, -1], [1, 0], [1, 1], [-1, -1], [0, 1]]),
            jsc.ERR_SEQ_CONSTRAINTS_IS_UNPAIRED_INVALID_VALUES,
        ),
        # is_unpaired disagrees with unpaired
        (
            6,
            2,
            2,
            np.array([0, 1, 0, 0, 0, 1]),
            np.array([1, 4]),
            np.array([[0, 5], [2, 3]]),
            np.array([-1, -1, 0, 1, -1, -1]),
            np.array([[0, 0], [-1, -1], [1, 0], [1, 1], [-1, -1], [0, 1]]),
            jsc.ERR_SEQ_CONSTRAINTS_INVALID_IS_UNPAIRED,
        ),
        # paired nucleotide mapped to valid unpaired index
        (
            6,
            2,
            2,
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([1, 4]),
            np.array([[0, 5], [2, 3]]),
            np.array([-1, -1, 0, 1, -1, 0]),
            np.array([[0, 0], [-1, -1], [1, 0], [1, 1], [-1, -1], [0, 1]]),
            jsc.ERR_SEQ_CONSTRAINTS_PAIRED_NT_MAPPED_TO_UNPAIRED,
        ),
        # incomplete unpaired index mapper
        (
            6,
            2,
            2,
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([1, 4]),
            np.array([[0, 5], [2, 3]]),
            np.array([-1, 0, -1, -1, -1, -1]),
            np.array([[0, 0], [-1, -1], [1, 0], [1, 1], [-1, -1], [0, 1]]),
            jsc.ERR_SEQ_CONSTRAINTS_INCOMPLETE_UNPAIRED_MAPPED_IDXS,
        ),
        # unpaired nucleotide mapped to a base paired nucleotide
        (
            6,
            2,
            2,
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([1, 4]),
            np.array([[0, 5], [2, 3]]),
            np.array([-1, 0, -1, -1, 1, -1]),
            np.array([[0, 0], [-1, -1], [1, 0], [1, 1], [0, -1], [0, 1]]),
            jsc.ERR_SEQ_CONSTRAINTS_UNPAIRED_NT_MAPPED_TO_PAIRED,
        ),
        # incomplete bp idx mapper
        (
            6,
            2,
            2,
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([1, 4]),
            np.array([[0, 5], [2, 3]]),
            np.array([-1, 0, -1, -1, 1, -1]),
            np.array([[0, 0], [-1, -1], [1, 0], [1, 1], [-1, -1], [0, -1]]),
            jsc.ERR_SEQ_CONSTRAINTS_INCOMPLETE_BP_MAPPED_IDXS,
        ),
    ],
)
def test_seq_constraints_class_validation_raises_value_error(
    n_nucleotides: int,
    n_unpaired: int,
    n_bp: int,
    is_unpaired: typ.Arr_Nucleotide_Int,
    unpaired: typ.Arr_Unpaired,
    bps: typ.Arr_Bp,
    idx_to_unpaired_idx: typ.Arr_Nucleotide_Int,
    idx_to_bp_idx: typ.Arr_Nucleotide_2_Int,
    expected_error: str,
):
    with pytest.raises(ValueError, match=expected_error):
        jsc.SequenceConstraints(
            n_nucleotides=n_nucleotides,
            n_unpaired=n_unpaired,
            n_bp=n_bp,
            is_unpaired=is_unpaired,
            unpaired=unpaired,
            bps=bps,
            idx_to_unpaired_idx=idx_to_unpaired_idx,
            idx_to_bp_idx=idx_to_bp_idx,
        )


@pytest.mark.parametrize(
    ("n_nucleotides", "bps", "expected_err"),
    [
        (4, np.array([[0, 3], [0, 1]]), jsc.ERR_BP_ARR_CONTAINS_DUPLICATES),
        (
            4,
            np.array(
                [
                    [0, 5],
                ]
            ),
            jsc.ERR_INVALID_BP_INDICES,
        ),
    ],
)
def test_from_bps_raises_value_error(n_nucleotides: int, bps: typ.Arr_Bp, expected_err: str):
    with pytest.raises(ValueError, match=expected_err):
        jsc.from_bps(n_nucleotides, bps)


@pytest.mark.parametrize(
    ("n_nucleotides", "bps", "expected"),
    [
        (
            4,
            np.array([[0, 3]]),
            jsc.SequenceConstraints(
                n_nucleotides=4,
                n_unpaired=2,
                n_bp=1,
                is_unpaired=np.array([0, 1, 1, 0]),
                unpaired=np.array([1, 2]),
                bps=np.array([[0, 3]]),
                idx_to_unpaired_idx=np.array([-1, 0, 1, -1]),
                idx_to_bp_idx=np.array([[0, 0], [-1, -1], [-1, -1], [0, 1]]),
            ),
        ),
    ],
)
def test_from_bps(n_nucleotides: int, bps: typ.Arr_Bp, expected: jsc.SequenceConstraints):
    actual = dc.asdict(jsc.from_bps(n_nucleotides, bps))
    expected = dc.asdict(expected)

    for key in expected:
        if key in ["is_unpaired", "unpaired", "bps", "idx_to_unpaired_idx", "idx_to_bp_idx"]:
            np.testing.assert_allclose(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]


@pytest.mark.parametrize(
    ("dseq", "sc", "expected"),
    [
        (
            np.array([0, 1, 2, 3]),
            jsc.SequenceConstraints(
                n_nucleotides=4,
                n_unpaired=2,
                n_bp=1,
                is_unpaired=np.array([0, 1, 1, 0]),
                unpaired=np.array([1, 2]),
                bps=np.array([[0, 3]]),
                idx_to_unpaired_idx=np.array([-1, 0, 1, -1]),
                idx_to_bp_idx=np.array([[0, 0], [-1, -1], [-1, -1], [0, 1]]),
            ),
            (np.array([[0, 1, 0, 0], [0, 0, 1, 0]]), np.array([[1, 0, 0, 0]])),
        ),
    ],
)
def test_dseq_to_pseq(dseq: typ.Discrete_Sequence, sc: jsc.SequenceConstraints, expected: typ.Probabilistic_Sequence):
    actual = jsc.dseq_to_pseq(dseq, sc)
    actual_up_pseq, actual_bp_pseq = actual

    expected_up_pseq, expected_bp_pseq = expected

    np.testing.assert_allclose(actual_up_pseq, expected_up_pseq)
    np.testing.assert_allclose(actual_bp_pseq, expected_bp_pseq)


@pytest.mark.parametrize(
    ("dseq", "sc", "expected_err"),
    [
        (
            np.array([0, 1, 2, 0]),
            jsc.SequenceConstraints(
                n_nucleotides=4,
                n_unpaired=2,
                n_bp=1,
                is_unpaired=np.array([0, 1, 1, 0]),
                unpaired=np.array([1, 2]),
                bps=np.array([[0, 3]]),
                idx_to_unpaired_idx=np.array([-1, 0, 1, -1]),
                idx_to_bp_idx=np.array([[0, 0], [-1, -1], [-1, -1], [0, 1]]),
            ),
            jsc.ERR_DSEQ_TO_PSEQ_INVALID_BP,
        ),
    ],
)
def test_dseq_to_pseq_raises_value_error(dseq: typ.Discrete_Sequence, sc: jsc.SequenceConstraints, expected_err: str):
    with pytest.raises(ValueError, match=expected_err):
        jsc.dseq_to_pseq(dseq, sc)
