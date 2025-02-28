import re
from pathlib import Path

import numpy as np
import pytest

import jax_dna.input.trajectory as jdt

TEST_FILES_DIR = Path(__file__).parent / "test_files"


@pytest.mark.parametrize(
    (
        "n_nucleotides",
        "strand_lengths",
        "times",
        "energies",
        "states",
        "expected_msg",
    ),
    [
        # fails because n_nucleotides != sum(strand_lengths)
        (
            10,  # ok
            [5, 6],  # Doesn't match above
            np.array([0.0, 1.0]),  # ok
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),  # ok
            [
                jdt.NucleotideState(array=np.zeros((10, 15))),
                jdt.NucleotideState(array=np.zeros((10, 15))),
            ],  # ok
            re.escape(jdt.ERR_TRAJECTORY_N_NUCLEOTIDE_STRAND_LEGNTHS),
        ),
        # fails because n_nucleotides != sum(strand_lengths)
        (
            11,  # Doesn't match below
            [5, 5],  # ok
            np.array([0.0, 1.0]),  # ok
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),  # ok
            [
                jdt.NucleotideState(array=np.zeros((10, 15))),
                jdt.NucleotideState(array=np.zeros((10, 15))),
            ],  # ok
            re.escape(jdt.ERR_TRAJECTORY_N_NUCLEOTIDE_STRAND_LEGNTHS),
        ),
        # fails because times, energies, and states do not have the same length
        (
            10,  # ok
            [5, 5],  # ok
            np.array([0.0, 1.0, 2.0]),  # too long, len=3
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),  # ok, len=2
            [
                jdt.NucleotideState(array=np.zeros((10, 15))),
                jdt.NucleotideState(array=np.zeros((10, 15))),
            ],  # ok, len=2
            re.escape(jdt.ERR_TRAJECTORY_T_E_S_LENGTHS),
        ),
        # fails because times, energies, and states do not have the same length
        (
            10,  # ok
            [5, 5],  # ok
            np.array([0.0, 1.0]),  # ok, len=2
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),  # ok, len=2
            [
                jdt.NucleotideState(array=np.zeros((10, 15))),
            ],  # not ok, len=1
            re.escape(jdt.ERR_TRAJECTORY_T_E_S_LENGTHS),
        ),
        # fails because times is not a 1D array
        (
            10,  # ok
            [5, 5],  # ok
            np.array([0.0, 1.0]).reshape([-1, 1]),  # not ok
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),  # ok
            [
                jdt.NucleotideState(array=np.zeros((10, 15))),
                jdt.NucleotideState(array=np.zeros((10, 15))),
            ],  # ok
            re.escape(jdt.ERR_TRAJECTORY_TIMES_DIMS),
        ),
        # fails because energies is not a 2D array
        (
            10,  # ok
            [5, 5],  # ok
            np.array([0.0, 1.0]),  # ok
            np.array([[0.0, 1.0]]).flatten(),  # not ok
            [
                jdt.NucleotideState(array=np.zeros((10, 15))),
                jdt.NucleotideState(array=np.zeros((10, 15))),
            ],  # ok
            re.escape(jdt.ERR_TRAJECTORY_ENERGIES_SHAPE),
        ),
        # fails because energies second dimension is not 3
        (
            10,  # ok
            [5, 5],  # ok
            np.array([0.0, 1.0, 0.0]),  # ok
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).reshape([3, 2]),  # not ok
            [
                jdt.NucleotideState(array=np.zeros((10, 15))),
                jdt.NucleotideState(array=np.zeros((10, 15))),
                jdt.NucleotideState(array=np.zeros((10, 15))),
            ],  # ok
            re.escape(jdt.ERR_TRAJECTORY_ENERGIES_SHAPE),
        ),
    ],
)
def test_trajectory_class_validation_raises_value_error(
    n_nucleotides: int,
    strand_lengths: list[int],
    times: np.ndarray,
    energies: np.ndarray,
    states: list[jdt.NucleotideState],
    expected_msg: str,
):
    with pytest.raises(ValueError, match=expected_msg):
        jdt.Trajectory(
            n_nucleotides=n_nucleotides,
            strand_lengths=strand_lengths,
            times=times,
            energies=energies,
            states=states,
        )


@pytest.mark.parametrize(
    (
        "n_nucleotides",
        "strand_lengths",
        "times",
        "energies",
        "states",
        "expected_msg",
    ),
    [
        # fails because times is not a numpy array
        (
            10,  # ok
            [5, 5],  # ok
            [0.0, 1.0],  # not the right type
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),  # ok
            [
                jdt.NucleotideState(array=np.zeros((10, 15))),
                jdt.NucleotideState(array=np.zeros((10, 15))),
            ],  # ok
            re.escape(jdt.ERR_TRAJECTORY_TIMES_TYPE),
        ),
        # fails because energies is not a numpy array
        (
            10,  # ok
            [5, 5],  # ok
            np.array([0.0, 1.0]),  # ok
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],  # not the right type
            [
                jdt.NucleotideState(array=np.zeros((10, 15))),
                jdt.NucleotideState(array=np.zeros((10, 15))),
            ],  # ok
            re.escape(jdt.ERR_TRAJECTORY_ENERGIES_TYPE),
        ),
    ],
)
def test_trajectory_class_validation_raises_type_error(
    n_nucleotides: int,
    strand_lengths: list[int],
    times: np.ndarray,
    energies: np.ndarray,
    states: list[jdt.NucleotideState],
    expected_msg: str,
):
    with pytest.raises(TypeError, match=expected_msg):
        jdt.Trajectory(
            n_nucleotides=n_nucleotides, strand_lengths=strand_lengths, times=times, energies=energies, states=states
        )


@pytest.mark.parametrize(
    ("n_nucleotides", "strand_lengths", "times", "energies", "states"),
    [
        (
            10,  # ok
            [5, 5],  # ok
            np.array([0.0, 1.0]),  # ok
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),  # ok
            [
                jdt.NucleotideState(array=np.zeros((10, 15))),
                jdt.NucleotideState(array=np.zeros((10, 15))),
            ],  # ok
        ),
    ],
)
def test_trajectory_class_validation(
    n_nucleotides: int,
    strand_lengths: list[int],
    times: np.ndarray,
    energies: np.ndarray,
    states: list[jdt.NucleotideState],
):
    jdt.Trajectory(
        n_nucleotides=n_nucleotides,
        strand_lengths=strand_lengths,
        times=times,
        energies=energies,
        states=states,
    )


def test_nucleotide_basic_properties():
    arr = np.arange(150).reshape((10, 15))
    ns = jdt.NucleotideState(array=arr)

    slices = [
        slice(0, 3),
        slice(3, 6),
        slice(6, 9),
        slice(9, 12),
        slice(12, 15),
    ]
    properties = [
        "com",
        "back_base_vector",
        "base_normal",
        "velocity",
        "angular_velocity",
    ]
    for slc, prop in zip(slices, properties, strict=True):
        np.testing.assert_equal(getattr(ns, prop), arr[:, slc])


@pytest.mark.parametrize(
    ("array", "expected_msg"),
    [
        (np.zeros((10, 15)).flatten(), "^[" + jdt.ERR_NUCLEOTIDE_STATE_SHAPE + "]"),  # not 2D
        (np.zeros((10, 14)), "^[" + jdt.ERR_NUCLEOTIDE_STATE_SHAPE + "]"),  # not enough columns
    ],
)
def test_nucleotide_state_class_validation_raises_value_error(array: np.ndarray, expected_msg: str):
    with pytest.raises(ValueError, match=expected_msg):
        jdt.NucleotideState(array=array)


@pytest.mark.parametrize(
    ("array", "expected_msg"),
    [
        (list(range(15)), "^[" + jdt.ERR_NUCLEOTIDE_STATE_TYPE + "]"),  # not a numpy array
    ],
)
def test_nucleotide_state_class_validation_raises_type_error(array: np.ndarray, expected_msg: str):
    with pytest.raises(TypeError, match=expected_msg):
        jdt.NucleotideState(array=array)


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((10, 15)),
    ],
)
def test_nucleotide_state_euler_angle_shape(array: np.ndarray):
    ns = jdt.NucleotideState(array=array)
    n_nucleotides = array.shape[0]
    expected_shape = (n_nucleotides,)
    assert ns.euler_angles[0].shape == expected_shape
    assert ns.euler_angles[1].shape == expected_shape
    assert ns.euler_angles[2].shape == expected_shape


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((10, 15)),
    ],
)
def test_nucleotide_state_quaternion_shape(array: np.ndarray):
    ns = jdt.NucleotideState(array=array)
    n_nucleotides = array.shape[0]
    expected_shape = (n_nucleotides, 4)
    assert ns.quaternions.shape == expected_shape


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((10, 15)),
    ],
)
def test_nucleotide_state_to_rigid_body(array: np.ndarray):
    ns = jdt.NucleotideState(array=array)
    n_nucleotides = array.shape[0]
    rb = ns.to_rigid_body()
    assert rb.center.shape == (n_nucleotides, 3)
    np.testing.assert_equal(rb.center, ns.com)
    assert rb.orientation.vec.shape == (n_nucleotides, 4)
    # need to cast to numpy here otherise numpy will raise an error
    np.testing.assert_equal(np.array(rb.orientation.vec), np.array(ns.quaternions))


@pytest.mark.parametrize(
    ("state_box_sizes"),
    [
        ([np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])],),  # ok
    ],
)
def test_validate_box_size(state_box_sizes: list[np.ndarray]):
    jdt.validate_box_size(state_box_sizes)


# similar test but now raises value error
@pytest.mark.parametrize(
    ("state_box_sizes", "expected_msg"),
    [
        (
            [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), np.array([2.0, 1.0, 1.0])],
            re.escape(jdt.ERR_FIXED_BOX_SIZE),
        ),  # too many boxes
    ],
)
def test_validate_box_size_raises_value_error(state_box_sizes: list[np.ndarray], expected_msg: str):
    with pytest.raises(ValueError, match=expected_msg):
        jdt.validate_box_size(state_box_sizes)


@pytest.mark.parametrize(
    (
        "datafile",
        "strand_lengths",
        "is_oxdna",
        "n_procs",
        "expected_ts",
        "expected_energies",
        "expected_state_col1_vals",
    ),
    [
        (
            TEST_FILES_DIR / "simple-helix-8bp-5steps.conf",
            [8, 8],
            True,
            1,
            np.arange(5),
            np.tile(np.arange(5), (3, 1)).T,
            np.concatenate([np.arange(20, 28)[::-1], np.arange(28, 36)[::-1]]),
        ),
        (
            TEST_FILES_DIR / "simple-helix-8bp-5steps.conf",
            [8, 8],
            False,
            1,
            np.arange(5),
            np.tile(np.arange(5), (3, 1)).T,
            np.concatenate([np.arange(20, 28), np.arange(28, 36)]),
        ),
        (
            TEST_FILES_DIR / "simple-helix-8bp-5steps.conf",
            [8, 8],
            True,
            2,
            np.arange(5),
            np.tile(np.arange(5), (3, 1)).T,
            np.concatenate([np.arange(20, 28)[::-1], np.arange(28, 36)[::-1]]),
        ),
        (
            TEST_FILES_DIR / "simple-helix-8bp-5steps.conf",
            [8, 8],
            False,
            2,
            np.arange(5),
            np.tile(np.arange(5), (3, 1)).T,
            np.concatenate([np.arange(20, 28), np.arange(28, 36)]),
        ),
        (
            TEST_FILES_DIR / "simple-helix-8bp-5steps.conf",
            [8, 8],
            True,
            3,
            np.arange(5),
            np.tile(np.arange(5), (3, 1)).T,
            np.concatenate([np.arange(20, 28)[::-1], np.arange(28, 36)[::-1]]),
        ),
        (
            TEST_FILES_DIR / "simple-helix-8bp-5steps.conf",
            [8, 8],
            False,
            3,
            np.arange(5),
            np.tile(np.arange(5), (3, 1)).T,
            np.concatenate([np.arange(20, 28), np.arange(28, 36)]),
        ),
    ],
)
def test_trajectory_from_file(
    datafile: Path,
    strand_lengths: list[int],
    is_oxdna: bool,  # noqa: FBT001
    n_procs: int,
    expected_ts: np.ndarray,
    expected_energies: np.ndarray,
    expected_state_col1_vals: np.ndarray,
):
    trajectory = jdt.from_file(
        datafile,
        strand_lengths,
        is_oxdna=is_oxdna,
        n_processes=n_procs,
    )

    np.testing.assert_equal(expected_ts, trajectory.times)
    np.testing.assert_equal(expected_energies, trajectory.energies)
    for i in range(len(trajectory.states)):
        np.testing.assert_equal(
            expected_state_col1_vals,
            trajectory.states[i].array[:, 0].astype(int),
        )


def test_trajectory_from_file_raises_file_not_found():
    regx_pat = "^[" + re.escape(jdt.ERR_TRAJECTORY_FILE_NOT_FOUND.format("")) + "]"
    with pytest.raises(FileNotFoundError, match=regx_pat):
        jdt.from_file(TEST_FILES_DIR / "does-not-exist.conf", [8, 8])
