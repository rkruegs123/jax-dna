"""Trajectory information for RNA/DNA strands."""

import concurrent.futures as cf
import functools
import itertools
import multiprocessing as mp
from pathlib import Path

import chex
import jax.numpy as jnp
import jax_md
import numpy as np

import jax_dna.utils.math as jdm
import jax_dna.utils.types as typ

TRAJECTORY_TIMES_DIMS = 1
TRAJECTORY_ENERGIES_SHAPE = (None, 3)
NUCLEOTIDE_STATE_SHAPE = (None, 15)

ERR_TRAJECTORY_FILE_NOT_FOUND = "Trajectory file not found: {}"
ERR_TRAJECTORY_N_NUCLEOTIDE_STRAND_LEGNTHS = "n_nucleotides and sum(strand_lengths) do not match"
ERR_TRAJECTORY_TIMES_TYPE = "times must be a numpy array"
ERR_TRAJECTORY_ENERGIES_TYPE = "energies must be a numpy array"
ERR_TRAJECTORY_T_E_S_LENGTHS = "times, energies, and states do not have the same length"
ERR_TRAJECTORY_TIMES_DIMS = "times must be a 1D array"
ERR_TRAJECTORY_ENERGIES_SHAPE = "energies must be a 2D array with shape (n_states, 3)"

ERR_NUCLEOTIDE_STATE_TYPE = "Invalid type for nucleotide states:"
ERR_NUCLEOTIDE_STATE_SHAPE = "Invalid shape for nucleotide states:"

ERR_FIXED_BOX_SIZE = "Only trajecories in a fixed box size are supported"


@chex.dataclass(frozen=True)
class Trajectory:
    """Trajectory information for a RNA/DNA strand."""

    n_nucleotides: int
    strand_lengths: list[int]
    times: typ.Arr_States
    energies: typ.Arr_States_3
    states: list["NucleotideState"]

    def __post_init__(self) -> None:
        """Validate the input."""
        if self.n_nucleotides != sum(self.strand_lengths):
            raise ValueError(ERR_TRAJECTORY_N_NUCLEOTIDE_STRAND_LEGNTHS)

        if not isinstance(self.times, np.ndarray):
            raise TypeError(ERR_TRAJECTORY_TIMES_TYPE)

        if not isinstance(self.energies, np.ndarray):
            raise TypeError(ERR_TRAJECTORY_ENERGIES_TYPE)

        if len(self.times) != len(self.energies) or len(self.times) != len(self.states):
            raise ValueError(ERR_TRAJECTORY_T_E_S_LENGTHS)

        if len(self.times.shape) != TRAJECTORY_TIMES_DIMS:
            raise ValueError(ERR_TRAJECTORY_TIMES_DIMS)

        if (
            len(self.energies.shape) != len(TRAJECTORY_ENERGIES_SHAPE)
            or self.energies.shape[1] != TRAJECTORY_ENERGIES_SHAPE[1]
        ):
            raise ValueError(ERR_TRAJECTORY_ENERGIES_SHAPE)

    @property
    def state_rigid_bodies(self) -> list[jax_md.rigid_body.RigidBody]:
        return [state.to_rigid_body() for state in self.states]

    @property
    def state_rigid_body(self) -> jax_md.rigid_body.RigidBody:
        return jax_md.rigid_body.RigidBody(
            center=jnp.stack([state.com for state in self.states]),
            orientation=jax_md.rigid_body.Quaternion(jnp.stack([state.quaternions for state in self.states])),
        )

    def slice(self, key: int | slice) -> "Trajectory":
        """Get a subset of the trajectory."""
        return Trajectory(
            n_nucleotides=self.n_nucleotides,
            strand_lengths=self.strand_lengths,
            times=self.times[key],
            energies=self.energies[key],
            states=self.states[key],
        )

    def __repr__(self) -> str:
        """Return a string representation of the trajectory."""
        return "\n".join(
            [
                "Trajectory:",
                f"n_nucleotides: {self.n_nucleotides}",
                f"strand_lengths: {self.strand_lengths}",
                f"# times: {len(self.times)}",
                f"# energies: {len(self.energies)}",
                f"# states: {len(self.states)}",
            ]
        )


@chex.dataclass(frozen=True)
class NucleotideState:
    """State information for the nucleotides in a single state."""

    array: typ.Arr_Nucleotide_15

    def __post_init__(self) -> None:
        """Validate the input array."""
        if not isinstance(self.array, np.ndarray):
            raise TypeError(ERR_NUCLEOTIDE_STATE_TYPE + str(type(self.array)))
        if len(self.array.shape) != len(NUCLEOTIDE_STATE_SHAPE) or self.array.shape[1] != NUCLEOTIDE_STATE_SHAPE[1]:
            raise ValueError(ERR_NUCLEOTIDE_STATE_SHAPE + str(self.array.shape))

    @property
    def com(self) -> typ.Arr_Nucleotide_3:
        """Center of mass of the nucleotides."""
        return self.array[:, :3]

    @property
    def back_base_vector(self) -> typ.Arr_Nucleotide_3:
        """Backbone base vector."""
        return self.array[:, 3:6]

    @property
    def base_normal(self) -> typ.Arr_Nucleotide_3:
        """Base normal to the base plane."""
        return self.array[:, 6:9]

    @property
    def velocity(self) -> typ.Arr_Nucleotide_3:
        """Velocity of the nucleotides."""
        return self.array[:, 9:12]

    @property
    def angular_velocity(self) -> typ.Arr_Nucleotide_3:
        """Angular velocity of the nucleotides."""
        return self.array[:, 12:15]

    @property
    def euler_angles(self) -> tuple[typ.Arr_Nucleotide, typ.Arr_Nucleotide, typ.Arr_Nucleotide]:
        """Convert principal axes to Tait-Bryan Euler angles."""
        return jdm.principal_axes_to_euler_angles(
            self.back_base_vector,
            np.cross(self.base_normal, self.back_base_vector),
            self.base_normal,
        )

    @property
    def quaternions(self) -> typ.Arr_Nucleotide_4:
        """Convert Euler angles to quaternions."""
        return jdm.euler_angles_to_quaternion(*self.euler_angles)

    def to_rigid_body(self) -> jax_md.rigid_body.RigidBody:
        """Convert the nucleotide state to jax-md rigid bodies."""
        return jax_md.rigid_body.RigidBody(
            self.com,
            jax_md.rigid_body.Quaternion(self.quaternions),
        )


def validate_box_size(state_box_sizes: list[typ.Vector3D]) -> None:
    """Validate the volume for a simulation is fixed."""
    state_box_sizes = np.array(state_box_sizes)
    if not np.all(state_box_sizes == state_box_sizes[0]):
        raise ValueError(ERR_FIXED_BOX_SIZE)


def from_file(
    path: typ.PathOrStr,
    strand_lengths: list[int],
    *,
    is_oxdna: bool = True,
    n_processes: int = 1,
) -> Trajectory:
    """Parse a trajectory file.

    Trajectory files are in the following format:
    t = number
    b = number number number
    E = number number number
    com_x com_y com_z a1_x a1_y a1_z a3_x a3_y a3_z v_x v_y v_z L_x L_y L_z
    ...repeated n_nucleotides times in total
    com_x com_y com_z a1_x a1_y a1_z a3_x a3_y a3_z v_x v_y v_z L_x L_y L_z

    where the com_x, ..., L_z are all floating point numbers.

    This can be repeated a total of "timestep" number of times.

    In oxDNA the states are stored in 3'->5' order so we flip the order per strand
    and need the topology to get the boundaries of each strand.

    Args:
        path (PathOrStr): path to the trajectory file
        strand_lengths (list[int]): if this is an oxDNA trajectory,
            the lengths of each strand, so that they can be flipped to 5'->3' order
        is_oxdna (bool): whether the trajectory is in oxDNA format
        n_processes (int): number of processors to use for reading the file

    Returns:
        Trajectory: trajectory information

    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(ERR_TRAJECTORY_FILE_NOT_FOUND.format(path))

    boundaries = np.linspace(0, path.stat().st_size, n_processes + 1, dtype=np.int64)
    n_runs = len(boundaries) - 1
    with cf.ProcessPoolExecutor(n_processes, mp_context=mp.get_context("spawn")) as pool:
        vals = list(
            pool.map(
                _read_file_process_wrapper,
                zip(
                    itertools.repeat(path, times=n_runs),
                    boundaries[:-1],
                    boundaries[1:],
                    itertools.repeat(strand_lengths, times=n_runs),
                    itertools.repeat(is_oxdna, times=n_runs),
                    strict=True,
                ),
            ),
        )

    # this is now an list of iterables where each iterable is a concatenated
    # list of the output of _read_file for each process
    concatenated_vals = list(
        map(
            itertools.chain.from_iterable,
            zip(*vals, strict=False),
        )
    )

    # convert the iterables to lists and unpack list
    ts, bs, es, states = list(map(list, concatenated_vals))

    validate_box_size(bs)

    return Trajectory(
        n_nucleotides=sum(strand_lengths),
        strand_lengths=strand_lengths,
        times=np.array(ts, dtype=np.float64),
        energies=np.array(es, dtype=np.float64),
        states=list(map(lambda s: NucleotideState(array=s), states)),
    )


def _read_file_process_wrapper(
    args: tuple[Path, int, int, list[int], bool],
) -> tuple[
    list[typ.Scalar],
    list[typ.Vector3D],
    list[typ.Vector3D],
    list[typ.Arr_Nucleotide_15],
]:
    """Wrapper for reading a trajectory file."""
    file_path, start, end, strand_lengths, is_3p_5p = args
    return _read_file(file_path, start, end, strand_lengths, is_3p_5p=is_3p_5p)


def _read_file(
    file_path: Path,
    start: int,
    end: int,
    strand_lengths: list[int],
    *,
    is_3p_5p: bool,
) -> tuple[
    list[typ.Scalar],
    list[typ.Vector3D],
    list[typ.Vector3D],
    list[typ.Arr_Nucleotide_15],
]:
    """Read a trajectory file object."""
    # we don't know where we are in the file, but we can be only in one of two
    # situations: We are at the start of the state or we are in the midle of a
    # state. If we are in the middle of a state, we need to read until the next
    # state starts and then parse the states from there. Importantly, we need
    # to pass our 'end' if the end is in the middle of a state, because the
    # worker ahead of in the file will not read it.
    parse_str = functools.partial(np.fromstring, sep=" ", dtype=np.float64)

    state_length = sum(strand_lengths)
    strand_bounds = list(itertools.pairwise([0, *itertools.accumulate(strand_lengths)]))

    file_obj = file_path.open()
    file_obj.seek(start)

    line = file_obj.readline()
    while not line.startswith("t"):
        line = file_obj.readline()

    ts, bs, es, states = [], [], [], []
    state = []
    current = file_obj.tell()
    while current < end:
        if line[0] == "t":
            t = float(line.strip().split("=")[1])
            ts.append(t)
        elif line[0] == "b":
            b = parse_str(line.strip().split("=")[1])
            bs.append(b)
        elif line[0] == "E":
            e = parse_str(line.strip().split("=")[1])
            es.append(e)
        else:
            state.append(parse_str(line.strip()))
            if len(state) == state_length:
                # if the trajectory is stored in 3'->5' order, we need to flip
                # the order of the nucleotides in each strand
                if is_3p_5p:
                    state = list(itertools.chain.from_iterable([state[s:e][::-1] for s, e in strand_bounds]))
                states.append(np.array(state, dtype=np.float64))
                state = []
                current = file_obj.tell()

        line = file_obj.readline()

    return ts, bs, es, states
