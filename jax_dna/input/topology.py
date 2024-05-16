"""Topological information for DNA/RNA."""

import dataclasses as dc
import itertools
from pathlib import Path
from typing import Callable

import numpy as np

import jax_dna.utils.types as typ

NUCLEOTIDES_ONEHOT: dict[str, typ.Vector4D] = {
    "A": np.array([1, 0, 0, 0], dtype=np.float64),
    "G": np.array([0, 1, 0, 0], dtype=np.float64),
    "C": np.array([0, 0, 1, 0], dtype=np.float64),
    "T": np.array([0, 0, 0, 1], dtype=np.float64),
    "U": np.array([0, 0, 0, 1], dtype=np.float64),
}

N_1ST_LINE_OXDNA_CLASSIC = 2
N_1ST_LINE_OXDNA_NEW = 3
VALID_NEIGHBOR_SECOND_DIM = 2

ERR_TOPOLOGY_INVALID_NUMBER_NUCLEOTIDES = "Invalid number of nucleotides"
ERR_TOPOLOGY_INVALID_STRAND_COUNTS = "Invalid strand counts"
ERR_TOPOLOGY_SEQ_NOT_MATCH_NUCLEOTIDES = "Sequence does not match number of nucleotides"
ERR_TOPOLOGY_STRAND_COUNTS_NOT_MATCH = "Strand counts do not match number of nucleotides"
ERR_TOPOLOGY_BONDED_NEIGHBORS_INVALID_SHAPE = "Invalid bonded neighbors shape"
ERR_TOPOLOGY_UNBONDED_NEIGHBORS_INVALID_SHAPE = "Invalid unbonded neighbors shape"
ERR_TOPOLOGY_INVALID_SEQUENCE_LENGTH = "Invalid sequence length"
ERR_TOPOLOGY_INVALID_SEQUENCE_NUCLEOTIDES = "Invalid sequence nucleotides"
ERR_TOPOLOGY_INVALID_ONE_HOT_SEQUENCE_SHAPE = "Invalid one-hot sequence shape"
ERR_TOPOLOGY_INVALID_ONE_HOT_SEQUENCE_VALUES = "Invalid one-hot sequence values, must be 0 or 1 and sum to 1"
ERR_INVALID_OXDNA_FORMAT = "Invalid oxDNA topology format. See https://lorenzo-rovigatti.github.io/oxDNA/configurations.html#topology-file for more information."


@dc.dataclass(frozen=True)
class Topology:
    """Topology information for a RNA/DNA strand."""

    n_nucleotides: int
    strand_counts: np.ndarray
    bonded_neighbors: np.ndarray
    unbonded_neighbors: np.ndarray
    seq: str
    seq_one_hot: np.ndarray

    def __post_init__(self) -> None:
        """Check that the topology is valid."""
        if self.n_nucleotides < 1:
            raise ValueError(ERR_TOPOLOGY_INVALID_NUMBER_NUCLEOTIDES)

        if self.n_nucleotides != len(self.seq):
            raise ValueError(ERR_TOPOLOGY_SEQ_NOT_MATCH_NUCLEOTIDES)

        if len(self.strand_counts) == 0 or sum(self.strand_counts) == 0:
            raise ValueError(ERR_TOPOLOGY_INVALID_STRAND_COUNTS)

        if self.n_nucleotides != sum(self.strand_counts):
            raise ValueError(ERR_TOPOLOGY_STRAND_COUNTS_NOT_MATCH)

        if (
            len(self.bonded_neighbors.shape) != VALID_NEIGHBOR_SECOND_DIM
            or self.bonded_neighbors.shape[1] != VALID_NEIGHBOR_SECOND_DIM
        ):
            raise ValueError(ERR_TOPOLOGY_BONDED_NEIGHBORS_INVALID_SHAPE)

        if (
            len(self.unbonded_neighbors.shape) != VALID_NEIGHBOR_SECOND_DIM
            or self.unbonded_neighbors.shape[1] != VALID_NEIGHBOR_SECOND_DIM
        ):
            raise ValueError(ERR_TOPOLOGY_UNBONDED_NEIGHBORS_INVALID_SHAPE)

        if len(set(self.seq) - set("AGCTU")) > 0:
            raise ValueError(ERR_TOPOLOGY_INVALID_SEQUENCE_NUCLEOTIDES)

        if self.seq_one_hot.shape != (self.n_nucleotides, 4):
            raise ValueError(ERR_TOPOLOGY_INVALID_ONE_HOT_SEQUENCE_SHAPE)

        if self.seq_one_hot.astype(int).sum() != self.n_nucleotides:
            raise ValueError(ERR_TOPOLOGY_INVALID_ONE_HOT_SEQUENCE_VALUES)


def from_oxdna_file(path: typ.PathOrStr) -> Topology:
    """Read topology information from an oxDNA file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Topology file not found: {path}")

    with path.open() as f:
        lines = f.readlines()

    _, parse_f = _determine_oxdna_format(lines[0])

    return parse_f(lines)


def _determine_oxdna_format(first_line: str) -> tuple[typ.OxdnaFormat, Callable[[list[str]], Topology]]:
    """Determine the format of an oxDNA file from the first line of the file."""
    tokens = first_line.strip().split()

    if len(tokens) == N_1ST_LINE_OXDNA_CLASSIC:
        fmt = typ.OxdnaFormat.CLASSIC
        func = _from_file_oxdna_classic
    elif len(tokens) == N_1ST_LINE_OXDNA_NEW:
        fmt = typ.OxdnaFormat.NEW
        func = _from_file_oxdna_new
    else:
        raise ValueError(ERR_INVALID_OXDNA_FORMAT)

    return fmt, func


def _from_file_oxdna_classic(lines: list[str]) -> Topology:
    """Read topology information from a file in the classix oxDNA format.

    See https://lorenzo-rovigatti.github.io/oxDNA/configurations.html#topology-file
    for more information.

    Args:
        lines (List[str]): lines from topology file

    Returns:
        Topology: Topology object

    """
    n_nucleotides, n_strands = list(map(int, lines[0].strip().split()))

    # after the first line the topology files are space delimited with the
    # following columns:
    # - strand id (1 - indexed)
    # - nucleotide base (A=0, G=1, C=2, T=3, U=3), use char for now
    #   TODO(ryanhausen): support integers and special nucleotides
    #         https://lorenzo-rovigatti.github.io/oxDNA/configurations.html#special-nucleotides
    # - 3' neighbor (0-indexed), -1 if none
    # - 5' neighbor (0-indexed), -1 if none
    #
    # A more common convention is to store the nucleotides in the 5' -> 3' direction
    # so we need to reverse the order, which seems to be as easy as reversing the
    # order of the nucleotides per strand.
    # TODO(ryanhausen): confirm with @rkruegs123

    # splitting the lines and converting to the correct types could be done in
    # a more functional way
    # TODO(ryanhausen): refactor into function for testing
    strand_ids, bases, neighbor_5p, neighbor_3p = list(zip(*[line.strip().split() for line in lines[1:]], strict=True))
    strand_ids = list(map(int, strand_ids))
    # TODO(ryanhausen): unique sorts ids, is this ok to assume?
    _, strand_counts = np.unique(strand_ids, return_counts=True)
    neighbor_5p = list(map(int, neighbor_5p))
    neighbor_3p = list(map(int, neighbor_3p))

    # TODO(ryanhausen): this could be more clever, not sure if it's worth it
    # reverse the order of the nucleotides per strand
    # TODO(ryanhausen): refactor into function for testing
    reversed_bases = []
    for i in range(1, n_strands + 1):
        strand_bases = [
            id_nucleotide[1] for id_nucleotide in zip(strand_ids, bases, strict=True) if id_nucleotide[0] == i
        ]
        reversed_bases.extend(strand_bases[::-1])
    sequence = "".join(reversed_bases)

    # TODO: refactor into function for testing
    bonded_neighbors = set(
        filter(
            lambda nid_n3: nid_n3[1] != -1,
            enumerate(neighbor_3p),
        )
    )

    # get unbonded neighbors
    # TODO: refactor into function for testing
    all_possible_pairs = set(itertools.combinations(range(n_nucleotides), 2))
    self_bonds = {(i, i) for i in range(n_nucleotides)}
    bonded_pairs = bonded_neighbors
    unbonded_pairs = all_possible_pairs - bonded_pairs - self_bonds

    return Topology(
        n_nucleotides=n_nucleotides,
        strand_counts=strand_counts,
        bonded_neighbors=np.array(list(bonded_neighbors)),
        unbonded_neighbors=np.array(list(unbonded_pairs)),
        seq=sequence,
        seq_one_hot=np.array([NUCLEOTIDES_ONEHOT[s] for s in sequence], dtype=np.float64),
    )


def _from_file_oxdna_new(path: Path) -> Topology:
    _ = path
    raise NotImplementedError("New oxDNA format not yet supported")
