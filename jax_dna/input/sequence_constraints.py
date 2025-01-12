"""Sequence constraint information for DNA/RNA."""


import chex
import jax.numpy as jnp
import numpy as np

import jax_dna.utils.constants as jd_const
import jax_dna.utils.types as typ

ERR_SEQ_CONSTRAINTS_INVALID_NUMBER_NUCLEOTIDES = "Invalid number of nucleotides"
ERR_SEQ_CONSTRAINTS_INVALID_UNPAIRED_SHAPE = "Invalid shape for unpaired nucleotides"
ERR_INVALID_BP_SHAPE = "Invalid shape for base pairs"
ERR_SEQ_CONSTRAINTS_INVALID_IS_UNPAIRED_SHAPE = "Invalid shape for array specifying if unpaired"
ERR_SEQ_CONSTRAINTS_INVALID_UNPAIRED_MAPPER_SHAPE = "Invalid shape for unpaired nucleotide index mapper"
ERR_SEQ_CONSTRAINTS_INVALID_BP_MAPPER_SHAPE = "Invalid shape for base pair index mapper"
ERR_SEQ_CONSTRAINTS_MISMATCH_NUM_TYPES = (
    "Number of nucleotides should equal the number of unpaired base pairs plus the number of coupled base pairs"
)
ERR_SEQ_CONSTRAINTS_INVALID_COVER = "Unpaired and coupled nucleotides do not cover all nucleotides"
ERR_SEQ_CONSTRAINTS_IS_UNPAIRED_INVALID_VALUES = (
    "Array specifying if unpaired contains invalid values, can only be one-hot"
)
ERR_SEQ_CONSTRAINTS_INVALID_IS_UNPAIRED = "Array specifying if is_unpaired disagrees with list of unpaired nucleotides"
ERR_SEQ_CONSTRAINTS_PAIRED_NT_MAPPED_TO_UNPAIRED = "Base paired nucleotides cannot be mapped to an unpaired nucleotide"
ERR_SEQ_CONSTRAINTS_INCOMPLETE_UNPAIRED_MAPPED_IDXS = (
    "Map of position indices to indices of unpaired nucleotides does not cover number of unpaired nucleotides"
)
ERR_SEQ_CONSTRAINTS_UNPAIRED_NT_MAPPED_TO_PAIRED = "Unpaired nucleotides cannot be mapped to a base paired nucleotide"
ERR_SEQ_CONSTRAINTS_INCOMPLETE_BP_MAPPED_IDXS = (
    "Map of position indices to indices of base paired nucleotides does not cover number of base paired nucleotides"
)
ERR_BP_ARR_CONTAINS_DUPLICATES = "Array specifying base paired indices cannot contain duplicates"
ERR_INVALID_BP_INDICES = "Base paired indices must be between 0 and n_nucleotides-1"
ERR_DSEQ_TO_PSEQ_INVALID_BP = (
    "Invalid base pair encountered when converting discrete sequence to probabilistic sequence"
)


def check_consistent_constraints(
    n_unpaired: int,
    n_bp: int,
    unpaired: typ.Arr_Unpaired,
    idx_to_unpaired_idx: typ.Arr_Nucleotide_Int,
    idx_to_bp_idx: typ.Arr_Nucleotide_2_Int
) -> None:
    """Checks for consistency between specified nucleotide constraints and index mappers."""
    unpaired_mapped_idxs = []
    for idx, mapped_idx in enumerate(np.array(idx_to_unpaired_idx)):
        if idx in set(np.array(unpaired)):
            unpaired_mapped_idxs.append(mapped_idx)
        elif mapped_idx != -1:
            raise ValueError(ERR_SEQ_CONSTRAINTS_PAIRED_NT_MAPPED_TO_UNPAIRED)
    if set(unpaired_mapped_idxs) != set(np.arange(n_unpaired)):
        raise ValueError(ERR_SEQ_CONSTRAINTS_INCOMPLETE_UNPAIRED_MAPPED_IDXS)

    bp_mapped_idxs = []
    for idx, (mapped_idx1, mapped_idx2) in enumerate(np.array(idx_to_bp_idx)):
        if idx not in set(np.array(unpaired)):
            bp_mapped_idxs.append((mapped_idx1, mapped_idx2))
        elif mapped_idx1 != -1 or mapped_idx2 != -1:
            raise ValueError(ERR_SEQ_CONSTRAINTS_UNPAIRED_NT_MAPPED_TO_PAIRED)
    expected_bp_idxs = [(bp_idx, 0) for bp_idx in range(n_bp)]
    expected_bp_idxs += [(bp_idx, 1) for bp_idx in range(n_bp)]
    if set(bp_mapped_idxs) != set(expected_bp_idxs):
        raise ValueError(ERR_SEQ_CONSTRAINTS_INCOMPLETE_BP_MAPPED_IDXS)


def check_cover(
    n_nucleotides: int,
    n_unpaired: int,
    n_bp: int,
    unpaired: typ.Arr_Unpaired,
    bps: typ.Arr_Bp
) -> None:
    """Checks if unpaired and paired nucleotides cover the entire set of nucleotides."""
    if n_unpaired + 2 * n_bp != n_nucleotides:
        raise ValueError(ERR_SEQ_CONSTRAINTS_MISMATCH_NUM_TYPES)
    if set(np.concatenate([unpaired, bps.flatten()])) != set(np.arange(n_nucleotides)):
        raise ValueError(ERR_SEQ_CONSTRAINTS_INVALID_COVER)


@chex.dataclass(frozen=True)
class SequenceConstraints:
    """Constraint information for a RNA/DNA strand."""

    n_nucleotides: int
    n_unpaired: int
    n_bp: int
    is_unpaired: typ.Arr_Nucleotide_Int

    unpaired: typ.Arr_Unpaired
    bps: typ.Arr_Bp

    idx_to_unpaired_idx: typ.Arr_Nucleotide_Int
    idx_to_bp_idx: typ.Arr_Nucleotide_2_Int

    def __post_init__(self) -> None:
        """Check that the sequence constraints are valid."""
        # Check valid numbers
        if self.n_nucleotides < 1:
            raise ValueError(ERR_SEQ_CONSTRAINTS_INVALID_NUMBER_NUCLEOTIDES)

        # Check valid shapes
        if self.unpaired.shape != (self.n_unpaired,):
            raise ValueError(ERR_SEQ_CONSTRAINTS_INVALID_UNPAIRED_SHAPE)
        if self.bps.shape != (self.n_bp, 2):
            raise ValueError(ERR_INVALID_BP_SHAPE)
        if self.is_unpaired.shape != (self.n_nucleotides,):
            raise ValueError(ERR_SEQ_CONSTRAINTS_INVALID_IS_UNPAIRED_SHAPE)
        if self.idx_to_unpaired_idx.shape != (self.n_nucleotides,):
            raise ValueError(ERR_SEQ_CONSTRAINTS_INVALID_UNPAIRED_MAPPER_SHAPE)
        if self.idx_to_bp_idx.shape != (self.n_nucleotides, 2):
            raise ValueError(ERR_SEQ_CONSTRAINTS_INVALID_BP_MAPPER_SHAPE)

        # Check cover
        check_cover(self.n_nucleotides, self.n_unpaired, self.n_bp, self.unpaired, self.bps)

        # Check values
        if not set(np.array(self.is_unpaired)).issubset({0, 1}):
            raise ValueError(ERR_SEQ_CONSTRAINTS_IS_UNPAIRED_INVALID_VALUES)
        for idx, idx_unpaired in enumerate(self.is_unpaired):
            valid = idx_unpaired == 1 if idx in set(np.array(self.unpaired)) else idx_unpaired == 0

            if not valid:
                raise ValueError(ERR_SEQ_CONSTRAINTS_INVALID_IS_UNPAIRED)


        check_consistent_constraints(
            self.n_unpaired,
            self.n_bp,
            self.unpaired,
            self.idx_to_unpaired_idx,
            self.idx_to_bp_idx,
        )


def from_bps(n_nucleotides: int, bps: typ.Arr_Bp) -> SequenceConstraints:
    """Construct a SequenceConstraints object from a set of base pairs."""
    # Check format of base pairs
    if len(bps.shape) != jd_const.TWO_DIMENSIONS or bps.shape[1] != jd_const.N_NT_PER_BP \
       or jd_const.N_NT_PER_BP * bps.shape[0] > n_nucleotides:
        raise ValueError(ERR_INVALID_BP_SHAPE)

    paired_nucleotides = bps.flatten()

    has_duplicates = len(np.unique(paired_nucleotides)) < len(paired_nucleotides)
    if has_duplicates:
        raise ValueError(ERR_BP_ARR_CONTAINS_DUPLICATES)

    in_range = np.all((paired_nucleotides >= 0) & (paired_nucleotides < n_nucleotides))
    if not in_range:
        raise ValueError(ERR_INVALID_BP_INDICES)

    # Infer the unpaired nucleotides
    unpaired = np.setdiff1d(np.arange(n_nucleotides), paired_nucleotides)
    n_unpaired = unpaired.shape[0]

    # Construct the index mapper for unpaired nucleotides
    idx_to_unpaired_idx = np.full((n_nucleotides,), -1, dtype=np.int32)
    for up_idx, idx in enumerate(unpaired):
        idx_to_unpaired_idx[idx] = up_idx
    idx_to_unpaired_idx = np.array(idx_to_unpaired_idx)

    # Construct the index mapper for base paired nucleotides
    idx_to_bp_idx = np.full((n_nucleotides, 2), -1, dtype=np.int32)
    for bp_idx, (nt1, nt2) in enumerate(bps):
        idx_to_bp_idx[nt1] = [bp_idx, 0]
        idx_to_bp_idx[nt2] = [bp_idx, 1]
    idx_to_bp_idx = np.array(idx_to_bp_idx)

    # Construct additional metadata
    is_unpaired = np.array([(i in set(np.array(unpaired))) for i in range(n_nucleotides)])
    n_bp = bps.shape[0]

    # Construct a SequenceConstraints object
    return SequenceConstraints(
        n_nucleotides=n_nucleotides,
        n_unpaired=n_unpaired,
        n_bp=n_bp,
        is_unpaired=jnp.array(is_unpaired),
        unpaired=jnp.array(unpaired),
        bps=jnp.array(bps),
        idx_to_unpaired_idx=jnp.array(idx_to_unpaired_idx),
        idx_to_bp_idx=jnp.array(idx_to_bp_idx),
    )


def dseq_to_pseq(dseq: typ.Discrete_Sequence, sc: SequenceConstraints) -> typ.Probabilistic_Sequence:
    """Converts a discrete sequence to a probabilistic sequence."""
    # First, generate unpaired pseq
    unpaired = sc.unpaired
    n_unpaired = sc.n_unpaired

    up_pseq = np.zeros((n_unpaired, jd_const.N_NT), dtype=np.float64)
    for up_idx, idx in enumerate(unpaired):
        nt = dseq[idx]
        up_pseq[up_idx][nt] = 1.0

    # Second, generate base paired pseq
    bps = sc.bps
    n_bp = sc.n_bp

    bp_pseq = np.zeros((n_bp, 4), dtype=np.float64)
    for bp_idx, (idx1, idx2) in enumerate(bps):
        nt1, nt2 = dseq[idx1], dseq[idx2]

        bp_tuple = (int(nt1), int(nt2))
        if bp_tuple not in jd_const.BP_IDX_MAP:
            raise ValueError(ERR_DSEQ_TO_PSEQ_INVALID_BP)

        bp_type_idx = jd_const.BP_IDX_MAP[bp_tuple]
        bp_pseq[bp_idx][bp_type_idx] = 1.0

    return (jnp.array(up_pseq), jnp.array(bp_pseq))
