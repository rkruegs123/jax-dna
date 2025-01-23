"""Utility functions for energy calculations."""

import functools

import jax.numpy as jnp
import jax_md
import numpy as np
from jax import vmap

import jax_dna.utils.constants as jd_const
import jax_dna.utils.types as typ


@vmap
def q_to_back_base(q: jax_md.rigid_body.Quaternion) -> jnp.ndarray:
    """Get the vector from the center to the base of the nucleotide."""
    q0, q1, q2, q3 = q.vec
    return jnp.array([q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)])


@vmap
def q_to_base_normal(q: jax_md.rigid_body.Quaternion) -> jnp.ndarray:
    """Get the normal vector to the base of the nucleotide."""
    q0, q1, q2, q3 = q.vec
    return jnp.array([2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), q0**2 - q1**2 - q2**2 + q3**2])


@vmap
def q_to_cross_prod(q: jax_md.rigid_body.Quaternion) -> jnp.ndarray:
    """Get the cross product vector of the nucleotide."""
    q0, q1, q2, q3 = q.vec
    return jnp.array([2 * (q1 * q2 - q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 + q0 * q1)])


@functools.partial(vmap, in_axes=(None, 0, 0), out_axes=0)
def get_pair_probs(seq: typ.Arr_Nucleotide_4, i: int, j: int) -> jnp.ndarray:
    """Get the pair probabilities for a sequence."""
    return jnp.kron(seq[i], seq[j])


def compute_seq_dep_weight(
    pseq: typ.Probabilistic_Sequence,
    nt1: int,
    nt2: int,
    weights_table: np.ndarray,
    is_unpaired: typ.Arr_Nucleotide_Int,
    idx_to_unpaired_idx: typ.Arr_Nucleotide_Int,
    idx_to_bp_idx: typ.Arr_Nucleotide_2_Int,
) -> float:
    """Computes the sequence-dependent weight for an interaction given a probabilistic sequence."""
    unpaired_pseq, bp_pseq = pseq
    flattened_weights_table = weights_table.flatten()

    nt1_unpaired = is_unpaired[nt1]
    nt2_unpaired = is_unpaired[nt2]

    # Case 1: Both unpaired
    pair_probs = jnp.kron(unpaired_pseq[idx_to_unpaired_idx[nt1]], unpaired_pseq[idx_to_unpaired_idx[nt2]])
    pair_weight_unpaired = jnp.dot(pair_probs, flattened_weights_table)

    # Case 2: nt1 unpaired, nt2 base paired
    nt1_nt_probs = unpaired_pseq[idx_to_unpaired_idx[nt1]]
    nt2_bp_idx, within_nt2_bp_idx = idx_to_bp_idx[nt2]
    bp_probs = bp_pseq[nt2_bp_idx]

    def nt1_up_fn(nt1_nt: int, nt2_bp_type_idx: int) -> float:
        nt2_nt = jd_const.BP_IDXS[nt2_bp_type_idx][within_nt2_bp_idx]
        return nt1_nt_probs[nt1_nt] * bp_probs[nt2_bp_type_idx] * weights_table[nt1_nt, nt2_nt]

    pair_weight_nt1_up = vmap(vmap(nt1_up_fn, (None, 0)), (0, None))(
        jnp.arange(jd_const.N_NT), jnp.arange(jd_const.N_BP_TYPES)
    ).sum()

    # Case 3: nt2 unpaired, nt1 base paired

    nt2_nt_probs = unpaired_pseq[idx_to_unpaired_idx[nt2]]
    nt1_bp_idx, within_nt1_bp_idx = idx_to_bp_idx[nt1]
    bp_probs = bp_pseq[nt1_bp_idx]

    def nt2_up_fn(nt2_nt: int, nt1_bp_type_idx: int) -> float:
        nt1_nt = jd_const.BP_IDXS[nt1_bp_type_idx][within_nt1_bp_idx]
        return nt2_nt_probs[nt2_nt] * bp_probs[nt1_bp_type_idx] * weights_table[nt1_nt, nt2_nt]

    pair_weight_nt2_up = vmap(vmap(nt2_up_fn, (None, 0)), (0, None))(
        jnp.arange(jd_const.N_NT), jnp.arange(jd_const.N_BP_TYPES)
    ).sum()

    # Case 4: both nt1 and nt2 are base paired

    nt1_bp_idx, within_nt1_bp_idx = idx_to_bp_idx[nt1]
    nt2_bp_idx, within_nt2_bp_idx = idx_to_bp_idx[nt2]

    ## Case 4.I: nt1 and nt2 are in the same base pair
    bp_probs = bp_pseq[nt1_bp_idx]

    def same_bp_fn(bp_idx: int) -> float:
        bp_prob = bp_probs[bp_idx]
        bp_nt1, bp_nt2 = jd_const.BP_IDXS[bp_idx][jnp.array([within_nt1_bp_idx, within_nt2_bp_idx])]
        return bp_prob * weights_table[bp_nt1, bp_nt2]

    pair_weight_same_bp = vmap(same_bp_fn)(jnp.arange(jd_const.N_BP_TYPES)).sum()

    ## Case 4.II: nt1 and nt2 are in different base pairs
    bp1_probs = bp_pseq[nt1_bp_idx]
    bp2_probs = bp_pseq[nt2_bp_idx]

    def diff_bps_fn(bp1_idx: int, bp2_idx: int) -> float:
        bp1_prob = bp1_probs[bp1_idx]
        nt1_nt = jd_const.BP_IDXS[bp1_idx][within_nt1_bp_idx]

        bp2_prob = bp2_probs[bp2_idx]
        nt2_nt = jd_const.BP_IDXS[bp2_idx][within_nt2_bp_idx]

        return bp1_prob * bp2_prob * weights_table[nt1_nt, nt2_nt]

    pair_weight_diff_bps = vmap(vmap(diff_bps_fn, (None, 0)), (0, None))(
        jnp.arange(jd_const.N_BP_TYPES), jnp.arange(jd_const.N_BP_TYPES)
    ).sum()

    pair_weight_both_paired = jnp.where(nt1_bp_idx == nt2_bp_idx, pair_weight_same_bp, pair_weight_diff_bps)

    return jnp.where(
        jnp.logical_and(nt1_unpaired, nt2_unpaired),
        pair_weight_unpaired,
        jnp.where(
            nt1_unpaired, pair_weight_nt1_up, jnp.where(nt2_unpaired, pair_weight_nt2_up, pair_weight_both_paired)
        ),
    )
