"""Stacking energy function for DNA1 model."""

import chex
import jax.numpy as jnp
from jax import vmap
from typing_extensions import override

import jax_dna.energy.dna1.nucleotide as dna1_nucleotide
import jax_dna.energy.utils as je_utils
import jax_dna.utils.types as typ
from jax_dna.energy.dna1 import stacking as jd_stk
from jax_dna.input import sequence_constraints as jd_sc


@chex.dataclass(frozen=True)
class ExpectedStackingConfiguration(jd_stk.StackingConfiguration):
    """Configuration for the expected stacking energy function."""

    # New parameter
    sequence_constraints: jd_sc.SequenceConstraints | None = None

    # Override the `required_params` to include the new parameter
    required_params: tuple[str] = (
        *jd_stk.StackingConfiguration.required_params,
        "sequence_constraints",
    )


@chex.dataclass(frozen=True)
class ExpectedStacking(jd_stk.Stacking):
    """Expected stacking energy function for DNA1 model."""

    params: ExpectedStackingConfiguration

    def weight(self, i: int, j: int, seq: typ.Probabilistic_Sequence) -> float:
        """Computes the sequence-dependent weight for a bonded pair."""
        sc = self.params.sequence_constraints
        return je_utils.compute_seq_dep_weight(
            seq, i, j, self.params.ss_stack_weights, sc.is_unpaired, sc.idx_to_unpaired_idx, sc.idx_to_bp_idx
        )

    @override
    def __call__(
        self,
        body: dna1_nucleotide.Nucleotide,
        seq: typ.Probabilistic_Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Scalar:
        # Compute sequence-independent energy for each bonded pair
        v_stack = self.compute_v_stack(body, bonded_neighbors)

        # Compute sequence-dependent weight for each bonded pair
        nn_i = bonded_neighbors[:, 0]
        nn_j = bonded_neighbors[:, 1]
        stack_weights = vmap(self.weight, (0, 0, None))(nn_i, nn_j, seq)

        # Return the weighted sum
        return jnp.dot(stack_weights, v_stack)
