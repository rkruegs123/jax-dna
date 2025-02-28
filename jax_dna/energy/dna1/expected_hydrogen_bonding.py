"""Expected hydrogen bonding energy function for DNA1 model."""

import chex
import jax.numpy as jnp
from jax import vmap
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.utils as je_utils
import jax_dna.utils.types as typ
from jax_dna.energy.dna1 import hydrogen_bonding as jd_hb
from jax_dna.input import sequence_constraints as jd_sc


@chex.dataclass(frozen=True)
class ExpectedHydrogenBondingConfiguration(jd_hb.HydrogenBondingConfiguration):
    """Configuration for the expected hydrogen bonding energy function."""

    # New parameter
    sequence_constraints: jd_sc.SequenceConstraints | None = None

    # Override the `required_params` to include the new parameter
    required_params: tuple[str] = (
        *jd_hb.HydrogenBondingConfiguration.required_params,
        "sequence_constraints",
    )


@chex.dataclass(frozen=True)
class ExpectedHydrogenBonding(jd_hb.HydrogenBonding):
    """Expected hydrogen bonding energy function for DNA1 model."""

    params: ExpectedHydrogenBondingConfiguration

    def weight(self, i: int, j: int, seq: typ.Probabilistic_Sequence) -> float:
        """Computes the sequence-dependent weight for an unbonded pair."""
        sc = self.params.sequence_constraints

        return je_utils.compute_seq_dep_weight(
            seq, i, j, self.params.ss_hb_weights, sc.is_unpaired, sc.idx_to_unpaired_idx, sc.idx_to_bp_idx
        )

    @override
    def __call__(
        self,
        body: je_base.BaseNucleotide,
        seq: typ.Probabilistic_Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Scalar:
        # Compute sequence-independent energy for each unbonded pair
        v_hb = self.compute_v_hb(body, body, unbonded_neighbors)

        # Compute sequence-dependent weight for each unbonded pair
        op_i = unbonded_neighbors[0]
        op_j = unbonded_neighbors[1]
        hb_weights = vmap(self.weight, (0, 0, None))(op_i, op_j, seq)

        # Return the weighted sum
        return jnp.dot(hb_weights, v_hb)
