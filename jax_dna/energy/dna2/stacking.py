"""Stacking energy function for DNA1 model."""


import chex
import jax.numpy as jnp
from typing_extensions import override

import jax_dna.energy.dna1 as jd_energy1
import jax_dna.energy.dna2.nucleotide as dna2_nucleotide
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class Stacking(jd_energy1.Stacking):
    """Stacking energy function for DNA2 model."""

    params: jd_energy1.StackingConfiguration

    def pairwise_energies(
        self,
        body: dna2_nucleotide.Nucleotide,
        seq: typ.Discrete_Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
    ) -> typ.Arr_Bonded_Neighbors:
        """Computes the stacking energy for each bonded pair."""
        # Compute sequence-independent energy for each bonded pair
        v_stack = self.compute_v_stack(
            body.stack_sites,
            body.back_sites_dna1,
            body.base_normals,
            body.cross_prods,
            bonded_neighbors
        )

        # Compute sequence-dependent weight for each bonded pair
        nn_i = bonded_neighbors[:, 0]
        nn_j = bonded_neighbors[:, 1]
        stack_weights = self.params.ss_stack_weights[seq[nn_i], seq[nn_j]]

        return jnp.multiply(stack_weights, v_stack)

    @override
    def __call__(
        self,
        body: dna2_nucleotide.Nucleotide,
        seq: typ.Discrete_Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Scalar:
        dgs = self.pairwise_energies(body, seq, bonded_neighbors)
        return dgs.sum()
