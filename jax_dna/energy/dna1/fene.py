"""FENE energy function for DNA1 model."""

import chex
import jax.numpy as jnp
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class FeneConfiguration(config.BaseConfiguration):
    """Configuration for the FENE energy function."""

    eps_backbone: float | None = None
    r0_backbone: float | None = None
    delta_backbone: float | None = None
    fmax: float | None = None
    finf: float | None = None

    # override
    required_params: tuple[str] = ("eps_backbone", "r0_backbone", "delta_backbone", "fmax", "finf")

    @override
    def init_params(self) -> "FeneConfiguration":
        return self


@chex.dataclass(frozen=True)
class Fene(je_base.BaseEnergyFunction):
    """FENE energy function for DNA1 model."""

    params: FeneConfiguration

    def pairwise_energies(
        self,
        body: je_base.BaseNucleotide,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
    ) -> typ.Arr_Bonded_Neighbors:
        """Computes the FENE energy for each bonded pair."""
        nn_i = bonded_neighbors[:, 0]
        nn_j = bonded_neighbors[:, 1]

        dr_back_nn = self.displacement_mapped(body.back_sites[nn_i], body.back_sites[nn_j])
        r_back_nn = jnp.linalg.norm(dr_back_nn, axis=1)

        return dna1_interactions.v_fene_smooth(
            r_back_nn,
            self.params.eps_backbone,
            self.params.r0_backbone,
            self.params.delta_backbone,
            self.params.fmax,
            self.params.finf,
        )

    @override
    def __call__(
        self,
        body: je_base.BaseNucleotide,
        seq: typ.Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Scalar:
        dgs = self.pairwise_energies(body, bonded_neighbors)
        return dgs.sum()
