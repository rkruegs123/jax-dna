"""FENE energy function for NA1 model."""

import dataclasses as dc

import chex
import jax
import jax.numpy as jnp
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.energy.na1.nucleotide as na1_nucleotide
import jax_dna.energy.na1.utils as je_utils
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class FeneConfiguration(config.BaseConfiguration):
    """Configuration for the FENE energy function."""

    # independent parameters
    nt_type: typ.Arr_Nucleotide | None = None
    ## DNA2-specific
    dna_eps_backbone: float | None = None
    dna_r0_backbone: float | None = None
    dna_delta_backbone: float | None = None
    dna_fmax: float | None = None
    dna_finf: float | None = None
    ## RNA2-specific
    rna_eps_backbone: float | None = None
    rna_r0_backbone: float | None = None
    rna_delta_backbone: float | None = None
    rna_fmax: float | None = None
    rna_finf: float | None = None

    # dependent parameters
    dna_config: dna1_energy.FeneConfiguration | None = None
    rna_config: dna1_energy.FeneConfiguration | None = None

    # override
    required_params: tuple[str] = (
        "nt_type",
        # DNA2-specific
        "dna_eps_backbone",
        "dna_r0_backbone",
        "dna_delta_backbone",
        "dna_fmax",
        "dna_finf",
        # RNA2-specific
        "rna_eps_backbone",
        "rna_r0_backbone",
        "rna_delta_backbone",
        "rna_fmax",
        "rna_finf",
    )

    @override
    def init_params(self) -> "FeneConfiguration":
        dna_config = dna1_energy.FeneConfiguration(
            eps_backbone=self.dna_eps_backbone,
            r0_backbone=self.dna_r0_backbone,
            delta_backbone=self.dna_delta_backbone,
            fmax=self.dna_fmax,
            finf=self.dna_finf,
        ).init_params()

        rna_config = dna1_energy.FeneConfiguration(
            eps_backbone=self.rna_eps_backbone,
            r0_backbone=self.rna_r0_backbone,
            delta_backbone=self.rna_delta_backbone,
            fmax=self.rna_fmax,
            finf=self.rna_finf,
        ).init_params()

        return dc.replace(
            self,
            dna_config=dna_config,
            rna_config=rna_config,
        )


@chex.dataclass(frozen=True)
class Fene(je_base.BaseEnergyFunction):
    """FENE energy function for NA1 model."""

    params: FeneConfiguration

    @override
    def __call__(
        self,
        body: na1_nucleotide.HybridNucleotide,
        seq: typ.Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Scalar:
        nn_i = bonded_neighbors[:, 0]
        nn_j = bonded_neighbors[:, 1]

        is_rna_bond = jax.vmap(je_utils.is_rna_pair, (0, 0, None))(nn_i, nn_j, self.params.nt_type)

        dna_dgs = dna1_energy.Fene(
            displacement_fn=self.displacement_fn, params=self.params.dna_config
        ).pairwise_energies(
            body.dna,
            bonded_neighbors,
        )

        rna_dgs = dna1_energy.Fene(
            displacement_fn=self.displacement_fn, params=self.params.rna_config
        ).pairwise_energies(
            body.rna,
            bonded_neighbors,
        )

        dgs = jnp.where(is_rna_bond, rna_dgs, dna_dgs)
        return dgs.sum()
