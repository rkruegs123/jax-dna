"""Bonded excluded volume energy for DNA1 model."""

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
class BondedExcludedVolumeConfiguration(config.BaseConfiguration):
    """Configuration for the bonded excluded volume energy function."""

    # independent parameters
    nt_type: typ.Arr_Nucleotide | None = None
    ## DNA2-specific
    dna_eps_exc: float | None = None
    dna_dr_star_base: float | None = None
    dna_sigma_base: float | None = None
    dna_sigma_back_base: float | None = None
    dna_sigma_base_back: float | None = None
    dna_dr_star_back_base: float | None = None
    dna_dr_star_base_back: float | None = None
    ## RNA2-specific
    rna_eps_exc: float | None = None
    rna_dr_star_base: float | None = None
    rna_sigma_base: float | None = None
    rna_sigma_back_base: float | None = None
    rna_sigma_base_back: float | None = None
    rna_dr_star_back_base: float | None = None
    rna_dr_star_base_back: float | None = None

    # dependent parameters
    dna_config: dna1_energy.BondedExcludedVolumeConfiguration | None = None
    rna_config: dna1_energy.BondedExcludedVolumeConfiguration | None = None

    # override
    required_params: tuple[str] = (
        "nt_type",
        # DNA2-specific
        "dna_eps_exc",
        "dna_dr_star_base",
        "dna_sigma_base",
        "dna_sigma_back_base",
        "dna_sigma_base_back",
        "dna_dr_star_back_base",
        "dna_dr_star_base_back",
        # RNA2-specific
        "rna_eps_exc",
        "rna_dr_star_base",
        "rna_sigma_base",
        "rna_sigma_back_base",
        "rna_sigma_base_back",
        "rna_dr_star_back_base",
        "rna_dr_star_base_back",
    )

    @override
    def init_params(self) -> "BondedExcludedVolumeConfiguration":
        dna_config = dna1_energy.BondedExcludedVolumeConfiguration(
            eps_exc=self.dna_eps_exc,
            dr_star_base=self.dna_dr_star_base,
            sigma_base=self.dna_sigma_base,
            sigma_back_base=self.dna_sigma_back_base,
            sigma_base_back=self.dna_sigma_base_back,
            dr_star_back_base=self.dna_dr_star_back_base,
            dr_star_base_back=self.dna_dr_star_base_back,
        ).init_params()

        rna_config = dna1_energy.BondedExcludedVolumeConfiguration(
            eps_exc=self.rna_eps_exc,
            dr_star_base=self.rna_dr_star_base,
            sigma_base=self.rna_sigma_base,
            sigma_back_base=self.rna_sigma_back_base,
            sigma_base_back=self.rna_sigma_base_back,
            dr_star_back_base=self.rna_dr_star_back_base,
            dr_star_base_back=self.rna_dr_star_base_back,
        ).init_params()

        return self.replace(
            dna_config=dna_config,
            rna_config=rna_config,
        )


@chex.dataclass(frozen=True)
class BondedExcludedVolume(je_base.BaseEnergyFunction):
    """Bonded excluded volume energy function for NA1 model."""

    params: BondedExcludedVolumeConfiguration

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

        dna_dgs = dna1_energy.BondedExcludedVolume(
            displacement_fn=self.displacement_fn, params=self.params.dna_config
        ).pairwise_energies(
            body.dna,
            bonded_neighbors,
        )

        rna_dgs = dna1_energy.BondedExcludedVolume(
            displacement_fn=self.displacement_fn, params=self.params.rna_config
        ).pairwise_energies(
            body.rna,
            bonded_neighbors,
        )

        dgs = jnp.where(is_rna_bond, rna_dgs, dna_dgs)
        return dgs.sum()
