"""Unbonded excluded volume energy function for DNA1 model."""


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
class UnbondedExcludedVolumeConfiguration(config.BaseConfiguration):
    """Configuration for the unbonded excluded volume energy function."""

    # independent parameters
    nt_type: typ.Arr_Nucleotide | None = None
    ## DNA2-specific
    dna_eps_exc: float | None = None
    dna_dr_star_base: float | None = None
    dna_sigma_base: float | None = None
    dna_dr_star_back_base: float | None = None
    dna_sigma_back_base: float | None = None
    dna_dr_star_base_back: float | None = None
    dna_sigma_base_back: float | None = None
    dna_dr_star_backbone: float | None = None
    dna_sigma_backbone: float | None = None
    ## RNA2-specific
    rna_eps_exc: float | None = None
    rna_dr_star_base: float | None = None
    rna_sigma_base: float | None = None
    rna_dr_star_back_base: float | None = None
    rna_sigma_back_base: float | None = None
    rna_dr_star_base_back: float | None = None
    rna_sigma_base_back: float | None = None
    rna_dr_star_backbone: float | None = None
    rna_sigma_backbone: float | None = None
    ## DNA/RNA-hybrid-specific
    drh_eps_exc: float | None = None
    drh_dr_star_base: float | None = None
    drh_sigma_base: float | None = None
    drh_dr_star_back_base: float | None = None
    drh_sigma_back_base: float | None = None
    drh_dr_star_base_back: float | None = None
    drh_sigma_base_back: float | None = None
    drh_dr_star_backbone: float | None = None
    drh_sigma_backbone: float | None = None

    # dependent parameters
    dna_config: dna1_energy.UnbondedExcludedVolumeConfiguration | None = None
    rna_config: dna1_energy.UnbondedExcludedVolumeConfiguration | None = None
    drh_config: dna1_energy.UnbondedExcludedVolumeConfiguration | None = None

    # override
    required_params: tuple[str] = (
        "nt_type",
        # DNA2-specific
        "dna_eps_exc",
        "dna_dr_star_base",
        "dna_sigma_base",
        "dna_dr_star_back_base",
        "dna_sigma_back_base",
        "dna_dr_star_base_back",
        "dna_sigma_base_back",
        "dna_dr_star_backbone",
        "dna_sigma_backbone",
        # RNA2-specific
        "rna_eps_exc",
        "rna_dr_star_base",
        "rna_sigma_base",
        "rna_dr_star_back_base",
        "rna_sigma_back_base",
        "rna_dr_star_base_back",
        "rna_sigma_base_back",
        "rna_dr_star_backbone",
        "rna_sigma_backbone",
        # DNA/RNA-hybrid-specific
        "drh_eps_exc",
        "drh_dr_star_base",
        "drh_sigma_base",
        "drh_dr_star_back_base",
        "drh_sigma_back_base",
        "drh_dr_star_base_back",
        "drh_sigma_base_back",
        "drh_dr_star_backbone",
        "drh_sigma_backbone",
    )

    @override
    def init_params(self) -> "UnbondedExcludedVolumeConfiguration":
        dna_config = dna1_energy.UnbondedExcludedVolumeConfiguration(
            eps_exc=self.dna_eps_exc,
            dr_star_base=self.dna_dr_star_base,
            sigma_base=self.dna_sigma_base,
            sigma_back_base=self.dna_sigma_back_base,
            sigma_base_back=self.dna_sigma_base_back,
            dr_star_back_base=self.dna_dr_star_back_base,
            dr_star_base_back=self.dna_dr_star_base_back,
            dr_star_backbone=self.dna_dr_star_backbone,
            sigma_backbone=self.dna_sigma_backbone,
        ).init_params()

        rna_config = dna1_energy.UnbondedExcludedVolumeConfiguration(
            eps_exc=self.rna_eps_exc,
            dr_star_base=self.rna_dr_star_base,
            sigma_base=self.rna_sigma_base,
            sigma_back_base=self.rna_sigma_back_base,
            sigma_base_back=self.rna_sigma_base_back,
            dr_star_back_base=self.rna_dr_star_back_base,
            dr_star_base_back=self.rna_dr_star_base_back,
            dr_star_backbone=self.rna_dr_star_backbone,
            sigma_backbone=self.rna_sigma_backbone,
        ).init_params()

        drh_config = dna1_energy.UnbondedExcludedVolumeConfiguration(
            eps_exc=self.drh_eps_exc,
            dr_star_base=self.drh_dr_star_base,
            sigma_base=self.drh_sigma_base,
            sigma_back_base=self.drh_sigma_back_base,
            sigma_base_back=self.drh_sigma_base_back,
            dr_star_back_base=self.drh_dr_star_back_base,
            dr_star_base_back=self.drh_dr_star_base_back,
            dr_star_backbone=self.drh_dr_star_backbone,
            sigma_backbone=self.drh_sigma_backbone,
        ).init_params()

        return self.replace(
            dna_config=dna_config,
            rna_config=rna_config,
            drh_config=drh_config,
        )


@chex.dataclass(frozen=True)
class UnbondedExcludedVolume(je_base.BaseEnergyFunction):
    """Unbonded excluded volume energy function for NA1 model."""

    params: UnbondedExcludedVolumeConfiguration

    @override
    def __call__(
        self,
        body: na1_nucleotide.HybridNucleotide,
        seq: typ.Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Scalar:
        op_i = unbonded_neighbors[0]
        op_j = unbonded_neighbors[1]

        is_rna_bond = jax.vmap(je_utils.is_rna_pair, (0, 0, None))(op_i, op_j, self.params.nt_type)
        is_drh_bond = jax.vmap(je_utils.is_dna_rna_pair, (0, 0, None))(op_i, op_j, self.params.nt_type)
        is_rdh_bond = jax.vmap(je_utils.is_dna_rna_pair, (0, 0, None))(op_j, op_i, self.params.nt_type)

        mask = jnp.array(op_i < body.dna.center.shape[0], dtype=jnp.float32)


        dna_dgs = dna1_energy.UnbondedExcludedVolume(
            displacement_fn=self.displacement_fn,
            params=self.params.dna_config
        ).pairwise_energies(
            body.dna,
            body.dna,
            unbonded_neighbors,
        )

        rna_dgs = dna1_energy.UnbondedExcludedVolume(
            displacement_fn=self.displacement_fn,
            params=self.params.rna_config
        ).pairwise_energies(
            body.rna,
            body.rna,
            unbonded_neighbors,
        )

        drh_dgs = dna1_energy.UnbondedExcludedVolume(
            displacement_fn=self.displacement_fn,
            params=self.params.drh_config
        ).pairwise_energies(
            body.dna,
            body.rna,
            unbonded_neighbors,
        )

        rdh_dgs = dna1_energy.UnbondedExcludedVolume(
            displacement_fn=self.displacement_fn,
            params=self.params.drh_config
        ).pairwise_energies(
            body.rna,
            body.dna,
            unbonded_neighbors,
        )


        dgs = jnp.where(is_rna_bond, rna_dgs,
                        jnp.where(is_drh_bond, drh_dgs,
                                  jnp.where(is_rdh_bond, rdh_dgs, dna_dgs)))
        dgs = jnp.where(mask, dgs, 0.0)

        return dgs.sum()
