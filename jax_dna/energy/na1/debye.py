"""Debye-huckel function for NA2 model."""

import dataclasses as dc

import chex
import jax
import jax.numpy as jnp
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna2 as dna2_energy
import jax_dna.energy.na1.nucleotide as na1_nucleotide
import jax_dna.energy.na1.utils as je_utils
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class DebyeConfiguration(config.BaseConfiguration):
    """Configuration for the debye-huckel energy function."""

    # independent parameters
    nt_type: typ.Arr_Nucleotide | None = None
    is_end: typ.Arr_Nucleotide_Int | None = None
    half_charged_ends: bool | None = None
    kt: float | None = None
    salt_conc: float | None = None
    ## DNA2-specific
    dna_q_eff: float | None = None
    dna_lambda_factor: float | None = None
    dna_prefactor_coeff: float | None = None
    ## RNA2-specific
    rna_q_eff: float | None = None
    rna_lambda_factor: float | None = None
    rna_prefactor_coeff: float | None = None
    ## DNA/RNA-hybrid-specific
    drh_q_eff: float | None = None
    drh_lambda_factor: float | None = None
    drh_prefactor_coeff: float | None = None

    # dependent parameters
    dna_config: dna2_energy.DebyeConfiguration | None = None
    rna_config: dna2_energy.DebyeConfiguration | None = None
    drh_config: dna2_energy.DebyeConfiguration | None = None

    # override
    required_params: tuple[str] = (
        "nt_type",
        "is_end",
        "half_charged_ends",
        "kt",
        "salt_conc",
        # DNA2-specific
        "dna_q_eff",
        "dna_lambda_factor",
        "dna_prefactor_coeff",
        # RNA2-specific
        "rna_q_eff",
        "rna_lambda_factor",
        "rna_prefactor_coeff",
    )

    @override
    def init_params(self) -> "DebyeConfiguration":
        dna_config = dna2_energy.DebyeConfiguration(
            q_eff=self.dna_q_eff,
            lambda_factor=self.dna_lambda_factor,
            prefactor_coeff=self.dna_prefactor_coeff,
            kt=self.kt,
            salt_conc=self.salt_conc,
            is_end=self.is_end,
            half_charged_ends=self.half_charged_ends,
        ).init_params()

        rna_config = dna2_energy.DebyeConfiguration(
            q_eff=self.rna_q_eff,
            lambda_factor=self.rna_lambda_factor,
            prefactor_coeff=self.rna_prefactor_coeff,
            kt=self.kt,
            salt_conc=self.salt_conc,
            is_end=self.is_end,
            half_charged_ends=self.half_charged_ends,
        ).init_params()

        drh_config = dna2_energy.DebyeConfiguration(
            q_eff=self.drh_q_eff,
            lambda_factor=self.drh_lambda_factor,
            prefactor_coeff=self.drh_prefactor_coeff,
            kt=self.kt,
            salt_conc=self.salt_conc,
            is_end=self.is_end,
            half_charged_ends=self.half_charged_ends,
        ).init_params()

        return dc.replace(
            self,
            dna_config=dna_config,
            rna_config=rna_config,
            drh_config=drh_config,
        )


@chex.dataclass(frozen=True)
class Debye(je_base.BaseEnergyFunction):
    """Debye-huckel energy function for NA1 model."""

    params: DebyeConfiguration

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

        dna_dgs = dna2_energy.Debye(
            displacement_fn=self.displacement_fn, params=self.params.dna_config
        ).pairwise_energies(
            body.dna,
            body.dna,
            unbonded_neighbors,
        )

        rna_dgs = dna2_energy.Debye(
            displacement_fn=self.displacement_fn, params=self.params.rna_config
        ).pairwise_energies(
            body.rna,
            body.rna,
            unbonded_neighbors,
        )

        drh_dgs = dna2_energy.Debye(
            displacement_fn=self.displacement_fn, params=self.params.drh_config
        ).pairwise_energies(
            body.dna,
            body.rna,
            unbonded_neighbors,
        )

        rdh_dgs = dna2_energy.Debye(
            displacement_fn=self.displacement_fn, params=self.params.drh_config
        ).pairwise_energies(
            body.rna,
            body.dna,
            unbonded_neighbors,
        )

        dgs = jnp.where(is_rna_bond, rna_dgs, jnp.where(is_drh_bond, drh_dgs, jnp.where(is_rdh_bond, rdh_dgs, dna_dgs)))
        dgs = jnp.where(mask, dgs, 0.0)

        return dgs.sum()
