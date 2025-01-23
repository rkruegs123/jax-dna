"""Stacking energy function for DNA1 model."""

import dataclasses as dc

import chex
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.energy.dna2 as dna2_energy
import jax_dna.energy.na1.nucleotide as na1_nucleotide
import jax_dna.energy.na1.utils as je_utils
import jax_dna.energy.rna2 as rna2_energy
import jax_dna.utils.types as typ
from jax_dna.energy.dna1.stacking import STACK_WEIGHTS_SA


@chex.dataclass(frozen=True)
class StackingConfiguration(config.BaseConfiguration):
    """Configuration for the stacking energy function."""

    # independent parameters
    nt_type: typ.Arr_Nucleotide | None = None
    kt: float | None = None
    ## DNA2-specific
    dna_eps_stack_base: float | None = None
    dna_eps_stack_kt_coeff: float | None = None
    dna_dr_low_stack: float | None = None
    dna_dr_high_stack: float | None = None
    dna_a_stack: float | None = None
    dna_dr0_stack: float | None = None
    dna_dr_c_stack: float | None = None
    dna_theta0_stack_4: float | None = None
    dna_delta_theta_star_stack_4: float | None = None
    dna_a_stack_4: float | None = None
    dna_theta0_stack_5: float | None = None
    dna_delta_theta_star_stack_5: float | None = None
    dna_a_stack_5: float | None = None
    dna_theta0_stack_6: float | None = None
    dna_delta_theta_star_stack_6: float | None = None
    dna_a_stack_6: float | None = None
    dna_neg_cos_phi1_star_stack: float | None = None
    dna_a_stack_1: float | None = None
    dna_neg_cos_phi2_star_stack: float | None = None
    dna_a_stack_2: float | None = None
    dna_ss_stack_weights: np.ndarray | None = dc.field(default_factory=lambda: STACK_WEIGHTS_SA)
    ## RNA2-specific
    rna_eps_stack_base: float | None = None
    rna_eps_stack_kt_coeff: float | None = None
    rna_dr_low_stack: float | None = None
    rna_dr_high_stack: float | None = None
    rna_a_stack: float | None = None
    rna_dr0_stack: float | None = None
    rna_dr_c_stack: float | None = None
    rna_theta0_stack_5: float | None = None
    rna_delta_theta_star_stack_5: float | None = None
    rna_a_stack_5: float | None = None
    rna_theta0_stack_6: float | None = None
    rna_delta_theta_star_stack_6: float | None = None
    rna_a_stack_6: float | None = None
    rna_theta0_stack_9: float | None = None
    rna_delta_theta_star_stack_9: float | None = None
    rna_a_stack_9: float | None = None
    rna_theta0_stack_10: float | None = None
    rna_delta_theta_star_stack_10: float | None = None
    rna_a_stack_10: float | None = None
    rna_neg_cos_phi1_star_stack: float | None = None
    rna_a_stack_1: float | None = None
    rna_neg_cos_phi2_star_stack: float | None = None
    rna_a_stack_2: float | None = None
    rna_ss_stack_weights: np.ndarray | None = dc.field(default_factory=lambda: STACK_WEIGHTS_SA)

    # dependent parameters
    dna_config: dna1_energy.StackingConfiguration | None = None
    rna_config: rna2_energy.StackingConfiguration | None = None

    required_params: tuple[str] = (
        # Type-independent
        "nt_type",
        "kt",
        # DNA2-specific
        "dna_eps_stack_base",
        "dna_eps_stack_kt_coeff",
        "dna_dr_low_stack",
        "dna_dr_high_stack",
        "dna_a_stack",
        "dna_dr0_stack",
        "dna_dr_c_stack",
        "dna_theta0_stack_4",
        "dna_delta_theta_star_stack_4",
        "dna_a_stack_4",
        "dna_theta0_stack_5",
        "dna_delta_theta_star_stack_5",
        "dna_a_stack_5",
        "dna_theta0_stack_6",
        "dna_delta_theta_star_stack_6",
        "dna_a_stack_6",
        "dna_neg_cos_phi1_star_stack",
        "dna_a_stack_1",
        "dna_neg_cos_phi2_star_stack",
        "dna_a_stack_2",
        "dna_ss_stack_weights",
        # RNA2-specific
        "rna_eps_stack_base",
        "rna_eps_stack_kt_coeff",
        "rna_dr_low_stack",
        "rna_dr_high_stack",
        "rna_a_stack",
        "rna_dr0_stack",
        "rna_dr_c_stack",
        "rna_theta0_stack_5",
        "rna_delta_theta_star_stack_5",
        "rna_a_stack_5",
        "rna_theta0_stack_6",
        "rna_delta_theta_star_stack_6",
        "rna_a_stack_6",
        "rna_theta0_stack_9",
        "rna_delta_theta_star_stack_9",
        "rna_a_stack_9",
        "rna_theta0_stack_10",
        "rna_delta_theta_star_stack_10",
        "rna_a_stack_10",
        "rna_neg_cos_phi1_star_stack",
        "rna_a_stack_1",
        "rna_neg_cos_phi2_star_stack",
        "rna_a_stack_2",
        "rna_ss_stack_weights",
    )

    @override
    def init_params(self) -> "StackingConfiguration":

        dna_config = dna1_energy.StackingConfiguration(
            eps_stack_base=self.dna_eps_stack_base,
            eps_stack_kt_coeff=self.dna_eps_stack_kt_coeff,
            dr_low_stack=self.dna_dr_low_stack,
            dr_high_stack=self.dna_dr_high_stack,
            a_stack=self.dna_a_stack,
            dr0_stack=self.dna_dr0_stack,
            dr_c_stack=self.dna_dr_c_stack,
            theta0_stack_4=self.dna_theta0_stack_4,
            delta_theta_star_stack_4=self.dna_delta_theta_star_stack_4,
            a_stack_4=self.dna_a_stack_4,
            theta0_stack_5=self.dna_theta0_stack_5,
            delta_theta_star_stack_5=self.dna_delta_theta_star_stack_5,
            a_stack_5=self.dna_a_stack_5,
            theta0_stack_6=self.dna_theta0_stack_6,
            delta_theta_star_stack_6=self.dna_delta_theta_star_stack_6,
            a_stack_6=self.dna_a_stack_6,
            neg_cos_phi1_star_stack=self.dna_neg_cos_phi1_star_stack,
            a_stack_1=self.dna_a_stack_1,
            neg_cos_phi2_star_stack=self.dna_neg_cos_phi2_star_stack,
            a_stack_2=self.dna_a_stack_2,
            kt=self.kt,
            ss_stack_weights=self.dna_ss_stack_weights,
        ).init_params()

        rna_config = rna2_energy.StackingConfiguration(
            eps_stack_base=self.rna_eps_stack_base,
            eps_stack_kt_coeff=self.rna_eps_stack_kt_coeff,
            dr_low_stack=self.rna_dr_low_stack,
            dr_high_stack=self.rna_dr_high_stack,
            a_stack=self.rna_a_stack,
            dr0_stack=self.rna_dr0_stack,
            dr_c_stack=self.rna_dr_c_stack,
            theta0_stack_5=self.rna_theta0_stack_5,
            delta_theta_star_stack_5=self.rna_delta_theta_star_stack_5,
            a_stack_5=self.rna_a_stack_5,
            theta0_stack_6=self.rna_theta0_stack_6,
            delta_theta_star_stack_6=self.rna_delta_theta_star_stack_6,
            a_stack_6=self.rna_a_stack_6,
            theta0_stack_9=self.rna_theta0_stack_9,
            delta_theta_star_stack_9=self.rna_delta_theta_star_stack_9,
            a_stack_9=self.rna_a_stack_9,
            theta0_stack_10=self.rna_theta0_stack_10,
            delta_theta_star_stack_10=self.rna_delta_theta_star_stack_10,
            a_stack_10=self.rna_a_stack_10,
            neg_cos_phi1_star_stack=self.rna_neg_cos_phi1_star_stack,
            a_stack_1=self.rna_a_stack_1,
            neg_cos_phi2_star_stack=self.rna_neg_cos_phi2_star_stack,
            a_stack_2=self.rna_a_stack_2,
            kt=self.kt,
            ss_stack_weights=self.rna_ss_stack_weights,
        ).init_params()

        return self.replace(
            dna_config=dna_config,
            rna_config=rna_config,
        )


@chex.dataclass(frozen=True)
class Stacking(je_base.BaseEnergyFunction):
    """Stacking energy function for DNA1 model."""

    params: StackingConfiguration

    @override
    def __call__(
        self,
        body: na1_nucleotide.HybridNucleotide,
        seq: typ.Discrete_Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Scalar:


        nn_i = bonded_neighbors[:, 0]
        nn_j = bonded_neighbors[:, 1]

        is_rna_bond = jax.vmap(je_utils.is_rna_pair, (0, 0, None))(nn_i, nn_j, self.params.nt_type)

        dna_dgs = dna2_energy.Stacking(
            displacement_fn=self.displacement_fn,
            params=self.params.dna_config
        ).pairwise_energies(
            body.dna,
            seq,
            bonded_neighbors,
        )

        rna_dgs = rna2_energy.Stacking(
            displacement_fn=self.displacement_fn,
            params=self.params.rna_config
        ).pairwise_energies(
            body.rna,
            seq,
            bonded_neighbors,
        )

        # Select based on bond type
        dgs = jnp.where(is_rna_bond, rna_dgs, dna_dgs)
        return dgs.sum()
