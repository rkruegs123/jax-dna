"""Stacking energy function for RNA2 model."""

import dataclasses as dc

import chex
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1.base_smoothing_functions as bsf
import jax_dna.energy.rna2.interactions as rna2_interactions
import jax_dna.energy.rna2.nucleotide as rna2_nucleotide
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as typ
from jax_dna.energy.dna1.stacking import STACK_WEIGHTS_SA


@chex.dataclass(frozen=True)
class StackingConfiguration(config.BaseConfiguration):
    """Configuration for the stacking energy function."""

    # independent parameters
    eps_stack_base: float | None = None
    eps_stack_kt_coeff: float | None = None
    dr_low_stack: float | None = None
    dr_high_stack: float | None = None
    a_stack: float | None = None
    dr0_stack: float | None = None
    dr_c_stack: float | None = None

    theta0_stack_5: float | None = None
    delta_theta_star_stack_5: float | None = None
    a_stack_5: float | None = None

    theta0_stack_6: float | None = None
    delta_theta_star_stack_6: float | None = None
    a_stack_6: float | None = None

    theta0_stack_9: float | None = None
    delta_theta_star_stack_9: float | None = None
    a_stack_9: float | None = None

    theta0_stack_10: float | None = None
    delta_theta_star_stack_10: float | None = None
    a_stack_10: float | None = None

    neg_cos_phi1_star_stack: float | None = None
    a_stack_1: float | None = None
    neg_cos_phi2_star_stack: float | None = None
    a_stack_2: float | None = None

    kt: float | None = None
    ss_stack_weights: np.ndarray | None = dc.field(default_factory=lambda: STACK_WEIGHTS_SA)

    # dependent parameters
    b_low_stack: float | None = None
    dr_c_low_stack: float | None = None
    b_high_stack: float | None = None
    dr_c_high_stack: float | None = None
    b_stack_5: float | None = None
    delta_theta_stack_5_c: float | None = None
    b_stack_6: float | None = None
    delta_theta_stack_6_c: float | None = None
    b_stack_9: float | None = None
    delta_theta_stack_9_c: float | None = None
    b_stack_10: float | None = None
    delta_theta_stack_10_c: float | None = None
    b_neg_cos_phi1_stack: float | None = None
    neg_cos_phi1_c_stack: float | None = None
    b_neg_cos_phi2_stack: float | None = None
    neg_cos_phi2_c_stack: float | None = None
    eps_stack: float | None = None

    required_params: tuple[str] = (
        "eps_stack_base",
        "eps_stack_kt_coeff",
        "dr_low_stack",
        "dr_high_stack",
        "a_stack",
        "dr0_stack",
        "dr_c_stack",
        "theta0_stack_5",
        "delta_theta_star_stack_5",
        "a_stack_5",
        "theta0_stack_6",
        "delta_theta_star_stack_6",
        "a_stack_6",
        "theta0_stack_9",
        "delta_theta_star_stack_9",
        "a_stack_9",
        "theta0_stack_10",
        "delta_theta_star_stack_10",
        "a_stack_10",
        "neg_cos_phi1_star_stack",
        "a_stack_1",
        "neg_cos_phi2_star_stack",
        "a_stack_2",
        "kt",
        "ss_stack_weights",
    )

    @override
    def init_params(self) -> "StackingConfiguration":
        eps_stack = self.eps_stack_base + self.eps_stack_kt_coeff * self.kt

        b_low_stack, dr_c_low_stack, b_high_stack, dr_c_high_stack = bsf.get_f1_smoothing_params(
            self.dr0_stack,
            self.a_stack,
            self.dr_c_stack,
            self.dr_low_stack,
            self.dr_high_stack,
        )

        b_stack_5, delta_theta_stack_5_c = bsf.get_f4_smoothing_params(
            self.a_stack_5,
            self.theta0_stack_5,
            self.delta_theta_star_stack_5,
        )

        b_stack_6, delta_theta_stack_6_c = bsf.get_f4_smoothing_params(
            self.a_stack_6,
            self.theta0_stack_6,
            self.delta_theta_star_stack_6,
        )

        b_stack_9, delta_theta_stack_9_c = bsf.get_f4_smoothing_params(
            self.a_stack_9,
            self.theta0_stack_9,
            self.delta_theta_star_stack_9,
        )

        b_stack_10, delta_theta_stack_10_c = bsf.get_f4_smoothing_params(
            self.a_stack_10,
            self.theta0_stack_10,
            self.delta_theta_star_stack_10,
        )

        b_neg_cos_phi1_stack, neg_cos_phi1_c_stack = bsf.get_f5_smoothing_params(
            self.a_stack_1,
            self.neg_cos_phi1_star_stack,
        )

        b_neg_cos_phi2_stack, neg_cos_phi2_c_stack = bsf.get_f5_smoothing_params(
            self.a_stack_2,
            self.neg_cos_phi2_star_stack,
        )

        return self.replace(
            b_low_stack=b_low_stack,
            dr_c_low_stack=dr_c_low_stack,
            b_high_stack=b_high_stack,
            dr_c_high_stack=dr_c_high_stack,
            b_stack_5=b_stack_5,
            delta_theta_stack_5_c=delta_theta_stack_5_c,
            b_stack_6=b_stack_6,
            delta_theta_stack_6_c=delta_theta_stack_6_c,
            b_stack_9=b_stack_9,
            delta_theta_stack_9_c=delta_theta_stack_9_c,
            b_stack_10=b_stack_10,
            delta_theta_stack_10_c=delta_theta_stack_10_c,
            b_neg_cos_phi1_stack=b_neg_cos_phi1_stack,
            neg_cos_phi1_c_stack=neg_cos_phi1_c_stack,
            b_neg_cos_phi2_stack=b_neg_cos_phi2_stack,
            neg_cos_phi2_c_stack=neg_cos_phi2_c_stack,
            eps_stack=eps_stack,
        )


@chex.dataclass(frozen=True)
class Stacking(je_base.BaseEnergyFunction):
    """Stacking energy function for DNA1 model."""

    params: StackingConfiguration

    def compute_v_stack(
        self,
        body: rna2_nucleotide.Nucleotide,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
    ) -> typ.Arr_Bonded_Neighbors:
        """Computes the sequence-independent energy for each bonded pair."""
        nn_i = bonded_neighbors[:, 0]
        nn_j = bonded_neighbors[:, 1]

        dr_stack_nn = self.displacement_mapped(body.stack5_sites[nn_i], body.stack3_sites[nn_j])
        r_stack_nn = jnp.linalg.norm(dr_stack_nn, axis=1)

        theta5 = jnp.pi - jnp.arccos(
            jd_math.clamp(jnp.einsum("ij, ij->i", dr_stack_nn, body.base_normals[nn_j]) / r_stack_nn)
        )
        theta6 = jnp.pi - jnp.arccos(
            jd_math.clamp(jnp.einsum("ij, ij->i", body.base_normals[nn_i], dr_stack_nn) / r_stack_nn)
        )

        dr_back_nn = self.displacement_mapped(body.back_sites[nn_i], body.back_sites[nn_j])  # N x N x 3
        r_back_nn = jnp.linalg.norm(dr_back_nn, axis=1)

        theta9 = jnp.arccos(jd_math.clamp(jnp.einsum("ij, ij->i", -body.bb_p3_sites[nn_j], dr_back_nn) / r_back_nn))

        theta10 = jnp.arccos(jd_math.clamp(jnp.einsum("ij, ij->i", -body.bb_p5_sites[nn_i], dr_back_nn) / r_back_nn))

        cosphi1 = -jnp.einsum("ij, ij->i", body.cross_prods[nn_i], dr_back_nn) / r_back_nn
        cosphi2 = -jnp.einsum("ij, ij->i", body.cross_prods[nn_j], dr_back_nn) / r_back_nn

        return rna2_interactions.stacking(
            r_stack_nn,
            theta5,
            theta6,
            theta9,
            theta10,
            cosphi1,
            cosphi2,
            self.params.dr_low_stack,
            self.params.dr_high_stack,
            self.params.eps_stack,
            self.params.a_stack,
            self.params.dr0_stack,
            self.params.dr_c_stack,
            self.params.dr_c_low_stack,
            self.params.dr_c_high_stack,
            self.params.b_low_stack,
            self.params.b_high_stack,
            self.params.theta0_stack_5,
            self.params.delta_theta_star_stack_5,
            self.params.a_stack_5,
            self.params.delta_theta_stack_5_c,
            self.params.b_stack_5,
            self.params.theta0_stack_6,
            self.params.delta_theta_star_stack_6,
            self.params.a_stack_6,
            self.params.delta_theta_stack_6_c,
            self.params.b_stack_6,
            self.params.theta0_stack_9,
            self.params.delta_theta_star_stack_9,
            self.params.a_stack_9,
            self.params.delta_theta_stack_9_c,
            self.params.b_stack_9,
            self.params.theta0_stack_10,
            self.params.delta_theta_star_stack_10,
            self.params.a_stack_10,
            self.params.delta_theta_stack_10_c,
            self.params.b_stack_10,
            self.params.neg_cos_phi1_star_stack,
            self.params.a_stack_1,
            self.params.neg_cos_phi1_c_stack,
            self.params.b_neg_cos_phi1_stack,
            self.params.neg_cos_phi2_star_stack,
            self.params.a_stack_2,
            self.params.neg_cos_phi2_c_stack,
            self.params.b_neg_cos_phi2_stack,
        )

    def pairwise_energies(
        self,
        body: rna2_nucleotide.Nucleotide,
        seq: typ.Discrete_Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
    ) -> typ.Arr_Bonded_Neighbors:
        """Computes the stacking energy for each bonded pair."""
        # Compute sequence-independent energy for each bonded pair
        v_stack = self.compute_v_stack(body, bonded_neighbors)

        # Compute sequence-dependent weight for each bonded pair
        nn_i = bonded_neighbors[:, 0]
        nn_j = bonded_neighbors[:, 1]
        stack_weights = self.params.ss_stack_weights[seq[nn_i], seq[nn_j]]

        return jnp.multiply(stack_weights, v_stack)

    @override
    def __call__(
        self,
        body: rna2_nucleotide.Nucleotide,
        seq: typ.Discrete_Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Scalar:
        # Compute sequence-independent energy for each bonded pair
        v_stack = self.compute_v_stack(body, bonded_neighbors)

        # Compute sequence-dependent weight for each bonded pair
        nn_i = bonded_neighbors[:, 0]
        nn_j = bonded_neighbors[:, 1]
        stack_weights = self.params.ss_stack_weights[seq[nn_i], seq[nn_j]]

        # Return the weighted sum
        return jnp.dot(stack_weights, v_stack)
