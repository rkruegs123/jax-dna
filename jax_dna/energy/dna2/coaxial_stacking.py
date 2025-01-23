"""Coaxial stacking energy function for DNA1 model."""

import chex
import jax.numpy as jnp
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1.base_smoothing_functions as bsf
import jax_dna.energy.dna2.interactions as dna2_interactions
import jax_dna.energy.dna2.nucleotide as dna2_nucleotide
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class CoaxialStackingConfiguration(config.BaseConfiguration):
    """Configuration for the coaxial stacking energy function."""

    # independent parameters ===================================================
    dr_low_coax: float | None = None
    dr_high_coax: float | None = None
    k_coax: float | None = None
    dr0_coax: float | None = None
    dr_c_coax: float | None = None
    theta0_coax_4: float | None = None
    delta_theta_star_coax_4: float | None = None
    a_coax_4: float | None = None
    theta0_coax_1: float | None = None
    delta_theta_star_coax_1: float | None = None
    a_coax_1: float | None = None
    theta0_coax_5: float | None = None
    delta_theta_star_coax_5: float | None = None
    a_coax_5: float | None = None
    theta0_coax_6: float | None = None
    delta_theta_star_coax_6: float | None = None
    a_coax_6: float | None = None
    a_coax_1_f6: float | None = None
    b_coax_1_f6: float | None = None

    # dependent parameters =====================================================
    b_low_coax: float | None = None
    dr_c_low_coax: float | None = None
    b_high_coax: float | None = None
    dr_c_high_coax: float | None = None
    b_coax_4: float | None = None
    delta_theta_coax_4_c: float | None = None
    b_coax_1: float | None = None
    delta_theta_coax_1_c: float | None = None
    b_coax_5: float | None = None
    delta_theta_coax_5_c: float | None = None
    b_coax_6: float | None = None
    delta_theta_coax_6_c: float | None = None

    # override
    required_params: tuple[str] = (
        "dr_low_coax",
        "dr_high_coax",
        "k_coax",
        "dr0_coax",
        "dr_c_coax",
        "theta0_coax_4",
        "delta_theta_star_coax_4",
        "a_coax_4",
        "theta0_coax_1",
        "delta_theta_star_coax_1",
        "a_coax_1",
        "theta0_coax_5",
        "delta_theta_star_coax_5",
        "a_coax_5",
        "theta0_coax_6",
        "delta_theta_star_coax_6",
        "a_coax_6",
        "a_coax_1_f6",
        "b_coax_1_f6"
    )

    @override
    def init_params(self) -> "CoaxialStackingConfiguration":
        # reference to f2(dr_coax)
        b_low_coax, dr_c_low_coax, b_high_coax, dr_c_high_coax = bsf.get_f2_smoothing_params(
            self.dr0_coax,
            self.dr_c_coax,
            self.dr_low_coax,
            self.dr_high_coax,
        )

        # reference to f4(theta_4)
        b_coax_4, delta_theta_coax_4_c = bsf.get_f4_smoothing_params(
            self.a_coax_4,
            self.theta0_coax_4,
            self.delta_theta_star_coax_4,
        )

        # reference to f4(theta_1)
        b_coax_1, delta_theta_coax_1_c = bsf.get_f4_smoothing_params(
            self.a_coax_1,
            self.theta0_coax_1,
            self.delta_theta_star_coax_1,
        )

        # reference to f4(theta_5) + f4(pi - theta_5)
        b_coax_5, delta_theta_coax_5_c = bsf.get_f4_smoothing_params(
            self.a_coax_5,
            self.theta0_coax_5,
            self.delta_theta_star_coax_5,
        )

        # reference to f4(theta_6) + f4(pi - theta_6)
        b_coax_6, delta_theta_coax_6_c = bsf.get_f4_smoothing_params(
            self.a_coax_6,
            self.theta0_coax_6,
            self.delta_theta_star_coax_6,
        )


        return self.replace(
            b_low_coax=b_low_coax,
            dr_c_low_coax=dr_c_low_coax,
            b_high_coax=b_high_coax,
            dr_c_high_coax=dr_c_high_coax,
            b_coax_4=b_coax_4,
            delta_theta_coax_4_c=delta_theta_coax_4_c,
            b_coax_1=b_coax_1,
            delta_theta_coax_1_c=delta_theta_coax_1_c,
            b_coax_5=b_coax_5,
            delta_theta_coax_5_c=delta_theta_coax_5_c,
            b_coax_6=b_coax_6,
            delta_theta_coax_6_c=delta_theta_coax_6_c,
        )


@chex.dataclass(frozen=True)
class CoaxialStacking(je_base.BaseEnergyFunction):
    """Coaxial stacking energy function for DNA1 model."""

    params: CoaxialStackingConfiguration

    def pairwise_energies(
        self,
        body_i: dna2_nucleotide.Nucleotide,
        body_j: dna2_nucleotide.Nucleotide,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Arr_Unbonded_Neighbors:
        """Computes the coaxial stacking energy for each unbonded pair."""
        op_i = unbonded_neighbors[0]
        op_j = unbonded_neighbors[1]
        mask = jnp.array(op_i < body_i.center.shape[0], dtype=jnp.float32)

        dr_stack_op = self.displacement_mapped(body_j.stack_sites[op_j], body_i.stack_sites[op_i])
        dr_stack_norm_op = dr_stack_op / jnp.linalg.norm(dr_stack_op, axis=1, keepdims=True)

        theta4_op = jnp.arccos(
            jd_math.clamp(jnp.einsum("ij, ij->i", body_i.base_normals[op_i], body_j.base_normals[op_j]))
        )
        theta1_op = jnp.arccos(
            jd_math.clamp(jnp.einsum("ij, ij->i", -body_i.back_base_vectors[op_i], body_j.back_base_vectors[op_j]))
        )

        theta5_op = jnp.arccos(jd_math.clamp(jnp.einsum("ij, ij->i", body_i.base_normals[op_i], dr_stack_norm_op)))
        theta6_op = jnp.arccos(jd_math.clamp(jnp.einsum("ij, ij->i", -body_j.base_normals[op_j], dr_stack_norm_op)))

        cx_stack_dg = dna2_interactions.coaxial_stacking(
            dr_stack_op,
            theta4_op,
            theta1_op,
            theta5_op,
            theta6_op,
            self.params.dr_low_coax,
            self.params.dr_high_coax,
            self.params.dr_c_low_coax,
            self.params.dr_c_high_coax,
            self.params.k_coax,
            self.params.dr0_coax,
            self.params.dr_c_coax,
            self.params.b_low_coax,
            self.params.b_high_coax,
            self.params.theta0_coax_4,
            self.params.delta_theta_star_coax_4,
            self.params.delta_theta_coax_4_c,
            self.params.a_coax_4,
            self.params.b_coax_4,
            self.params.theta0_coax_1,
            self.params.delta_theta_star_coax_1,
            self.params.delta_theta_coax_1_c,
            self.params.a_coax_1,
            self.params.b_coax_1,
            self.params.a_coax_1_f6,
            self.params.b_coax_1_f6,
            self.params.theta0_coax_5,
            self.params.delta_theta_star_coax_5,
            self.params.delta_theta_coax_5_c,
            self.params.a_coax_5,
            self.params.b_coax_5,
            self.params.theta0_coax_6,
            self.params.delta_theta_star_coax_6,
            self.params.delta_theta_coax_6_c,
            self.params.a_coax_6,
            self.params.b_coax_6,
        )

        return jnp.where(mask, cx_stack_dg, 0.0)

    @override
    def __call__(
        self,
        body: dna2_nucleotide.Nucleotide,
        seq: typ.Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Scalar:
        dgs = self.pairwise_energies(body, body, unbonded_neighbors)
        return dgs.sum()
