import chex
import jax.numpy as jnp

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1.base_smoothing_functions as bsf
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.energy.dna1.nucleotide as dna1_nucleotide
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class CoaxialStackingConfiguration(config.BaseConfiguration):
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
    cos_phi3_star_coax: float | None = None
    a_coax_3p: float | None = None
    cos_phi4_star_coax: float | None = None
    a_coax_4p: float | None = None

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
    b_cos_phi3_coax: float | None = None
    cos_phi3_c_coax: float | None = None
    b_cos_phi4_coax: float | None = None
    cos_phi4_c_coax: float | None = None

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
        "cos_phi3_star_coax",
        "a_coax_3p",
        "cos_phi4_star_coax",
        "a_coax_4p",
    )

    def init_params(self) -> "CoaxialStackingConfiguration":
        ## f2(dr_coax)
        b_low_coax, dr_c_low_coax, b_high_coax, dr_c_high_coax = bsf.get_f2_smoothing_params(
            self.k_coax,
            self.dr0_coax,
            self.dr_c_coax,
            self.dr_low_coax,
            self.dr_high_coax,
        )

        ## f4(theta_4)
        b_coax_4, delta_theta_coax_4_c = bsf.get_f4_smoothing_params(
            self.a_coax_4,
            self.theta0_coax_4,
            self.delta_theta_star_coax_4,
        )

        ## f4(theta_1) + f4(2*pi - theta_1)
        b_coax_1, delta_theta_coax_1_c = bsf.get_f4_smoothing_params(
            self.a_coax_1,
            self.theta0_coax_1,
            self.delta_theta_star_coax_1,
        )

        ## f4(theta_5) + f4(pi - theta_5)
        b_coax_5, delta_theta_coax_5_c = bsf.get_f4_smoothing_params(
            self.a_coax_5,
            self.theta0_coax_5,
            self.delta_theta_star_coax_5,
        )

        ## f4(theta_6) + f4(pi - theta_6)
        b_coax_6, delta_theta_coax_6_c = bsf.get_f4_smoothing_params(
            self.a_coax_6,
            self.theta0_coax_6,
            self.delta_theta_star_coax_6,
        )

        ## f5(cos(phi3))
        b_cos_phi3_coax, cos_phi3_c_coax = bsf.get_f5_smoothing_params(
            self.a_coax_3p,
            self.cos_phi3_star_coax,
        )

        ## f5(cos(phi4))
        b_cos_phi4_coax, cos_phi4_c_coax = bsf.get_f5_smoothing_params(
            self.a_coax_4p,
            self.cos_phi4_star_coax,
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
            b_cos_phi3_coax=b_cos_phi3_coax,
            cos_phi3_c_coax=cos_phi3_c_coax,
            b_cos_phi4_coax=b_cos_phi4_coax,
            cos_phi4_c_coax=cos_phi4_c_coax,
        )


@chex.dataclass(frozen=True)
class CoaxialStacking(je_base.BaseEnergyFunction):
    params: CoaxialStackingConfiguration

    def __call__(
        self,
        body: dna1_nucleotide.Nucleotide,
        seq: jnp.ndarray,
        bonded_neighbors: typ.Arr_Bonded_Neighbors,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors,
    ) -> typ.Scalar:
        op_i = unbonded_neighbors[0]
        op_j = unbonded_neighbors[1]
        mask = jnp.array(op_i < body.center.shape[0], dtype=jnp.float32)

        dr_stack_op = self.displacement_mapped(body.stack_sites[op_j], body.stack_sites[op_i])  # note: reversed
        dr_stack_norm_op = dr_stack_op / jnp.linalg.norm(dr_stack_op, axis=1, keepdims=True)
        dr_backbone_op = self.displacement_mapped(body.back_sites[op_j], body.back_sites[op_i])  # Note the flip here
        dr_backbone_norm_op = dr_backbone_op / jnp.linalg.norm(dr_backbone_op, axis=1, keepdims=True)

        theta4_op = jnp.arccos(jd_math.clamp(jnp.einsum("ij, ij->i", body.base_normals[op_i], body.base_normals[op_j])))
        theta1_op = jnp.arccos(
            jd_math.clamp(jnp.einsum("ij, ij->i", -body.back_base_vectors[op_i], body.back_base_vectors[op_j]))
        )

        theta5_op = jnp.arccos(jd_math.clamp(jnp.einsum("ij, ij->i", body.base_normals[op_i], dr_stack_norm_op)))
        theta6_op = jnp.arccos(jd_math.clamp(jnp.einsum("ij, ij->i", -body.base_normals[op_j], dr_stack_norm_op)))
        cosphi3_op = jnp.einsum(
            "ij, ij->i", dr_stack_norm_op, jnp.cross(dr_backbone_norm_op, body.back_base_vectors[op_j])
        )
        cosphi4_op = jnp.einsum(
            "ij, ij->i", dr_stack_norm_op, jnp.cross(dr_backbone_norm_op, body.back_base_vectors[op_i])
        )

        cx_stack_dg = dna1_interactions.coaxial_stacking(
            dr_stack_op,
            theta4_op,
            theta1_op,
            theta5_op,
            theta6_op,
            cosphi3_op,
            cosphi4_op,
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
            self.params.cos_phi3_star_coax,
            self.params.cos_phi3_c_coax,
            self.params.a_coax_3p,
            self.params.b_cos_phi3_coax,
            self.params.cos_phi4_star_coax,
            self.params.cos_phi4_c_coax,
            self.params.a_coax_4p,
            self.params.b_cos_phi4_coax,
        )
        # cx_stack_dg = jnp.where(mask, cx_stack_dg, 0.0).sum() # Mask for neighbors
        cx_stack_dg = (mask * cx_stack_dg).sum()

        return cx_stack_dg
