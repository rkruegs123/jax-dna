import chex
import jax.numpy as jnp

import jax_dna.energy.base as je_base
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.energy.dna1.nucleotide as dna1_nucleotide
import jax_dna.energy.utils as je_utils
import jax_dna.input.dna1.unbonded as config
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class ExcludedVolume(je_base.BaseEnergyFunction):
    params: config.ExcludedVolumeConfiguration

    def __call__(
        self,
        body: dna1_nucleotide.Nucleotide,
        seq: jnp.ndarray,
        bonded_neghbors: typ.Arr_Bonded_Neighbors,
        unbonded_neghbors: typ.Arr_Unbonded_Neighbors,
    ) -> typ.Scalar:
        op_i = unbonded_neghbors[0]
        op_j = unbonded_neghbors[1]

        mask = jnp.array(op_i < body.center.shape[0], dtype=jnp.float32)

        dr_base_op = self.displacement_mapped(body.base_sites[op_j], body.base_sites[op_i])  # Note the flip here
        dr_backbone_op = self.displacement_mapped(body.back_sites[op_j], body.back_sites[op_i])  # Note the flip here
        dr_back_base_op = self.displacement_mapped(
            body.back_sites[op_i], body.base_sites[op_j]
        )  # Note: didn't flip this one (and others) because no need, but should look into at some point
        dr_base_back_op = self.displacement_mapped(body.base_sites[op_i], body.back_sites[op_j])

        exc_vol_unbonded_dg = dna1_interactions.exc_vol_unbonded(
            dr_base_op,
            dr_backbone_op,
            dr_back_base_op,
            dr_base_back_op,
            self.params.eps_exc,
            self.params.dr_star_base,
            self.params.sigma_base,
            self.params.b_base,
            self.params.dr_c_base,
            self.params.dr_star_back_base,
            self.params.sigma_back_base,
            self.params.b_back_base,
            self.params.dr_c_back_base,
            self.params.dr_star_base_back,
            self.params.sigma_base_back,
            self.params.b_base_back,
            self.params.dr_c_base_back,
            self.params.dr_star_backbone,
            self.params.sigma_backbone,
            self.params.b_backbone,
            self.params.dr_c_backbone,
        )

        # used to be:
        # return jnp.where(mask, exc_vol_unbonded_dg, 0.0).sum()
        return (mask * exc_vol_unbonded_dg).sum()


@chex.dataclass(frozen=True)
class HydrogenBonding(je_base.BaseEnergyFunction):
    params: config.HydrogenBondingConfiguration

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

        hb_probs = je_utils.get_pair_probs(
            seq, op_i, op_j
        )  # get the probabilities of all possibile hydrogen bonds for all neighbors
        hb_weights = jnp.dot(hb_probs, self.params.ss_hb_weights.flatten())

        dr_base_op = self.displacement_mapped(body.base_sites[op_j], body.base_sites[op_i])  # Note the flip here
        r_base_op = jnp.linalg.norm(dr_base_op, axis=1)

        theta1_op = jnp.arccos(jd_math.clamp(jd_math.mult(-body.back_base_vectors[op_i], body.back_base_vectors[op_j])))
        theta2_op = jnp.arccos(jd_math.clamp(jd_math.mult(-body.back_base_vectors[op_j], dr_base_op) / r_base_op))
        theta3_op = jnp.arccos(jd_math.clamp(jd_math.mult(body.back_base_vectors[op_i], dr_base_op) / r_base_op))
        theta4_op = jnp.arccos(jd_math.clamp(jd_math.mult(body.base_normals[op_i], body.base_normals[op_j])))
        # note: are these swapped in Lorenzo's code?
        theta7_op = jnp.arccos(jd_math.clamp(jd_math.mult(-body.base_normals[op_j], dr_base_op) / r_base_op))
        theta8_op = jnp.pi - jnp.arccos(jd_math.clamp(jd_math.mult(body.base_normals[op_i], dr_base_op) / r_base_op))

        v_hb = dna1_interactions.hydrogen_bonding(
            dr_base_op,
            theta1_op,
            theta2_op,
            theta3_op,
            theta4_op,
            theta7_op,
            theta8_op,
            self.params.dr_low_hb,
            self.params.dr_high_hb,
            self.params.dr_c_low_hb,
            self.params.dr_c_high_hb,
            self.params.eps_hb,
            self.params.a_hb,
            self.params.dr0_hb,
            self.params.dr_c_hb,
            self.params.b_low_hb,
            self.params.b_high_hb,
            self.params.theta0_hb_1,
            self.params.delta_theta_star_hb_1,
            self.params.a_hb_1,
            self.params.delta_theta_hb_1_c,
            self.params.b_hb_1,
            self.params.theta0_hb_2,
            self.params.delta_theta_star_hb_2,
            self.params.a_hb_2,
            self.params.delta_theta_hb_2_c,
            self.params.b_hb_2,
            self.params.theta0_hb_3,
            self.params.delta_theta_star_hb_3,
            self.params.a_hb_3,
            self.params.delta_theta_hb_3_c,
            self.params.b_hb_3,
            self.params.theta0_hb_4,
            self.params.delta_theta_star_hb_4,
            self.params.a_hb_4,
            self.params.delta_theta_hb_4_c,
            self.params.b_hb_4,
            self.params.theta0_hb_7,
            self.params.delta_theta_star_hb_7,
            self.params.a_hb_7,
            self.params.delta_theta_hb_7_c,
            self.params.b_hb_7,
            self.params.theta0_hb_8,
            self.params.delta_theta_star_hb_8,
            self.params.a_hb_8,
            self.params.delta_theta_hb_8_c,
            self.params.b_hb_8,
        )

        # v_hb = jnp.where(mask, v_hb, 0.0) # Mask for neighbors
        v_hb = mask * v_hb

        hb_dg = jnp.dot(hb_weights, v_hb)

        return hb_dg


class CrossStacking(je_base.BaseEnergyFunction):
    params: config.CrossStackingConfiguration

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

        dr_base_op = self.displacement_mapped(body.base_sites[op_j], body.base_sites[op_i])  # Note the flip here
        r_base_op = jnp.linalg.norm(dr_base_op, axis=1)

        theta1_op = jnp.arccos(jd_math.clamp(jd_math.mult(-body.back_base_vectors[op_i], body.back_base_vectors[op_j])))
        theta2_op = jnp.arccos(jd_math.clamp(jd_math.mult(-body.back_base_vectors[op_j], dr_base_op) / r_base_op))
        theta3_op = jnp.arccos(jd_math.clamp(jd_math.mult(body.back_base_vectors[op_i], dr_base_op) / r_base_op))
        theta4_op = jnp.arccos(jd_math.clamp(jd_math.mult(body.base_normals[op_i], body.base_normals[op_j])))
        # note: are these swapped in Lorenzo's code?
        theta7_op = jnp.arccos(jd_math.clamp(jd_math.mult(-body.base_normals[op_j], dr_base_op) / r_base_op))
        theta8_op = jnp.pi - jnp.arccos(jd_math.clamp(jd_math.mult(body.base_normals[op_i], dr_base_op) / r_base_op))

        cr_stack_dg = dna1_interactions.cross_stacking(
            r_base_op,
            theta1_op,
            theta2_op,
            theta3_op,
            theta4_op,
            theta7_op,
            theta8_op,
            self.params.dr_low_cross,
            self.params.dr_high_cross,
            self.params.dr_c_low_cross,
            self.params.dr_c_high_cross,
            self.params.k_cross,
            self.params.r0_cross,
            self.params.dr_c_cross,
            self.params.b_low_cross,
            self.params.b_high_cross,
            self.params.theta0_cross_1,
            self.params.delta_theta_star_cross_1,
            self.params.delta_theta_cross_1_c,
            self.params.a_cross_1,
            self.params.b_cross_1,
            self.params.theta0_cross_2,
            self.params.delta_theta_star_cross_2,
            self.params.delta_theta_cross_2_c,
            self.params.a_cross_2,
            self.params.b_cross_2,
            self.params.theta0_cross_3,
            self.params.delta_theta_star_cross_3,
            self.params.delta_theta_cross_3_c,
            self.params.a_cross_3,
            self.params.b_cross_3,
            self.params.theta0_cross_4,
            self.params.delta_theta_star_cross_4,
            self.params.delta_theta_cross_4_c,
            self.params.a_cross_4,
            self.params.b_cross_4,
            self.params.theta0_cross_7,
            self.params.delta_theta_star_cross_7,
            self.params.delta_theta_cross_7_c,
            self.params.a_cross_7,
            self.params.b_cross_7,
            self.params.theta0_cross_8,
            self.params.delta_theta_star_cross_8,
            self.params.delta_theta_cross_8_c,
            self.params.a_cross_8,
            self.params.b_cross_8,
        )
        # cr_stack_dg = jnp.where(mask, cr_stack_dg, 0.0).sum() # Mask for neighbors
        cr_stack_dg = (mask * cr_stack_dg).sum()

        return cr_stack_dg


class CoaxialStacking(je_base.BaseEnergyFunction):
    params: config.CoaxialStackingConfiguration

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
