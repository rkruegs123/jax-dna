from typing import Callable

import jax.numpy as jnp
import jax_md

import jax_dna.energy.base as je_base
import jax_dna.energy.dna1.defaults as dna1_defaults
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.energy.utils as je_utils
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as typ


class Stacking_DG(je_base.BaseEnergyFunction):
    def __init__(self, displacement_fn: Callable, params: dict[str, float], opt_params: dict[str, float]):
        super().__init__(displacement_fn, dna1_defaults.STACKING_DG | params, opt_params)

    def __call__(
        self,
        body: jax_md.rigid_body.RigidBody,
        seq: jnp.ndarray,
        bonded_neghbors: typ.Arr_Bonded_Neighbors,
        unbonded_neghbors: typ.Arr_Unbonded_Neighbors,
    ) -> typ.Scalar:
        com_to_backbone = self.get_param("com_to_backbone")
        com_to_stacking = self.get_param("com_to_stacking")
        nn_i = bonded_neghbors[:, 0]
        nn_j = bonded_neghbors[:, 1]
        Q = body.orientation

        back_base_vectors = je_utils.q_to_back_base(Q)  # space frame, normalized
        base_normals = je_utils.q_to_base_normal(Q)  # space frame, normalized
        cross_prods = je_utils.q_to_cross_prod(Q)  # space frame, normalized

        stack_sites = body.center + com_to_stacking * back_base_vectors
        back_sites = body.center + com_to_backbone * back_base_vectors

        ## Fene variables
        dr_back_nn = self.displacement_mapped(back_sites[nn_i], back_sites[nn_j])  # N x N x 3
        r_back_nn = jnp.linalg.norm(dr_back_nn, axis=1)

        dr_stack_nn = self.displacement_mapped(stack_sites[nn_i], stack_sites[nn_j])
        r_stack_nn = jnp.linalg.norm(dr_stack_nn, axis=1)
        theta4 = jnp.arccos(jd_math.clamp(jnp.einsum("ij, ij->i", base_normals[nn_i], base_normals[nn_j])))
        theta5 = jnp.pi - jnp.arccos(
            jd_math.clamp(jnp.einsum("ij, ij->i", dr_stack_nn, base_normals[nn_j]) / r_stack_nn)
        )
        theta6 = jnp.pi - jnp.arccos(
            jd_math.clamp(jnp.einsum("ij, ij->i", base_normals[nn_i], dr_stack_nn) / r_stack_nn)
        )
        cosphi1 = -jnp.einsum("ij, ij->i", cross_prods[nn_i], dr_back_nn) / r_back_nn
        cosphi2 = -jnp.einsum("ij, ij->i", cross_prods[nn_j], dr_back_nn) / r_back_nn

        v_stack = dna1_interactions.stacking(
            r_stack_nn,
            theta4,
            theta5,
            theta6,
            cosphi1,
            cosphi2,
            self.get_params(
                [
                    "dr_low_stack",
                    "dr_high_stack",
                    "eps_stack",
                    "a_stack",
                    "dr0_stack",
                    "dr_c_stack",
                    "dr_c_low_stack",
                    "dr_c_high_stack",
                    "b_low_stack",
                    "b_high_stack",
                    "theta0_stack_4",
                    "delta_theta_star_stack_4",
                    "a_stack_4",
                    "delta_theta_stack_4_c",
                    "b_stack_4",
                    "theta0_stack_5",
                    "delta_theta_star_stack_5",
                    "a_stack_5",
                    "delta_theta_stack_5_c",
                    "b_stack_5",
                    "theta0_stack_6",
                    "delta_theta_star_stack_6",
                    "a_stack_6",
                    "delta_theta_stack_6_c",
                    "b_stack_6",
                    "neg_cos_phi1_star_stack",
                    "a_stack_1",
                    "neg_cos_phi1_c_stack",
                    "b_neg_cos_phi1_stack",
                    "neg_cos_phi2_star_stack",
                    "a_stack_2",
                    "neg_cos_phi2_c_stack",
                    "b_neg_cos_phi2_stack",
                ]
            ),
        )

        stack_probs = je_utils.get_pair_probs(seq, nn_i, nn_j)
        stack_weights = jnp.dot(stack_probs, self.get_param("ss_stack_weights_flat"))

        stack_dg = jnp.dot(stack_weights, v_stack)

        return stack_dg


class CrossStacking_DG(je_base.BaseEnergyFunction):
    def __init__(self, displacement_fn: Callable, params: dict[str, float], opt_params: dict[str, float]):
        super().__init__(displacement_fn, dna1_defaults.CROSS_STACKING_DG | params, opt_params)

    def __call__(
        self,
        body: jax_md.rigid_body.RigidBody,
        seq: jnp.ndarray,
        bonded_neighbors: typ.Arr_Bonded_Neighbors,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors,
    ) -> typ.Scalar:
        com_to_hb = self.get_param("com_to_hb")
        op_i = unbonded_neighbors[0]
        op_j = unbonded_neighbors[1]

        Q = body.orientation
        back_base_vectors = je_utils.q_to_back_base(Q)  # space frame, normalized
        base_normals = je_utils.q_to_base_normal(Q)  # space frame, normalized

        mask = jnp.array(op_i < body.center.shape[0], dtype=jnp.float32)

        base_sites = body.center + com_to_hb * back_base_vectors

        dr_base_op = self.displacement_mapped(base_sites[op_j], base_sites[op_i])  # Note the flip here
        r_base_op = jnp.linalg.norm(dr_base_op, axis=1)

        theta1_op = jnp.arccos(jd_math.clamp(jd_math.mult(-back_base_vectors[op_i], back_base_vectors[op_j])))
        theta2_op = jnp.arccos(jd_math.clamp(jd_math.mult(-back_base_vectors[op_j], dr_base_op) / r_base_op))
        theta3_op = jnp.arccos(jd_math.clamp(jd_math.mult(back_base_vectors[op_i], dr_base_op) / r_base_op))
        theta4_op = jnp.arccos(jd_math.clamp(jd_math.mult(base_normals[op_i], base_normals[op_j])))
        # note: are these swapped in Lorenzo's code?
        theta7_op = jnp.arccos(jd_math.clamp(jd_math.mult(-base_normals[op_j], dr_base_op) / r_base_op))
        theta8_op = jnp.pi - jnp.arccos(jd_math.clamp(jd_math.mult(base_normals[op_i], dr_base_op) / r_base_op))

        cr_stack_dg = dna1_interactions.cross_stacking(
            r_base_op,
            theta1_op,
            theta2_op,
            theta3_op,
            theta4_op,
            theta7_op,
            theta8_op,
            **self.get_params(
                [
                    "dr_low_cross",
                    "dr_high_cross",
                    "dr_c_low_cross",
                    "dr_c_high_cross",
                    "k_cross",
                    "r0_cross",
                    "dr_c_cross",
                    "b_low_cross",
                    "b_high_cross",
                    "theta0_cross_1",
                    "delta_theta_star_cross_1",
                    "delta_theta_cross_1_c",
                    "a_cross_1",
                    "b_cross_1",
                    "theta0_cross_2",
                    "delta_theta_star_cross_2",
                    "delta_theta_cross_2_c",
                    "a_cross_2",
                    "b_cross_2",
                    "theta0_cross_3",
                    "delta_theta_star_cross_3",
                    "delta_theta_cross_3_c",
                    "a_cross_3",
                    "b_cross_3",
                    "theta0_cross_4",
                    "delta_theta_star_cross_4",
                    "delta_theta_cross_4_c",
                    "a_cross_4",
                    "b_cross_4",
                    "theta0_cross_7",
                    "delta_theta_star_cross_7",
                    "delta_theta_cross_7_c",
                    "a_cross_7",
                    "b_cross_7",
                    "theta0_cross_8",
                    "delta_theta_star_cross_8",
                    "delta_theta_cross_8_c",
                    "a_cross_8",
                    "b_cross_8",
                ]
            ),
        )
        # cr_stack_dg = jnp.where(mask, cr_stack_dg, 0.0).sum() # Mask for neighbors
        cr_stack_dg = (mask * cr_stack_dg).sum()

        return cr_stack_dg


class CoaxialStacking_DG(je_base.BaseEnergyFunction):
    def __init__(self, displacement_fn: Callable, params: dict[str, float], opt_params: dict[str, float]):
        super().__init__(displacement_fn, dna1_defaults.COAXIAL_STACKING_DG | params, opt_params)

    def __call__(
        self,
        body: jax_md.rigid_body.RigidBody,
        seq: jnp.ndarray,
        bonded_neighbors: typ.Arr_Bonded_Neighbors,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors,
    ) -> typ.Scalar:
        com_to_stacking = self.get_param("com_to_stacking")
        com_to_backbone = self.get_param("com_to_backbone")
        op_i = unbonded_neighbors[0]
        op_j = unbonded_neighbors[1]
        mask = jnp.array(op_i < body.center.shape[0], dtype=jnp.float32)

        Q = body.orientation
        back_base_vectors = je_utils.q_to_back_base(Q)  # space frame, normalized
        base_normals = je_utils.q_to_base_normal(Q)  # space frame, normalized

        back_sites = body.center + com_to_backbone * back_base_vectors
        stack_sites = body.center + com_to_stacking * back_base_vectors

        dr_stack_op = self.displacement_mapped(stack_sites[op_j], stack_sites[op_i])  # note: reversed
        dr_stack_norm_op = dr_stack_op / jnp.linalg.norm(dr_stack_op, axis=1, keepdims=True)
        dr_backbone_op = self.displacement_mapped(back_sites[op_j], back_sites[op_i])  # Note the flip here
        dr_backbone_norm_op = dr_backbone_op / jnp.linalg.norm(dr_backbone_op, axis=1, keepdims=True)

        theta4_op = jnp.arccos(jd_math.clamp(jnp.einsum("ij, ij->i", base_normals[op_i], base_normals[op_j])))
        theta1_op = jnp.arccos(
            jd_math.clamp(jnp.einsum("ij, ij->i", -back_base_vectors[op_i], back_base_vectors[op_j]))
        )

        theta5_op = jnp.arccos(jd_math.clamp(jnp.einsum("ij, ij->i", base_normals[op_i], dr_stack_norm_op)))
        theta6_op = jnp.arccos(jd_math.clamp(jnp.einsum("ij, ij->i", -base_normals[op_j], dr_stack_norm_op)))
        cosphi3_op = jnp.einsum("ij, ij->i", dr_stack_norm_op, jnp.cross(dr_backbone_norm_op, back_base_vectors[op_j]))
        cosphi4_op = jnp.einsum("ij, ij->i", dr_stack_norm_op, jnp.cross(dr_backbone_norm_op, back_base_vectors[op_i]))

        cx_stack_dg = dna1_interactions.coaxial_stacking(
            dr_stack_op,
            theta4_op,
            theta1_op,
            theta5_op,
            theta6_op,
            cosphi3_op,
            cosphi4_op,
            **self.params["coaxial_stacking"],
        )
        # cx_stack_dg = jnp.where(mask, cx_stack_dg, 0.0).sum() # Mask for neighbors
        cx_stack_dg = (mask * cx_stack_dg).sum()

        return cx_stack_dg
