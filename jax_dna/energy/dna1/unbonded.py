from typing import Callable

import jax.numpy as jnp

import jax_dna.energy.base as je_base
import jax_dna.energy.dna1.defaults as dna1_defaults
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.energy.dna1.nucleotide as dna1_nucleotide
import jax_dna.energy.utils as je_utils
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as typ


class ExcludedVolume(je_base.BaseEnergyFunction):
    def __init__(
        self,
        displacement_fn: Callable,
        params: dict[str, float] = {},
        opt_params: dict[str, float] = {},
    ):
        super().__init__(displacement_fn, dna1_defaults.UNBONDED_DG | params, opt_params)

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
            **self.get_params(
                [
                    "eps_exc",
                    "dr_star_base",
                    "sigma_base",
                    "b_base",
                    "dr_c_base",
                    "dr_star_back_base",
                    "sigma_back_base",
                    "b_back_base",
                    "dr_c_back_base",
                    "dr_star_base_back",
                    "sigma_base_back",
                    "b_base_back",
                    "dr_c_base_back",
                    "dr_star_backbone",
                    "sigma_backbone",
                    "b_backbone",
                    "dr_c_backbone",
                ]
            ),
        )

        # used to be:
        # return jnp.where(mask, exc_vol_unbonded_dg, 0.0).sum()
        return (mask * exc_vol_unbonded_dg).sum()


class HydrogenBonding(je_base.BaseEnergyFunction):
    def __init__(
        self,
        displacement_fn: Callable,
        params: dict[str, float] = {},
        opt_params: dict[str, float] = {},
    ):
        super().__init__(displacement_fn, dna1_defaults.HB_DG | params, opt_params)

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
        hb_weights = jnp.dot(hb_probs, self.get_param("hb_weights").flatten())

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
            **self.get_params(
                [
                    "dr_low_hb",
                    "dr_high_hb",
                    "dr_c_low_hb",
                    "dr_c_high_hb",
                    "eps_hb",
                    "a_hb",
                    "dr0_hb",
                    "dr_c_hb",
                    "b_low_hb",
                    "b_high_hb",
                    "theta0_hb_1",
                    "delta_theta_star_hb_1",
                    "a_hb_1",
                    "delta_theta_hb_1_c",
                    "b_hb_1",
                    "theta0_hb_2",
                    "delta_theta_star_hb_2",
                    "a_hb_2",
                    "delta_theta_hb_2_c",
                    "b_hb_2",
                    "theta0_hb_3",
                    "delta_theta_star_hb_3",
                    "a_hb_3",
                    "delta_theta_hb_3_c",
                    "b_hb_3",
                    "theta0_hb_4",
                    "delta_theta_star_hb_4",
                    "a_hb_4",
                    "delta_theta_hb_4_c",
                    "b_hb_4",
                    "theta0_hb_7",
                    "delta_theta_star_hb_7",
                    "a_hb_7",
                    "delta_theta_hb_7_c",
                    "b_hb_7",
                    "theta0_hb_8",
                    "delta_theta_star_hb_8",
                    "a_hb_8",
                    "delta_theta_hb_8_c",
                    "b_hb_8",
                ]
            ),
        )

        # v_hb = jnp.where(mask, v_hb, 0.0) # Mask for neighbors
        v_hb = mask * v_hb

        hb_dg = jnp.dot(hb_weights, v_hb)

        return hb_dg


class CrossStacking(je_base.BaseEnergyFunction):
    def __init__(
        self,
        displacement_fn: Callable,
        params: dict[str, float] = {},
        opt_params: dict[str, float] = {},
    ):
        super().__init__(displacement_fn, dna1_defaults.CROSS_STACKING_DG | params, opt_params)

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


class CoaxialStacking(je_base.BaseEnergyFunction):
    def __init__(
        self,
        displacement_fn: Callable,
        params: dict[str, float] = {},
        opt_params: dict[str, float] = {},
    ):
        super().__init__(displacement_fn, dna1_defaults.COAXIAL_STACKING_DG | params, opt_params)

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
        cosphi3_op = jnp.einsum("ij, ij->i", dr_stack_norm_op, jnp.cross(dr_backbone_norm_op, body.back_base_vectors[op_j]))
        cosphi4_op = jnp.einsum("ij, ij->i", dr_stack_norm_op, jnp.cross(dr_backbone_norm_op, body.back_base_vectors[op_i]))

        cx_stack_dg = dna1_interactions.coaxial_stacking(
            dr_stack_op,
            theta4_op,
            theta1_op,
            theta5_op,
            theta6_op,
            cosphi3_op,
            cosphi4_op,
            **self.get_params([
                "dr_low_coax",
                "dr_high_coax",
                "dr_c_low_coax",
                "dr_c_high_coax",
                "k_coax",
                "dr0_coax",
                "dr_c_coax",
                "b_low_coax",
                "b_high_coax",
                "theta0_coax_4",
                "delta_theta_star_coax_4",
                "delta_theta_coax_4_c",
                "a_coax_4",
                "b_coax_4",
                "theta0_coax_1",
                "delta_theta_star_coax_1",
                "delta_theta_coax_1_c",
                "a_coax_1",
                "b_coax_1",
                "theta0_coax_5",
                "delta_theta_star_coax_5",
                "delta_theta_coax_5_c",
                "a_coax_5",
                "b_coax_5",
                "theta0_coax_6",
                "delta_theta_star_coax_6",
                "delta_theta_coax_6_c",
                "a_coax_6",
                "b_coax_6",
                "cos_phi3_star_coax",
                "cos_phi3_c_coax",
                "a_coax_3p",
                "b_cos_phi3_coax",
                "cos_phi4_star_coax",
                "cos_phi4_c_coax",
                "a_coax_4p",
                "b_cos_phi4_coax",
            ])
        )
        # cx_stack_dg = jnp.where(mask, cx_stack_dg, 0.0).sum() # Mask for neighbors
        cx_stack_dg = (mask * cx_stack_dg).sum()

        return cx_stack_dg

