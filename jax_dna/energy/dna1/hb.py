from typing import Callable

import jax.numpy as jnp
import jax_md

import jax_dna.energy.base as je_base
import jax_dna.energy.dna1.defaults as dna1_defaults
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.energy.dna1.nuecleotide as dna1_nucleotide
import jax_dna.energy.utils as je_utils
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as typ


class HydrogenBonding(je_base.BaseEnergyFunction):
    def __init__(
        self,
        displacement_fn: Callable,
        params: dict[str, float] = dna1_defaults.HB_DG,
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
        hb_weights = jnp.dot(hb_probs, self.get_param("ss_hb_weights_flat"))

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
