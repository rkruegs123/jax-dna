import dataclasses as dc
from typing import Callable

import chex
import jax.numpy as jnp

import jax_dna.energy.base as je_base
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.energy.dna1.nucleotide as dna1_nucleotide
import jax_dna.energy.utils as je_utils
import jax_dna.input.dna1.bonded as config
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
        nn_i = bonded_neghbors[:, 0]
        nn_j = bonded_neghbors[:, 1]

        dr_base = self.displacement_mapped(body.base_sites[nn_i], body.base_sites[nn_j])
        dr_back_base = self.displacement_mapped(body.back_sites[nn_i], body.base_sites[nn_j])
        dr_base_back = self.displacement_mapped(body.base_sites[nn_i], body.back_sites[nn_j])

        exc_vol_bonded_dg = dna1_interactions.exc_vol_bonded(
            dr_base,
            dr_back_base,
            dr_base_back,
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
        ).sum()

        return exc_vol_bonded_dg


@chex.dataclass(frozen=True)
class Fene(je_base.BaseEnergyFunction):
    params: config.VFeneConfiguration

    def __call__(
        self,
        body: dna1_nucleotide.Nucleotide,
        seq: jnp.ndarray,
        bonded_neghbors: typ.Arr_Bonded_Neighbors,
        unbounded_neghbors: typ.Arr_Unbonded_Neighbors,
    ) -> typ.Scalar:
        nn_i = bonded_neghbors[:, 0]
        nn_j = bonded_neghbors[:, 1]

        dr_back_nn = self.displacement_mapped(body.back_sites[nn_i], body.back_sites[nn_j])
        r_back_nn = jnp.linalg.norm(dr_back_nn, axis=1)

        return dna1_interactions.v_fene_smooth(
            r_back_nn,
            self.params.eps_backbone,
            self.params.r0_backbone,
            self.params.delta_backbone,
            self.params.fmax,
            self.params.finf,
        ).sum()


# class Stacking(je_base.BaseEnergyFunction):
#     def __init__(
#         self,
#         displacement_fn: Callable,
#         params: dict[str, float] = {},
#         opt_params: dict[str, float] = {},
#     ):
#         super().__init__(displacement_fn, dna1_defaults.STACKING_DG | params, opt_params)

#     def __call__(
#         self,
#         body: dna1_nucleotide.Nucleotide,
#         seq: jnp.ndarray,
#         bonded_neghbors: typ.Arr_Bonded_Neighbors,
#         unbonded_neghbors: typ.Arr_Unbonded_Neighbors,
#     ) -> typ.Scalar:
#         nn_i = bonded_neghbors[:, 0]
#         nn_j = bonded_neghbors[:, 1]

#         ## Fene variables
#         dr_back_nn = self.displacement_mapped(body.back_sites[nn_i], body.back_sites[nn_j])  # N x N x 3
#         r_back_nn = jnp.linalg.norm(dr_back_nn, axis=1)

#         dr_stack_nn = self.displacement_mapped(body.stack_sites[nn_i], body.stack_sites[nn_j])
#         r_stack_nn = jnp.linalg.norm(dr_stack_nn, axis=1)
#         theta4 = jnp.arccos(jd_math.clamp(jnp.einsum("ij, ij->i", body.base_normals[nn_i], body.base_normals[nn_j])))
#         theta5 = jnp.pi - jnp.arccos(
#             jd_math.clamp(jnp.einsum("ij, ij->i", dr_stack_nn, body.base_normals[nn_j]) / r_stack_nn)
#         )
#         theta6 = jnp.pi - jnp.arccos(
#             jd_math.clamp(jnp.einsum("ij, ij->i", body.base_normals[nn_i], dr_stack_nn) / r_stack_nn)
#         )
#         cosphi1 = -jnp.einsum("ij, ij->i", body.cross_prods[nn_i], dr_back_nn) / r_back_nn
#         cosphi2 = -jnp.einsum("ij, ij->i", body.cross_prods[nn_j], dr_back_nn) / r_back_nn

#         v_stack = dna1_interactions.stacking(
#             r_stack_nn,
#             theta4,
#             theta5,
#             theta6,
#             cosphi1,
#             cosphi2,
#             **self.get_params(
#                 [
#                     "dr_low_stack",
#                     "dr_high_stack",
#                     "eps_stack",
#                     "a_stack",
#                     "dr0_stack",
#                     "dr_c_stack",
#                     "dr_c_low_stack",
#                     "dr_c_high_stack",
#                     "b_low_stack",
#                     "b_high_stack",
#                     "theta0_stack_4",
#                     "delta_theta_star_stack_4",
#                     "a_stack_4",
#                     "delta_theta_stack_4_c",
#                     "b_stack_4",
#                     "theta0_stack_5",
#                     "delta_theta_star_stack_5",
#                     "a_stack_5",
#                     "delta_theta_stack_5_c",
#                     "b_stack_5",
#                     "theta0_stack_6",
#                     "delta_theta_star_stack_6",
#                     "a_stack_6",
#                     "delta_theta_stack_6_c",
#                     "b_stack_6",
#                     "neg_cos_phi1_star_stack",
#                     "a_stack_1",
#                     "neg_cos_phi1_c_stack",
#                     "b_neg_cos_phi1_stack",
#                     "neg_cos_phi2_star_stack",
#                     "a_stack_2",
#                     "neg_cos_phi2_c_stack",
#                     "b_neg_cos_phi2_stack",
#                 ]
#             ),
#         )

#         stack_probs = je_utils.get_pair_probs(seq, nn_i, nn_j)
#         stack_weights = jnp.dot(stack_probs, self.get_param("ss_stack_weights").flatten())

#         stack_dg = jnp.dot(stack_weights, v_stack)

#         return stack_dg
