import typing
from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import jax_md

import jax_dna.energy.base as je_base
import jax_dna.energy.dna1.defaults as dna1_defaults
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.energy.utils as je_utils
import jax_dna.utils.types as typ


class Fene_DG(je_base.BaseEnergyFunction):
    def __init__(self, displacement_fn: Callable, params: dict[str, float], opt_params: dict[str, float]):
        super().__init__(displacement_fn, dna1_defaults.V_FENE | params, opt_params)

    @typing.override
    def __call__(
        self,
        body: jax_md.rigid_body.RigidBody,
        seq: jnp.ndarray,
        bonded_neghbors: typ.Arr_Bonded_Neighbors,
        unbounded_neghbors: typ.Arr_Unbonded_Neighbors,
    ) -> typ.Scalar:
        nn_i = bonded_neghbors[:, 0]
        nn_j = bonded_neghbors[:, 1]
        Q = body.orientation
        com_to_backbone = self.get_param("com_to_backbone")

        back_base_vectors = je_utils.q_to_back_base(Q)  # space frame, normalized

        back_sites = body.center + com_to_backbone * back_base_vectors

        dr_back_nn = self.displacement_mapped(back_sites[nn_i], back_sites[nn_j])
        r_back_nn = jnp.linalg.norm(dr_back_nn, axis=1)

        eps_backbone = self.get_param("eps_backbone")
        r0_backbone = self.get_param("r0_backbone")
        delta_backbone = self.get_param("delta_backbone")
        fmax = self.get_param("fmax")
        finf = self.get_param("finf")

        return dna1_interactions.v_fene_smooth(
            r_back_nn,
            eps_backbone,
            r0_backbone,
            delta_backbone,
            fmax,
            finf,
        ).sum()
