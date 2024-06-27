from typing import Callable

import jax.numpy as jnp

import jax_dna.energy.base as je_base
import jax_dna.energy.dna1.defaults as dna1_defaults
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.energy.dna1.nuecleotide as dna1_nucleotide
import jax_dna.utils.types as typ


class Bonded(je_base.BaseEnergyFunction):
    def __init__(
        self,
        displacement_fn: Callable,
        params: dict[str, float] = dna1_defaults.BONDED_DG,
        opt_params: dict[str, float] = {},
    ):
        super().__init__(displacement_fn, dna1_defaults.BONDED_DG | params, opt_params)

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
                ]
            ),
        ).sum()

        return exc_vol_bonded_dg


class Unbonded(je_base.BaseEnergyFunction):
    def __init__(
        self,
        displacement_fn: Callable,
        params: dict[str, float] = dna1_defaults.UNBONDED_DG,
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
