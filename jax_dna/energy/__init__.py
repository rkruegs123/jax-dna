"""The energy model and function for jax_dna."""

from collections.abc import Callable

import jax_md
import jax_dna.energy.base as base
import jax_dna.energy.configuration as configuration
import jax_dna.utils.types as jdna_types

def energy_fn_builder(
    energy_fns:list[base.BaseEnergyFunction],
    energy_configs:list[configuration.BaseConfiguration],
    transform_fn:Callable[[jdna_types.PyTree], jdna_types.PyTree],
    displacement_fn:Callable[[jdna_types.PyTree], jdna_types.PyTree] = jax_md.space.free()[0],
) -> Callable[[jdna_types.PyTree], float]:

    def energy_fn(
        opt_params:jdna_types.PyTree,
    ) -> float:
        """Energy function generated using jax_dna.energy.energy_fn_builder.

        Input:
            opt_params (jdna_types.PyTree): the parameters of the energy function

        Returns:
            float: the energy of the system
        """
        transformed_fns = [
            e_fn(
                displacement_fn=displacement_fn,
                params=(e_c | param).init_params(),
            )
            for param, e_c, e_fn in zip(opt_params, energy_configs, energy_fns, strict=True)
        ]

        return base.ComposedEnergyFunction(
            energy_fns=transformed_fns,
            rigid_body_transform_fn=transform_fn,

        )
    return energy_fn