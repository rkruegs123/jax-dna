import typing
from typing import Any

import chex
import jax_dna.energy.base as jdna_energy
import jax_dna.energy.configuration as jdna_energy_config
import jax_dna.losses.observable_wrappers as jdna_losses
import jax_dna.simulators.base as jdna_simulators
import jax_dna.utils.types as jdna_types


class Objective:
    loss_fn: jdna_losses.LossFn


@chex.dataclass(frozen=True)
class Optimization:
    energy_fns: list[jdna_energy.BaseEnergyFunction]
    energy_configs: list[jdna_energy_config.BaseConfiguration]
    objectives: list[Objective]
    aggregate_grad_fn: typing.Callable[[list[jdna_types.Grads]], jdna_types.Grads]
    simulators: list[jdna_simulators.BaseSimulation]

    def __post_init__(self):
        # check that the energy functions and configurations are compatible
        # with the parameters
        pass

    def step(self) -> tuple["Optimization", list[jdna_types.Grads]]:
        pass

    def update_params(params: jdna_types.Params, grads: jdna_types.Grads) -> jdna_types.Params:
        pass

    def post_step() -> Any:
        pass


if __name__ == "__main__":
    opt = Optimization(params={})
