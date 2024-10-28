import typing
from typing import Any

import chex
import jax_dna.energy.base as jdna_energy
import jax_dna.energy.configuration as jdna_energy_config
import jax_dna.losses.observable_wrappers as jdna_losses
import jax_dna.simulators.base as jdna_simulators
import jax_dna.simulators.io as jdna_sio
import jax_dna.utils.types as jdna_types
import ray

META_DATA = dict[str, Any]


class Objective:
    loss_fn: jdna_losses.LossFn


@ray.remote
class SimulatorActor:
    def __init__(self, simulator: jdna_simulators.BaseSimulation, meta_data: META_DATA):
        self.simulator = simulator
        self.meta_data = meta_data

    def run(
        self, params: jdna_types.Params
    ) -> tuple[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData, META_DATA]:
        sim_traj, sim_meta = self.simulator.run(params, self.meta_data)
        return sim_traj, sim_meta, self.meta_data


@chex.dataclass(frozen=True)
class Optimization:
    energy_fns: list[jdna_energy.BaseEnergyFunction]
    energy_configs: list[jdna_energy_config.BaseConfiguration]
    objectives: list[Objective]
    aggregate_grad_fn: typing.Callable[[list[jdna_types.Grads]], jdna_types.Grads]
    simulators: list[tuple[SimulatorActor, META_DATA]]

    def __post_init__(self):
        # check that the energy functions and configurations are compatible
        # with the parameters
        pass

    def step(self) -> tuple["Optimization", list[jdna_types.Grads]]:
        # run the simulators
        sim_results = [sim.run.remote() for sim, _ in self.simulators]

        # wait for the simulators to finish

        # collect the results and match them up to the objectives
        pass

    def update_params(params: jdna_types.Params, grads: jdna_types.Grads) -> jdna_types.Params:
        # aggregate function
        pass

    def post_step() -> Any:
        pass


if __name__ == "__main__":
    opt = Optimization(params={})
