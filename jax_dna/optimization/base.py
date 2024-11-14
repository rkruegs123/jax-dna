import dataclasses as dc
import itertools
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

ERR_OBJECTIVE_NOT_READY = "Not all required observables have been obtained."


@chex.dataclass(frozen=True)
class Objective:
    required_observables: list[str] = dc.field(default=None)
    needed_observables: list[str] = dc.field(default=None)
    obtained_observables: list[tuple[str, tuple[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData]]] = dc.field(default=None)
    grad_fn: typing.Callable[[tuple[tuple[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData], ...]], jdna_types.Grads] = dc.field(default=None)

    @property
    def is_ready(self) -> bool:
        obtained_keys = [obs[0] for obs in self.obtained_observables]
        return all([obs in obtained_keys for obs in self.required_observables])

    def update(
        self,
        sim_results: list[tuple[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData]],
    ) -> "Objective":
        new_obtained_observables = self.obtained_observables
        for sim_result in sim_results:
            sim_meta = sim_result[1]
            for expose in sim_meta.exposes:
                new_obtained_observables.append((expose, sim_result))
        return self.replace(obtained_observables=new_obtained_observables)


    # returns grads
    def calculate(self) -> list[jdna_types.Grads]:
        if not self.is_ready:
            raise ValueError(ERR_OBJECTIVE_NOT_READY)

        sorted_obs = list(map(
            lambda x: x[1],
            sorted(self.obtained_observables, key=lambda x: self.required_observables.index(x[0]),)
        ))

        return self.grad_fn(*sorted_obs)


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


def split_by_ready(objectives: list[Objective]) -> tuple[list[Objective], list[Objective]]:
    not_ready =  list(itertools.filterfalse(lambda x: x.is_ready, objectives))
    ready = list(filter(lambda x: not x.is_ready, objectives))

    return ready, not_ready


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
        # get the currently needed observables
        # some objectives might use difftre and not actually need something rerun
        # so check which objectives have observables that need to be run
        ready_objectives, not_ready_objectives = split_by_ready(self.objectives)

        grad_refs = [objective.calculate.remote() for objective in ready_objectives]

        need_observables = itertools.chain.from_iterable([co.needed_observables for co in not_ready_objectives])
        needed_simulators = [sim for sim in self.simulators if set(sim.exposes) & set(need_observables)]

        sim_results = [{sim.exposes:sim.run.remote()} for sim in needed_simulators]

        # wait for the simulators to finish
        n_runs = len(sim_results)
        while not_ready_objectives:
            # `done` is a list of object refs that are ready to collect.
            #  `_` is a list of object refs that are not ready to collect.
            done, _ = ray.wait(sim_results, num_returns=n_runs)
            if done:
                updated_objectives = [objective.update(done) for objective in not_ready_objectives]
                ready, not_ready_objectives = split_by_ready(updated_objectives)
                grad_refs += [objective.calculate.remote() for objective in ready]

        grads = ray.get(grad_refs)

        return self.post_step(), grads

    def update_params(self, params: jdna_types.Params, grads: jdna_types.Grads) -> jdna_types.Params:
        # aggregate function?
        pass

    def post_step(self,) -> "Optimization":
        objectives = [objective.post_step() for objective in self.objectives]
        return self.replace(objectives=objectives)


if __name__ == "__main__":
    opt = Optimization(params={})
