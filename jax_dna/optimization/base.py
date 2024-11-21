import dataclasses as dc
import itertools
import typing
from typing import Any

import chex
import examples.simulator_actor as sim_actor
import jax
import jax_dna.energy.base as jdna_energy
import jax_dna.energy.configuration as jdna_energy_config
import jax_dna.losses.observable_wrappers as jdna_losses
import jax_dna.simulators.base as jdna_simulators
import jax_dna.simulators.io as jdna_sio
import jax_dna.utils.types as jdna_types
import optax
import ray

META_DATA = dict[str, Any]


def split_by_ready(
    objectives: list[sim_actor.Objective],
) -> tuple[list[sim_actor.Objective], list[sim_actor.Objective]]:
    not_ready = list(itertools.filterfalse(lambda x: ray.get(x.is_ready.remote()), objectives))
    ready = list(filter(lambda x: not x.is_ready, objectives))

    return ready, not_ready


@chex.dataclass(frozen=True)
class Optimization:
    objectives: list[sim_actor.Objective]
    simulators: list[tuple[sim_actor.SimulatorActor, META_DATA]]
    aggregate_grad_fn: typing.Callable[[list[jdna_types.Grads]], jdna_types.Grads]
    optimizer: optax.GradientTransformation
    optimizer_state: optax.OptState | None = None

    def __post_init__(self):
        # check that the energy functions and configurations are compatible
        # with the parameters
        pass

    def step(self, params: jdna_types.Params) -> tuple["Optimization", list[jdna_types.Grads]]:
        # get the currently needed observables
        # some objectives might use difftre and not actually need something rerun
        # so check which objectives have observables that need to be run
        ready_objectives, not_ready_objectives = split_by_ready(self.objectives)

        grad_refs = [objective.calculate.remote() for objective in ready_objectives]

        need_observables = list(
            itertools.chain.from_iterable(ray.get([co.needed_observables.remote() for co in not_ready_objectives]))
        )
        needed_simulators = [
            sim for sim in self.simulators if set(ray.get(sim.exposes.remote())) & set(need_observables)
        ]

        sim_remotes = [sim.run.remote(params) for sim in needed_simulators]

        simid_exposes = {}
        for sr, sim in zip(sim_remotes, needed_simulators):
            simid_exposes[sr.task_id().hex()] = ray.get(sim.exposes.remote())

        # wait for the simulators to finish
        n_runs = len(sim_remotes)
        while not_ready_objectives:
            # `done` is a list of object refs that are ready to collect.
            #  `_` is a list of object refs that are not ready to collect.
            done, _ = ray.wait(sim_remotes, num_returns=n_runs)
            if done:
                captured_results = {}
                for d in done:
                    task_id = d.task_id().hex()
                    exposes = simid_exposes[task_id]
                    result = ray.get(d)
                    # need to accomodate mutliple exposes for a single simulator
                    print(result, exposes)
                    captured_results[exposes] = result
                print(captured_results)
                updated_objectives = ray.get(
                    [objective.update.remote(captured_results) for objective in not_ready_objectives]
                )
                ready, not_ready_objectives = split_by_ready(updated_objectives)
                grad_refs += [objective.calculate.remote() for objective in ready]

        grads = self.aggregate_grad_fn(ray.get(grad_refs))

        if self.optimizer_state is None:
            opt_state = self.optimizer.init(params)
        else:
            opt_state = self.optimizer_state

        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return self.post_step(opt_state), new_params

    def post_step(
        self,
        optimizer_state: optax.OptState,
    ) -> None:
        """"""
        _ = ray.get([o.post_step.remote() for o in self.objectives])
        return self.replace(optimizer_state=optimizer_state)
