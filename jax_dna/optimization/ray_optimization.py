import itertools
import typing

import chex
import optax
import ray

import jax_dna.optimization.simulation_actor as jdna_actor
import jax_dna.optimization.objective_actor as jdna_objective
import jax_dna.utils.types as jdna_types


ERR_MISSING_OBJECTIVES = "At least one objective is required."
ERR_MISSING_SIMULATORS = "At least one simulator is required."
ERR_MISSING_AGG_GRAD_FN = "An aggregate gradient function is required."
ERR_MISSING_OPTIMIZER = "An optimizer is required."

def split_by_ready(
    objectives: list[jdna_objective.Objective],
) -> tuple[list[jdna_objective.Objective], list[jdna_objective.Objective]]:
    """Splits a list of objectives into two lists: ready and not ready."""
    ready, not_ready = [], []
    for objective in objectives:
        if ray.get(objective.is_ready.remote()):
            ready.append(objective)
        else:
            not_ready.append(objective)

    return ready, not_ready


@chex.dataclass(frozen=True)
class Optimization:
    """Optimization of a list of objectives using a list of simulators.

    Attributes:
        objectives: A list of objectives to optimize.
        simulators: A list of simulators to use for the optimization.
        aggregate_grad_fn: A function that aggregates the gradients from the objectives.
        optimizer: An optax optimizer.
        optimizer_state: The state of the optimizer.
    """
    objectives: list[jdna_objective.Objective]
    simulators: list[tuple[jdna_actor.SimulatorActor, jdna_types.MetaData]]
    aggregate_grad_fn: typing.Callable[[list[jdna_types.Grads]], jdna_types.Grads]
    optimizer: optax.GradientTransformation
    optimizer_state: optax.OptState | None = None

    def __post_init__(self):
        if not self.objectives:
            raise ValueError(ERR_MISSING_OBJECTIVES)

        if not self.simulators:
            raise ValueError(ERR_MISSING_SIMULATORS)

        if self.aggregate_grad_fn is None:
            raise ValueError(ERR_MISSING_AGG_GRAD_FN)

        if self.optimizer is None:
            raise ValueError(ERR_MISSING_OPTIMIZER)


    def step(
        self,
        params: jdna_types.Params
    ) -> tuple[optax.OptState, list[jdna_types.Grads]]:
        """Perform a single optimization step.

        Args:
            params: The current parameters.

        Returns:
            A tuple containing the updated optimizer state and the gradients.
        """
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
        while not_ready_objectives:
            # `done` is a list of object refs that are ready to collect.
            #  sim_remotes is a list of object refs that are not ready to collect.
            done, sim_remotes = ray.wait(sim_remotes)
            if done:
                captured_results = []
                for d in done:
                    task_id = d.task_id().hex()
                    exposes = simid_exposes[task_id]
                    result = ray.get(d)
                    captured_results.append((exposes, result))
                # update the objectives with the new observables and check if they are ready
                ray.get(
                    [objective.update.remote(captured_results) for objective in not_ready_objectives]
                )
                ready, not_ready_objectives = split_by_ready(not_ready_objectives)
                grad_refs += [objective.calculate.remote() for objective in ready]

        grads_resolved = ray.get(grad_refs)
        grads = self.aggregate_grad_fn(grads_resolved)

        if self.optimizer_state is None:
            opt_state = self.optimizer.init(params)
        else:
            opt_state = self.optimizer_state

        updates, opt_state = self.optimizer.update(grads, opt_state, params)

        new_params = optax.apply_updates(params, updates)

        return opt_state, new_params

    def post_step(
        self,
        optimizer_state: optax.OptState,
        opt_params: jdna_types.Params,
    ) -> "Optimization":
        """An update step intended to be called after an optimization step."""
        _ = ray.get([o.post_step.remote(opt_params) for o in self.objectives])
        return self.replace(optimizer_state=optimizer_state)
