"""Runs an optimization loop using Ray actors for objectives and simulators."""

import dataclasses as dc
import itertools
import typing

import chex
import optax
import ray

import jax_dna.optimization.objective as jdna_objective
import jax_dna.optimization.simulator as jdna_actor
import jax_dna.utils.types as jdna_types
from jax_dna.ui.loggers import logger as jdna_logger

ERR_MISSING_OBJECTIVES = "At least one objective is required."
ERR_MISSING_SIMULATORS = "At least one simulator is required."
ERR_MISSING_AGG_GRAD_FN = "An aggregate gradient function is required."
ERR_MISSING_OPTIMIZER = "An optimizer is required."

# we assign at the global level to make it easier to mock for testing
get_fn = ray.get
wait_fn = ray.wait
grad_update_fn = optax.apply_updates


def split_by_ready(
    objectives: list[jdna_objective.Objective],
) -> tuple[list[jdna_objective.Objective], list[jdna_objective.Objective]]:
    """Splits a list of objectives into two lists: ready and not ready."""
    ready, not_ready = [], []
    for objective in objectives:
        if get_fn(objective.is_ready.remote()):
            ready.append(objective)
        else:
            not_ready.append(objective)

    return ready, not_ready


@chex.dataclass(frozen=True)
class Optimization:
    """Optimization of a list of objectives using a list of simulators.

    Parameters:
        objectives: A list of objectives to optimize.
        simulators: A list of simulators to use for the optimization.
        aggregate_grad_fn: A function that aggregates the gradients from the objectives.
        optimizer: An optax optimizer.
        optimizer_state: The state of the optimizer.
        logger: A logger to use for the optimization.
    """

    objectives: list[jdna_objective.Objective]
    simulators: list[tuple[jdna_actor.SimulatorActor, jdna_types.MetaData]]
    aggregate_grad_fn: typing.Callable[[list[jdna_types.Grads]], jdna_types.Grads]
    optimizer: optax.GradientTransformation
    optimizer_state: optax.OptState | None = None
    logger: jdna_logger.Logger = dc.field(default_factory=lambda: jdna_logger.Logger())

    def __post_init__(self) -> None:
        """Validate the initialization of the Optimization."""
        if not self.objectives:
            raise ValueError(ERR_MISSING_OBJECTIVES)

        if not self.simulators:
            raise ValueError(ERR_MISSING_SIMULATORS)

        if self.aggregate_grad_fn is None:
            raise ValueError(ERR_MISSING_AGG_GRAD_FN)

        if self.optimizer is None:
            raise ValueError(ERR_MISSING_OPTIMIZER)

    def step(self, params: jdna_types.Params) -> tuple[optax.OptState, list[jdna_types.Grads], list[jdna_types.Grads]]:
        """Perform a single optimization step.

        Args:
            params: The current parameters.

        Returns:
            A tuple containing the updated optimizer state, new params, and the gradients.
        """
        # get the currently needed observables
        # some objectives might use difftre and not actually need something rerun
        # so check which objectives have observables that need to be run
        ready_objectives, not_ready_objectives = split_by_ready(self.objectives)

        grad_refs = [objective.calculate.remote() for objective in ready_objectives]

        ready_names = get_fn([objective.name.remote() for objective in ready_objectives])
        ready_funcs = itertools.repeat(self.logger.set_objective_running, len(ready_names))

        not_ready_names = get_fn([objective.name.remote() for objective in not_ready_objectives])
        not_ready_funcs = itertools.repeat(self.logger.set_objective_started, len(not_ready_names))

        sim_names = get_fn([sim.name.remote() for sim in self.simulators])
        sim_funcs = itertools.repeat(self.logger.set_simulator_started, len(sim_names))

        names = itertools.chain(ready_names, not_ready_names, sim_names)
        funcs = itertools.chain(ready_funcs, not_ready_funcs, sim_funcs)
        for name, func in zip(names, funcs, strict=True):
            func(name)

        need_observables = list(
            itertools.chain.from_iterable(get_fn([co.needed_observables.remote() for co in not_ready_objectives]))
        )

        needed_simulators = [
            sim for sim in self.simulators if set(get_fn(sim.exposes.remote())) & set(need_observables)
        ]

        needed_names = get_fn([sim.name.remote() for sim in needed_simulators])
        needed_exposes = get_fn([sim.exposes.remote() for sim in needed_simulators])

        sim_remotes = [sim.run.remote(params) for sim in needed_simulators]

        simid_exposes = {}
        simid_name = {}
        for sr, name, exposes in zip(sim_remotes, needed_names, needed_exposes, strict=True):
            simid_exposes[sr.task_id().hex()] = exposes
            simid_name[sr.task_id().hex()] = name

            self.logger.set_simulator_running(name)
            [self.logger.set_observable_running(e) for e in exposes]

        # wait for the simulators to finish
        while not_ready_objectives:
            # `done` is a list of object refs that are ready to collect.
            #  sim_remotes is a list of object refs that are not ready to collect.
            done, sim_remotes = wait_fn(sim_remotes)
            if done:
                captured_results = []
                for d in done:
                    task_id = d.task_id().hex()
                    exposes = simid_exposes[task_id]
                    result = get_fn(d)
                    captured_results.append((exposes, result))
                    if self.logger:
                        self.logger.set_simulator_complete(simid_name[task_id])
                        for expose in exposes:
                            self.logger.set_observable_complete(expose)

                # update the objectives with the new observables and check if they are ready
                get_fn([objective.update.remote(captured_results) for objective in not_ready_objectives])
                ready_objectives, not_ready_objectives = split_by_ready(not_ready_objectives)
                for name in get_fn([objective.name.remote() for objective in ready_objectives]):
                    self.logger.set_objective_running(name)

                grad_refs += [objective.calculate.remote() for objective in ready_objectives]

        grads_resolved = get_fn(grad_refs)

        for name in get_fn([o.name.remote() for o in self.objectives]):
            self.logger.set_objective_complete(name)

        grads = self.aggregate_grad_fn(grads_resolved)

        opt_state = self.optimizer.init(params) if self.optimizer_state is None else self.optimizer_state

        updates, opt_state = self.optimizer.update(grads, opt_state, params)

        new_params = grad_update_fn(params, updates)

        return opt_state, new_params, grads

    def post_step(
        self,
        optimizer_state: optax.OptState,
        opt_params: jdna_types.Params,
    ) -> "Optimization":
        """An update step intended to be called after an optimization step."""
        _ = get_fn([o.post_step.remote(opt_params) for o in self.objectives])
        return self.replace(optimizer_state=optimizer_state)


@chex.dataclass(frozen=True)
class SimpleOptimizer:
    """A simple optimizer that uses a single objective and simulator."""

    objective: jdna_objective.Objective
    simulator: jdna_actor.SimulatorActor
    optimizer: optax.GradientTransformation
    optimizer_state: optax.OptState | None = None
    logger: jdna_logger.Logger = dc.field(default_factory=lambda: jdna_logger.Logger())

    def step(self, params: jdna_types.Params) -> tuple[optax.OptState, list[jdna_types.Grads], list[jdna_types.Grads]]:
        """Perform a single optimization step.

        Args:
            params: The current parameters.

        Returns:
            A tuple containing the updated optimizer state, new params, and the gradients.
        """
        # get the currently needed observables
        # some objectives might use difftre and not actually need something rerun
        # so check which objectives have observables that need to be run
        if self.objective.is_ready():
            grads = self.objective.calculate()
        else:
            observables = self.simulator.run(params)
            exposes = self.simulator.exposes()
            self.objective.update(
                [
                    (exposes, observables),
                ]
            )
            grads = self.objective.calculate()

        opt_state = self.optimizer.init(params) if self.optimizer_state is None else self.optimizer_state

        updates, opt_state = self.optimizer.update(grads, opt_state, params)

        new_params = grad_update_fn(params, updates)

        return opt_state, new_params, grads

    def post_step(self, optimizer_state: optax.OptState, opt_params: jdna_types.Params) -> "SimpleOptimizer":
        """An update step intended to be called after an optimization step."""
        self.objective.post_step(opt_params)
        return self.replace(optimizer_state=optimizer_state)
