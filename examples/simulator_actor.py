import dataclasses as dc
import itertools
from pathlib import Path
import typing

import chex
import jax
import jax_md
import ray

import jax_dna.simulators.base as jdna_simulators
import jax_dna.simulators.io as jdna_sio
import jax_dna.utils.types as jdna_types

ERR_OBJECTIVE_NOT_READY = "Not all required observables have been obtained."

@ray.remote
class Objective:

    def __init__(
        self,
        required_observables:list[str],
        needed_observables:list[str],
        logging_observables:list[str],
        grad_fn:typing.Callable[[tuple[tuple[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData], ...]], jdna_types.Grads]
    ):
        self.required_observables = required_observables
        self._needed_observables = needed_observables
        self.grad_fn = grad_fn
        self._obtained_observables = []
        self._logging_observables = logging_observables


    def needed_observables(self) -> list[str]:
        return self._needed_observables


    def obtained_observables(self) -> list[tuple[str, typing.Any]]:
        return self._obtained_observables


    def logging_observables(self) -> list[tuple[str, typing.Any]]:
        lastest_observed = self._obtained_observables
        return_values = []
        for log_obs in self._logging_observables:
            for obs in lastest_observed:
                if obs[0] == log_obs:
                    return_values.append((obs))
                    break
        return return_values


    def is_ready(self) -> bool:
        obtained_keys = [obs[0] for obs in self._obtained_observables]
        return all([obs in obtained_keys for obs in self.required_observables])


    def update(
        self,
        sim_results: list[tuple[list[str], typing.Any]],
    ) -> None:

        new_obtained_observables = self._obtained_observables
        currently_needed_observables = set(self._needed_observables)

        for sim_exposes, sim_output in sim_results:
            for exposed, output in filter(
                lambda e: e[0] in currently_needed_observables,
                zip(sim_exposes, sim_output)
            ):
                new_obtained_observables.append((exposed, output))
                currently_needed_observables.remove(exposed)

        self._obtained_observables = new_obtained_observables
        self._needed_observables = list(currently_needed_observables)


    def calculate(self) -> list[jdna_types.Grads]:
        if not self.is_ready():
            raise ValueError(ERR_OBJECTIVE_NOT_READY)

        sorted_obs = list(map(
            lambda x: x[1],
            sorted(
                self._obtained_observables,
                key=lambda x: self.required_observables.index(x[0]),
            )
        ))

        grads, loss = self.grad_fn(*sorted_obs)

        self._obtained_observables = [
            ("loss", loss),
            *[(ro, so) for (ro, so) in zip(self.required_observables, sorted_obs)],
        ]

        return grads


    def post_step(self) -> None:
        self._needed_observables = self.required_observables
        self._obtained_observables = []


@ray.remote
class SimulatorActor:
    def __init__(
        self,
        fn: typing.Callable[[jdna_types.Params, jdna_types.MetaData], tuple[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData]],
        exposes: list[str],
        meta_data: dict[str, typing.Any],
        writer_fn: typing.Callable[[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData, jdna_types.PathOrStr], None] = None,
    ):
        self.fn = fn
        self._exposes = exposes
        self.meta_data = meta_data
        self.writer_fn = writer_fn


    def exposes(self) -> list[str]:
        return self._exposes


    def run(
        self,
        params: jdna_types.Params,
    ) -> tuple[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData, dict[str, typing.Any]]:
        outs, aux = self.fn(params, self.meta_data)

        if self.writer_fn is not None:
            return self.writer_fn(outs, aux, self.meta_data)

        return outs, aux, self.meta_data


def split_by_ready(objectives: list[Objective]) -> tuple[list[Objective], list[Objective]]:
    not_ready =  list(itertools.filterfalse(lambda x: x.is_ready, objectives))
    ready = list(filter(lambda x: not x.is_ready, objectives))
    return ready, not_ready

