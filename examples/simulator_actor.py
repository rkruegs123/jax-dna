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


jax.config.update("jax_enable_x64", True)

ERR_OBJECTIVE_NOT_READY = "Not all required observables have been obtained."

@ray.remote
class Objective:

    def __init__(
        self,
        required_observables:list[str],
        needed_observables:list[str],
        grad_fn:typing.Callable[[tuple[tuple[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData], ...]], jdna_types.Grads]
    ):
        self.required_observables = required_observables
        self._needed_observables = needed_observables
        self.grad_fn = grad_fn
        self.obtained_observables = []


    def needed_observables(self) -> list[str]:
        return self._needed_observables



    def is_ready(self) -> bool:
        obtained_keys = [obs[0] for obs in self.obtained_observables]
        return all([obs in obtained_keys for obs in self.required_observables])


    def update(
        self,
        sim_results: list[tuple[list[str], typing.Any]],
    ) -> None:
        new_obtained_observables = self.obtained_observables
        currently_needed_observables = set(self._needed_observables)

        for sim_exposes, sim_output in sim_results:
            for exposed in filter(lambda e: e in currently_needed_observables, sim_exposes):
                new_obtained_observables.append((exposed, sim_output))
                currently_needed_observables.remove(exposed)

        self.obtained_observables = new_obtained_observables
        self.needed_observables = list(currently_needed_observables)


    # returns grads
    def calculate(self) -> list[jdna_types.Grads]:
        if not self.is_ready:
            raise ValueError(ERR_OBJECTIVE_NOT_READY)

        sorted_obs = list(map(
            lambda x: x[1],
            sorted(self.obtained_observables, key=lambda x: self.required_observables.index(x[0]),)
        ))

        grads, loss = self.grad_fn(*sorted_obs)

        self.obtained_observables.append([
            ("loss", loss),
            *self.obtained_observables
        ])

        return grads


    def post_step(self) -> None:
        self.needed_observables = self.required_observables


@ray.remote
class SimulatorActor:
    def __init__(
        self,
        fn: typing.Callable[[jdna_types.Params, jdna_types.MetaData], tuple[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData]],
        exposes: list[str],
        meta_data: dict[str, typing.Any],
        write_to: tuple[jdna_types.PathOrStr,...]|None = None,
        writer_fn: typing.Callable[[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData, jdna_types.PathOrStr], None] = None,
    ):
        self.fn = fn
        self._exposes = exposes
        self.meta_data = meta_data
        write_to = Path(write_to) if write_to is not None else None
        self.write_to = write_to
        self.writer_fn = writer_fn

        if writer_fn is not None:
            write_to.mkdir(parents=True, exist_ok=True)


    def exposes(self) -> list[str]:
        return self._exposes


    def run(
        self,
        params: jdna_types.Params,
    ) -> tuple[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData, dict[str, typing.Any]]:
        outs, aux = self.fn(params, self.meta_data)

        if self.writer_fn is not None:
            self.writer_fn(outs, aux)
            return self.write_to

        return outs, aux, self.meta_data


def split_by_ready(objectives: list[Objective]) -> tuple[list[Objective], list[Objective]]:
    not_ready =  list(itertools.filterfalse(lambda x: x.is_ready, objectives))
    ready = list(filter(lambda x: not x.is_ready, objectives))

    return ready, not_ready

