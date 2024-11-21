from typing import Any, Protocol

import chex

import jax_dna.simulators.io as jd_sio


@chex.dataclass(frozen=True)
class BaseSimulation:
    def run(self, *args, **kwargs) -> jd_sio.SimulatorTrajectory: ...
