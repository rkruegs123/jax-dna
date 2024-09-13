from typing import Any

import chex
import jax.numpy as jnp
import jax_md
from typing_extensions import override

import jax_dna.input.trajectory as jd_traj
import jax_dna.observables.base as jd_obs_base
import jax_dna.simulators.io as jd_sio

loss_input = jnp.ndarray | tuple[jnp.ndarray, dict[str, Any]]


@chex.dataclass
class LossFn:
    def __call__(self, actual: loss_input, target: loss_input, weights: jnp.ndarray) -> float:
        pass


class SquaredError(LossFn):
    @override
    def __call__(self, actual: jnp.ndarray, target: jnp.ndarray, weights: jnp.ndarray) -> float:
        return jnp.mean(weights * ((actual - target) ** 2) ** 0.5)


@chex.dataclass
class ObservableLossFn:
    observable: jd_obs_base.BaseObservable
    loss_fn: LossFn
    return_observable: bool = False

    def __call__(self, trajectory: jd_sio.SimulatorTrajectory, target: jnp.ndarray, weights: jnp.ndarray) -> float:
        if self.return_observable:
            observable = self.observable(trajectory)
            return self.loss_fn(observable, target, weights), observable
        else:
            return self.loss_fn(self.observable(trajectory), target, weights)
