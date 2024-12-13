"""Loss functions for observables."""

from typing import Any

import chex
import jax.numpy as jnp
from typing_extensions import override

import jax_dna.observables.base as jd_obs_base
import jax_dna.simulators.io as jd_sio

loss_input = jnp.ndarray | tuple[jnp.ndarray, dict[str, Any]]


@chex.dataclass
class LossFn:
    """Base class for loss functions."""

    def __call__(self, actual: loss_input, target: loss_input, weights: jnp.ndarray) -> float:
        """Calculate the loss."""
        raise NotImplementedError("Subclasses must implement this method.")


@chex.dataclass
class SquaredError(LossFn):
    """Calculate the squared error between the actual and target values."""

    @override
    def __call__(self, actual: jnp.ndarray, target: jnp.ndarray) -> float:
        return (target - actual) ** 2


@chex.dataclass
class ObservableLossFn:
    """A simple loss function wrapper for an observable."""

    observable: jd_obs_base.BaseObservable
    loss_fn: LossFn
    return_observable: bool = False

    def __call__(self, trajectory: jd_sio.SimulatorTrajectory, target: jnp.ndarray, weights: jnp.ndarray) -> float:
        """Calculate the loss for the observable over the trajectory."""
        observable = jnp.sum(self.observable(trajectory) * weights)
        vals = [self.loss_fn(observable, target)]

        if self.return_observable:
            vals.append(observable)

        return tuple(vals)


def l2_loss(actual: jnp.ndarray, target: jnp.ndarray) -> float:
    """Calculate the L2 loss between the actual and target values."""
    return jnp.sum((actual - target) ** 2)
