from collections.abc import Callable
from typing import Protocol

import chex
import jax.numpy as jnp

import jax_dna.input.trajectory as jd_traj


@chex.dataclass(frozen=True)
class BaseObservable:
    rigid_body_transform_fn: Callable

    def __call__(self, trajectory: jd_traj.Trajectory) -> jnp.ndarray:
        pass
