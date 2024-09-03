"""Utilities for JAX-MD samplers."""

from collections.abc import Callable
import functools
from typing import Protocol

import chex
import jax
import jax.numpy as jnp
import jax_md


ERR_CHKPNT_SCN = "`checkpoint_every` must evenly divide the length of `xs`. Got {} and {}."

class NeighborHelper(Protocol):
    """Helper class for managing neighbor lists."""

    @property
    def idx(self) -> jnp.ndarray:
        """Return the indices of the unbonded neighbors."""
        ...

    def allocate(self, locs: jax_md.rigid_body.RigidBody) -> "NeighborHelper":
        """Allocate memory for the neighbor list."""
        ...

    def update(self, locs: jax_md.rigid_body.RigidBody) -> "NeighborHelper":
        """Update the neighbor list."""
        ...


@chex.dataclass
class StaticSimulatorParams:
    """Static parameters for the simulator."""

    # this is commonly referred to as `init_body` in the code, but the argument is named `R` in jax_md
    R: jax_md.rigid_body.RigidBody
    seq: jnp.darray
    mass: jax_md.rigid_body.RigidBody
    bonded_neighbors: jnp.ndarray
    n_steps: int

    @property
    def init_fn(self) -> dict[str, jax_md.rigid_body.RigidBody | jnp.ndarray]:
        """Return the kwargs for initial state of the simulator."""
        return {
            "R": self.R,
            "seq": self.seq,
            "mass": self.mass,
            "bonded_neighbors": self.bonded_neighbors,
        }

    @property
    def step_fn(self) -> dict[str, jax_md.rigid_body.RigidBody | jnp.ndarray]:
        """Return the kwargs for the step_fn of the simulator."""
        return {
            "seq": self.seq,
            "bonded_neighbors": self.bonded_neighbors,
        }


def split_and_stack(x:jnp.ndarray, n:int) -> jnp.ndarray:
    """Split `xs` into `n` pieces and stack them."""
    return jax.tree_map(lambda y: jnp.stack(jnp.split(y, n)), x)


def flatten_n(x:jnp.ndarray, n:int) -> jnp.ndarray:
    """Flatten `x` by `n` levels."""
    return jax.tree_map(lambda y: jnp.reshape(y, (-1,) + y.shape[n:]), x)


def checkpoint_scan(f, init, xs, checkpoint_every):
    """Replicates the behavior of `jax.lax.scan` but checkpoints gradients every `checkpoint_every` steps."""
    flat_xs, _ = jax.tree_util.tree_flatten(xs)
    length = flat_xs[0].shape[0]
    outer_iterations, residual = divmod(length, checkpoint_every)
    if residual:
        raise ValueError(ERR_CHKPNT_SCN.format(checkpoint_every, length))
    reshaped_xs = split_and_stack(xs, outer_iterations)

    @jax.checkpoint
    def inner_loop(_init, _xs):
        return jax.lax.scan(f, _init, _xs)

    final, result = jax.lax.scan(inner_loop, init, reshaped_xs)

    return final, flatten_n(result, 2)

