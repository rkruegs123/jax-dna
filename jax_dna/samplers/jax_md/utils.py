"""Utilities for JAX-MD samplers."""

from typing import Protocol

import chex
import jax.numpy as jnp
import jax_md


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

    # this is commonly referred to as `init_body` in the code
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
