from typing import Protocol

import chex
import jax.numpy as jnp
import jax_md


class NeighborHelper(Protocol):
    @property
    def idx(self) -> jnp.ndarray: ...

    def allocate(self, locs: jax_md.rigid_body.RigidBody) -> "NeighborHelper": ...

    def update(self, locs: jax_md.rigid_body.RigidBody) -> "NeighborHelper": ...


@chex.dataclass
class StaticSimulatorParams:
    # this is commonly referred to as `init_body` in the code
    R: jax_md.rigid_body.RigidBody
    seq: jnp.darray
    mass: jax_md.rigid_body.RigidBody
    bonded_neighbors: jnp.ndarray
    n_steps: int

    @property
    def init_fn(self) -> dict[str, jax_md.rigid_body.RigidBody | jnp.ndarray]:
        return {
            "R": self.R,
            "seq": self.seq,
            "mass": self.mass,
            "bonded_neighbors": self.bonded_neighbors,
        }

    @property
    def step_fn(self) -> dict[str, jax_md.rigid_body.RigidBody | jnp.ndarray]:
        return {
            "seq": self.seq,
            "bonded_neighbors": self.bonded_neighbors,
        }
