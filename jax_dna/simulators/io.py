"""Common data structures for simulator I/O."""

import dataclasses as dc
from typing import Any

import chex
import jax.numpy as jnp
import jax_md


@chex.dataclass()
class SimulatorMetaData:
    """Metadata for a simulation run."""

    seq_oh: jnp.ndarray
    strand_lengths: list[int]
    grads: jax_md.rigid_body.RigidBody | None = None
    exposes: list[str] = dc.field(default_factory=list)
    misc_data: dict[str, Any] = dc.field(default_factory=dict)

    def slice(self, key: int | slice) -> "SimulatorMetaData":
        """Slice the metadata."""

        new_grads = self.grads
        if new_grads:
            centers = [{k: c[k][key] for k in c} for c in self.grads.center]
            orientations = [
                {k: jax_md.rigid_body.Quaternion(vec=c[k].vec[key]) for k in c} for c in self.grads.orientation
            ]

            new_grads = jax_md.rigid_body.RigidBody(
                center=centers,
                orientation=orientations,
            )

        return self.replace(
            grads=new_grads,
        )


@chex.dataclass()
class SimulatorTrajectory:
    """A trajectory of a simulation run."""

    rigid_body: jax_md.rigid_body.RigidBody
    location: str | None = None

    def slice(self, key: int | slice) -> "SimulatorTrajectory":
        """Slice the trajectory."""

        sub_rigid_body = jax_md.rigid_body.RigidBody(
            center=self.rigid_body.center[key],
            orientation=jax_md.rigid_body.Quaternion(
                vec=self.rigid_body.orientation.vec[key],
            ),
        )

        return self.replace(
            rigid_body=jax_md.rigid_body.RigidBody(
                center=self.rigid_body.center[key, ...],
                orientation=jax_md.rigid_body.Quaternion(
                    vec=self.rigid_body.orientation.vec[key, ...],
                ),
            )
        )

    def __len__(self):
        print(self.rigid_body.center.shape, self.rigid_body.center.shape[0])
        return self.rigid_body.center.shape[0]
