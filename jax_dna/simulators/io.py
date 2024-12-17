"""Common data structures for simulator I/O."""

import chex
import jax_md


@chex.dataclass()
class SimulatorTrajectory:
    """A trajectory of a simulation run."""

    rigid_body: jax_md.rigid_body.RigidBody

    def slice(self, key: int | slice) -> "SimulatorTrajectory":
        """Slice the trajectory."""
        if isinstance(key, int):
            key = slice(key, key + 1)

        return self.replace(
            rigid_body=jax_md.rigid_body.RigidBody(
                center=self.rigid_body.center[key, ...],
                orientation=jax_md.rigid_body.Quaternion(
                    vec=self.rigid_body.orientation.vec[key, ...],
                ),
            )
        )

    def length(self) -> int:
        """Return the length of the trajectory.

        Note, that this may have been more natural to implement as the built-in
        __len__ method. However, the chex.dataclass decorator overrides that
        method to be compatabile with the abc.Mapping interface

        See here:
        https://github.com/google-deepmind/chex/blob/8af2c9e8a19f3a57d9bd283c2a34148aef952f60/chex/_src/dataclass.py#L50
        """
        return self.rigid_body.center.shape[0]
