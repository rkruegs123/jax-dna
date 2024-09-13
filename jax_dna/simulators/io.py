import chex
import jax.numpy as jnp
import jax_md


@chex.dataclass()
class SimulatorTrajectory:
    """A trajectory of a simulation run."""
    seq_oh: jnp.ndarray
    strand_lengths: list[int]
    rigid_body: jax_md.rigid_body.RigidBody

    def slice(self, key: int | slice) -> "SimulatorTrajectory":
        """Slice the trajectory."""
        return self.replace(
            rigid_body=self.rigid_body[key],
        )
