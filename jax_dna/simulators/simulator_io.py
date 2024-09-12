import chex
import jax_md


@chex.dataclass
class SimulatorTrajectory:
    """A trajectory of a simulation run."""

    rigid_body: jax_md.rigid_body.RigidBody
