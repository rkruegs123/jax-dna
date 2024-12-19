"""Base class for a simulation."""

import chex

import jax_dna.simulators.io as jd_sio


@chex.dataclass(frozen=True)
class BaseSimulation:
    """Base class for a simulation."""

    def run(self, *args, **kwargs) -> jd_sio.SimulatorTrajectory:
        """Run the simulation."""
