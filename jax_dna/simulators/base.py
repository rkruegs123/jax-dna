"""Base class for a simulation."""

import jax_dna.simulators.io as jd_sio


class BaseSimulation:
    """Base class for a simulation."""

    def run(self, *args, **kwargs) -> jd_sio.SimulatorTrajectory:
        """Run the simulation."""

    def update(self, *args, **kwargs) -> None:
        """Update the simulation."""
