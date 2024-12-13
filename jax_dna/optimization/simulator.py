"""Simulation actors for use in an jax_dna.optimization.ray_optimization.Optimization loop."""

import typing

import ray

import jax_dna.utils.types as jdna_types


class BaseSimulator:
    """A base class for a simulator actor.

    The class is split this way to make testing easier.
    """

    def __init__(
        self,
        fn: typing.Callable[[jdna_types.Params, jdna_types.MetaData], tuple[str, ...]],
        exposes: list[str],
        meta_data: jdna_types.MetaData,
    ) -> "BaseSimulator":
        """Initializes a SimulatorActor.

        Args:
            fn: The simulation function to run.
            exposes: The list of observables exposed by the simulation.
            meta_data: The metadata to pass to the simulation function.
            writer_fn: A function that writes the simulation output to a file.
        """
        self._fn = fn
        self._exposes = exposes
        self._meta_data = meta_data

    def exposes(self) -> list[str]:
        """Returns the list of observables exposed by the simulation."""
        return self._exposes

    def meta_data(self) -> jdna_types.MetaData:
        """Returns the metadata used by the simulation."""
        return self._meta_data

    def run(
        self,
        params: jdna_types.Params,
    ) -> tuple[str, ...]:
        """Runs the simulation using the given params and returns the observables and metadata."""
        return self._fn(params, self._meta_data)


@ray.remote
class SimulatorActor(BaseSimulator):
    """A ray actor that runs a simulation and exposes observables.

    The simulator actor is wrapper around a simulator function so to be used in
    a jax_dna.optimization.ray_optimization.Optimization. Because a simulation
    trajectory and derived observables can be large, the simulation function
    should write the trajectory to a file and return the path to the file.
    """
