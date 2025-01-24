"""Tests for the optimization simulator module."""

import jax_dna.optimization.simulator as jdna_simulator


def test_simulator_init():
    """Test the initialization of the SimulatorActor."""
    fn = lambda x, y: ("a", "b")  # noqa: ARG005 -- This is just for testing
    exposes = ["a", "b"]
    meta_data = {"a": 1, "b": 2}
    name = "test"

    simulator = jdna_simulator.BaseSimulator(name, fn, exposes, meta_data)

    assert simulator._fn(None, None) == fn(None, None)
    assert simulator._exposes == exposes
    assert simulator._meta_data == meta_data
