"""Optimization directly through the simulation function."""

from collections.abc import Callable


def get_gradients(
    opt_params: dict[str, float],
    sim_fn: Callable,
    sim_fn_kwargs: dict[str, float],
) -> dict[str, float]:
    """Get the gradients for the current configuration."""
    pass
