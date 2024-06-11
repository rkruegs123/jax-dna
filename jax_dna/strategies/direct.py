from typing import Any, Callable

import jax
import jax.numpy as jnp


def optimize_structure(
    input_config: dict[str, Any],
) -> jnp.ndarray:
    pass


def optimize_energy_params(
    energy_params: dict[str, Any],
    energy_fn: Callable,
    input_config: dict[str, Any],
) -> dict[str, Any]:
    pass
