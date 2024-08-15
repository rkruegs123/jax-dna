"""Base energy functions for DNA2 model.

This function is based on the oxDNA2 model paper found here:
https://ora.ox.ac.uk/objects/uuid:241ae8d5-2092-4b24-b1d0-3fb7482b7bcd/files/m7422ee58d9747bbd7af00d6435b570e6
"""

import jax.numpy as jnp

import jax_dna.utils.types as typ
from jax_dna.energy.dna1.base_functions import f1, f2, f3, f4, f5


def f6(theta: typ.ARR_OR_SCALAR, a: typ.Scalar, b: typ.Scalar) -> typ.ARR_OR_SCALAR:
    """Replaces a coaxial stacking potential in the DNA1 model."""
    cond = theta >= b
    val = a / 2 * (theta - b) ** 2
    return jnp.where(cond, val, 0.0)


__all__ = ["f1", "f2", "f3", "f4", "f5", "f6"]
