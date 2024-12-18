"""Utility functions for computing stretch-torsion moduli."""

import chex
import jax
import jax.numpy as jnp
from jaxopt import GaussNewton
from typing import Tuple

import jax_dna.utils.math as jd_math
import jax_dna.utils.types as jd_types
import jax_dna.utils.units as jd_units



def stretch(
        forces: jnp.ndarray,
        extensions: jnp.ndarray
) -> Tuple[float, float, float]:
    """Computes the effective stretch modulus and relevant summary statistics from stretch experiments.

    Following Assenza and Perez (JCTC 2022), the effective stretch modulus can be computed as

    .. math::
      \\tilde{S} = \\frac{L_0}{A_1}

    where `A_1` and `L_0` are the slope and offset, respectively, of a linear force-extension fit.

    Args:
        forces (jnp.ndarray): the forces applied to the polymer
        extensions (jnp.ndarray): the equilibrium extensions under the applied forces

    Returns:
        Tuple[float, float, float]: the slope and offset of the linear fit, and the effective stretch modulus
    """

    # Format the forces for line-fitting
    forces_ = jnp.stack([jnp.ones_like(forces), forces], axis=1)

    # Fit a line
    # Note: we do not fix l0 to be the extension under 0 force. We fit it as a parameter.
    fit_ = jnp.linalg.lstsq(forces_, extensions)

    # Extract statistics
    a1 = fit_[0][1]
    l0 = fit_[0][0] # Note: this is the equilibrium extension at 0 force and torque, *not* the contour length

    # Compute effective stretch modulus
    s_eff = l0 / a1
    return a1, l0, s_eff


def torsion(
        torques: jnp.ndarray,
        extensions: jnp.ndarray,
        twists: jnp.ndarray
) -> Tuple[float, float]:
    """Computes the relevant summary statistics from torsion experiments.

    Following Assenza and Perez (JCTC 2022), the torsional modulus and twist-stretch coupling can be
    computed via linear fits to the extension and twist of a duplex under torque (when combined with
    similar statistics from stretching experiments). This function computes the slopes of these
    linear fits

    Args:
        torques (jnp.ndarray): the torques applied to the polymer
        extensions (jnp.ndarray): the equilibrium extensions under the applied torques
        twists (jnp.ndarray): the equilibrium twists under the applied torques

    Returns:
        Tuple[float, float]: the slopes of the linear fits to the extensions and twists, respectively
    """

    # Format the torques for line-fitting
    torques_ = jnp.stack([jnp.ones_like(torques), torques], axis=1)

    # Fit a line to the extensions
    fit_ = jnp.linalg.lstsq(torques_, extensions)
    a3 = fit_[0][1]

    # Fit a line to the twists
    fit_ = jnp.linalg.lstsq(torques_, twists)
    a4 = fit_[0][1]

    return a3, a4

def stretch_torsion(
        forces: jnp.ndarray,
        force_extensions: jnp.ndarray,
        torques: jnp.ndarray,
        torque_extensions: jnp.ndarray,
        torque_twists: jnp.ndarray,
) -> Tuple[float, float, float]:
    """Computes the effective stretch modulus, torsional modulus, and twist-stretch coupling from stretch-torsion experiments.

    Args:
        forces (jnp.ndarray): the forces applied to the polymer
        force_extensions (jnp.ndarray): the equilibrium extensions under the applied forces
        torques (jnp.ndarray): the torques applied to the polymer
        torque_extensions (jnp.ndarray): the equilibrium extensions under the applied torques
        torque_twists (jnp.ndarray): the equilibrium twists under the applied torques

    Returns:
        Tuple[float, float, float]: the effective stretch modulus, torsional modulus, and twist-stretch coupling
    """

    # Compute the effective stretch modulus and relevant summary statistics from stretching experiments
    a1, l0, s_eff = stretch(forces, force_extensions)

    # Compute the relevant summary statistics from torsion experiments
    a3, a4 = torsion(torques, torque_extensions, torque_twists)

    # Compute the torsional modulus and twist-stretch coupling
    c = a1 * l0 / (a4*a1 - a3**2)
    g = -(a3 * l0) / (a4 * a1 - a3**2)

    return s_eff, c, g
