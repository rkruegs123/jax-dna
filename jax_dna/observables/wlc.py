"""Utility functions for computing worm-like chain (WLC) fit."""

import jax.numpy as jnp
from jaxopt import GaussNewton

import jax_dna.utils.types as jd_types


def coth(x: jd_types.ARR_OR_SCALAR) -> jd_types.ARR_OR_SCALAR:
    """Hyperbolic cotangent function."""
    return (jnp.exp(2 * x) + 1) / (jnp.exp(2 * x) - 1)


def calculate_extension(
    force: jd_types.ARR_OR_SCALAR,
    l0: jd_types.ARR_OR_SCALAR,
    lp: jd_types.ARR_OR_SCALAR,
    k: jd_types.ARR_OR_SCALAR,
    kT: float,  # noqa: N803 -- kT is a special unit variable
) -> jd_types.ARR_OR_SCALAR:
    r"""Computes the extension under a specified force under the wormlike chain (WLC) model.

    Via the model of Odijk, the extension of an extensible wormlike chain (WLC) under a force
    F can be computed as

    .. math::
      x = L_0 \left (1 + \frac{F}{K} - \frac{kT}{2F} [1 + y\coth y] \right)

    where

    .. math::
      y = \left( \frac{FL_0^2}{L_p kT} \right)^{1/2}

    where `L_0` is the contour length and `L_p` is the persistence length.
    This function computes implements this model for computing the extension.

    Args:
        force (jd_types.ARR_OR_SCALAR): the force applied to the duplex
        l0 (jd_types.ARR_OR_SCALAR): the contour length
        lp (jd_types.ARR_OR_SCALAR): the persistence length
        k (jd_types.ARR_OR_SCALAR): the extensional modulus
        kT (float): the temperature

    Returns:
        jd_types.ARR_OR_SCALAR: the predicted extension
    """
    y = ((force * l0**2) / (lp * kT)) ** (1 / 2)
    return l0 * (1 + force / k - kT / (2 * force * l0) * (1 + y * coth(y)))


def loss(
    coeffs: jnp.ndarray,
    extensions: jnp.ndarray,
    forces: jnp.ndarray,
    kT: float,  # noqa: N803 -- kT is a special unit variable
) -> jnp.ndarray:
    """An objective function for the WLC model compatible with JAX solvers.

    Args:
        coeffs (jnp.ndarray): The parameters of the WLC model, ordered as [L_0, L_p, K]
        extensions (jnp.ndarray): The measured extensions (via simulation) to which we are fitting the model
        forces (jnp.ndarray): The forces under which the extensions were measured
        kT (float): the temperature

    Returns:
        jnp.ndarray: the residual for each measured extension
    """
    # Extract the coefficients
    # Note: coefficients ordering: [L0, Lp, K]
    l0 = coeffs[0]
    lp = coeffs[1]
    k = coeffs[2]

    # Compute the extensions as predicted with the designated parameters
    extensions_calc = calculate_extension(forces, l0, lp, k, kT)

    # Compute the residuals with the measured extensions
    return extensions - extensions_calc


def fit_wlc(
    extensions: jnp.ndarray,
    forces: jnp.ndarray,
    init_guess: jnp.ndarray,
    kT: float,  # noqa: N803 -- kT is a special unit variable
    *,
    implicit_diff: bool = True,
) -> jnp.ndarray:
    """Fit the WLC model via nonlinear least squares given a set of forces and measured extensions.

    Args:
        extensions (jnp.ndarray): The measured extensions (via simulation) to which we are fitting the model
        forces (jnp.ndarray): The forces under which the extensions were measured
        init_guess (jnp.ndarray): An initial guess for the parameters of the WLC model, ordered as [L_0, L_p, K]
        kT (float): the temperature
        implicit_diff (bool): Whether or not to use implicit differentiation for the numerical solver

    Returns:
        jnp.ndarray: the fit parameters of the WLC model, ordered as [L_0, L_p, K]
    """
    gn = GaussNewton(residual_fun=loss, implicit_diff=implicit_diff)
    res = gn.run(init_guess, extensions=extensions, forces=forces, kT=kT)
    return res.params
