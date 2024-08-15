"""Smoothing functions for the base functions in DNA1 model."""

import jax.numpy as jnp

import jax_dna.utils.types as typ


def _solve_f1_b(x: typ.Scalar, a: typ.Scalar, x0: typ.Scalar, xc: typ.Scalar) -> typ.Scalar:
    """Solve for the smoothing parameter b in the f1 smoothing function."""
    return (
        a**2
        * (
            -jnp.exp(a * (3 * x0 + 2 * xc))
            + 2 * jnp.exp(a * (x + 2 * x0 + 2 * xc))
            - jnp.exp(a * (2 * x + x0 + 2 * xc))
        )
        * jnp.exp(-2 * a * x)
        / (
            2 * jnp.exp(a * (x + 2 * xc))
            + jnp.exp(a * (2 * x + x0))
            - 2 * jnp.exp(a * (2 * x + xc))
            - jnp.exp(a * (x0 + 2 * xc))
        )
    )


def _solve_f1_xc_star(x: typ.Scalar, a: typ.Scalar, x0: typ.Scalar, xc: typ.Scalar) -> typ.Scalar:
    """Solve for the smoothing parameter xc_star in the f1 smoothing function."""
    return (
        (
            a * x * jnp.exp(a * (x + 2 * xc))
            - a * x * jnp.exp(a * (x0 + 2 * xc))
            + 2 * jnp.exp(a * (x + 2 * xc))
            + jnp.exp(a * (2 * x + x0))
            - 2 * jnp.exp(a * (2 * x + xc))
            - jnp.exp(a * (x0 + 2 * xc))
        )
        * jnp.exp(-2 * a * xc)
        / (a * (jnp.exp(a * x) - jnp.exp(a * x0)))
    )


def get_f1_smoothing_params(
    x0: typ.Scalar, a: typ.Scalar, xc: typ.Scalar, x_low: typ.Scalar, x_high: typ.Scalar
) -> typ.Tuple[typ.Scalar, typ.Scalar, typ.Scalar, typ.Scalar]:
    """Get the smoothing parameters for the f1 smoothing function."""
    solved_b_low = _solve_f1_b(x_low, a, x0, xc)
    solved_b_high = _solve_f1_b(x_high, a, x0, xc)

    solved_xc_low = _solve_f1_xc_star(x_low, a, x0, xc)
    solved_xc_high = _solve_f1_xc_star(x_high, a, x0, xc)

    return solved_b_low, solved_xc_low, solved_b_high, solved_xc_high


def _solve_f2_b(x: typ.Scalar, x0: typ.Scalar, xc: typ.Scalar) -> typ.Scalar:
    """Solve for the smoothing parameter b in the f2 smoothing function."""
    return (x - x0) ** 2 / (2 * (x - xc) * (x - 2 * x0 + xc))


def _solve_f2_xc_star(x: typ.Scalar, x0: typ.Scalar, xc: typ.Scalar) -> typ.Scalar:
    """Solve for the smoothing parameter xc_star in the f2 smoothing function."""
    return (x * x0 - 2 * x0 * xc + xc**2) / (x - x0)


def get_f2_smoothing_params(
    x0:typ.Scalar,
    xc:typ.Scalar,
    x_low:typ.Scalar,
    x_high:typ.Scalar
) -> typ.Tuple[typ.Scalar, typ.Scalar, typ.Scalar, typ.Scalar]:
    """Get the smoothing parameters for the f2 smoothing function."""
    solved_b_low = _solve_f2_b(x_low, x0, xc)
    solved_b_high = _solve_f2_b(x_high, x0, xc)

    solved_xc_low = _solve_f2_xc_star(x_low, x0, xc)
    solved_xc_high = _solve_f2_xc_star(x_high, x0, xc)

    return solved_b_low, solved_xc_low, solved_b_high, solved_xc_high


def _solve_f3_b(x: typ.Scalar, sigma: typ.Scalar) -> typ.Scalar:
    """Solve for the smoothing parameter b in the f3 smoothing function."""
    return (
        -36
        * sigma**6
        * (-2 * sigma**6 + x**6) ** 2
        / (x**14 * (-sigma + x) * (sigma + x) * (sigma**2 - sigma * x + x**2) * (sigma**2 + sigma * x + x**2))
    )


def _solve_f3_xc(x: typ.Scalar, sigma: typ.Scalar) -> typ.Scalar:
    """Solve for the smoothing parameter xc in the f3 smoothing function."""
    return x * (-7 * sigma**6 + 4 * x**6) / (3 * (-2 * sigma**6 + x**6))


def get_f3_smoothing_params(r_star: typ.Scalar, sigma:typ.Scalar) -> typ.Tuple[typ.Scalar, typ.Scalar]:
    """Get the smoothing parameters for the f3 smoothing function."""
    solved_b = _solve_f3_b(r_star, sigma)
    solved_xc = _solve_f3_xc(r_star, sigma)

    return solved_b, solved_xc


def _solve_f4_b(x: typ.Scalar, x0: typ.Scalar, a: typ.Scalar) -> typ.Scalar:
    """Solve for the smoothing parameter b in the f4 smoothing function."""
    return -(a**2) * (x - x0) ** 2 / (a * x**2 - 2 * a * x * x0 + a * x0**2 - 1)


def _solve_f4_xc(x: typ.Scalar, x0: typ.Scalar, a: typ.Scalar) -> typ.Scalar:
    """Solve for the smoothing parameter xc in the f4 smoothing function."""
    return (-a * x * x0 + a * x0**2 - 1) / (a * (-x + x0))


def get_f4_smoothing_params(
    a: typ.Scalar, x0: typ.Scalar, delta_x_star: typ.Scalar
) -> typ.Tuple[typ.Scalar, typ.Scalar]:
    """Get the smoothing parameters for the f4 smoothing function."""
    solved_b_plus = _solve_f4_b(x0 + delta_x_star, x0, a)

    solved_xc_plus = _solve_f4_xc(x0 + delta_x_star, x0, a)
    solved_delta_xc_plus = solved_xc_plus - x0

    return solved_b_plus, solved_delta_xc_plus


def _solve_f5_b(x: typ.Scalar, x0: typ.Scalar, a: typ.Scalar) -> typ.Scalar:
    """Solve for the smoothing parameter b in the f5 smoothing function."""
    return -(a**2) * (x - x0) ** 2 / (a * x**2 - 2 * a * x * x0 + a * x0**2 - 1)


def _solve_f5_xc(x: typ.Scalar, x0: typ.Scalar, a:typ.Scalar) -> typ.Scalar:
    """Solve for the smoothing parameter xc in the f5 smoothing function."""
    return (a * x * x0 - a * x0**2 + 1) / (a * (x - x0))


def get_f5_smoothing_params(a: typ.Scalar, x_star: typ.Scalar) -> typ.Tuple[typ.Scalar, typ.Scalar]:
    """Get the smoothing parameters for the f5 smoothing function."""
    solved_b = _solve_f5_b(x_star, 0.0, a)
    solved_xc = _solve_f5_xc(x_star, 0.0, a)

    return solved_b, solved_xc
