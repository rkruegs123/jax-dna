import jax.numpy as jnp

import jax_dna.utils.types as jd_types

# https://ora.ox.ac.uk/objects/uuid:b2415bb2-7975-4f59-b5e2-8c022b4a3719/files/mdcac62bc9133143fc05070ed20048c50
# Section 2.4.1


def v_fene(
    r: jd_types.ARR_OR_SCALAR, eps: jd_types.Scalar, r0: jd_types.Scalar, delt: jd_types.Scalar
) -> jd_types.ARR_OR_SCALAR:
    """The FENE potential.

    This is based on equation 2.1 from the oxDNA paper.
    """
    x = (r - r0) ** 2 / delt**2
    return -eps / 2.0 * jnp.log(1 - x)


def v_morse(
    r: jd_types.ARR_OR_SCALAR,
    eps: jd_types.Scalar,
    r0: jd_types.Scalar,
    a: jd_types.Scalar,
) -> jd_types.ARR_OR_SCALAR:
    """The Morse potential.

    This is based on equation 2.2 from the oxDNA paper.
    """
    x = -(r - r0) * a
    return eps * (1 - jnp.exp(x)) ** 2


def v_harmonic(
    r: jd_types.ARR_OR_SCALAR,
    k: jd_types.Scalar,
    r0: jd_types.Scalar,
) -> jd_types.ARR_OR_SCALAR:
    """The Harmonic potential.

    This is based on equation 2.3 from the oxDNA paper.
    """
    return k / 2 * (r - r0) ** 2


def v_lj(r: jd_types.ARR_OR_SCALAR, eps: jd_types.Scalar, sigma: jd_types.Scalar) -> jd_types.ARR_OR_SCALAR:
    """The Lennard-Jones potential.

    This is based on equation 2.4 from the oxDNA paper.
    """
    x = (sigma / r) ** 12 - (sigma / r) ** 6
    return 4 * eps * x


def v_mod(theta: jd_types.Scalar, a: jd_types.Scalar, theta0: jd_types.Scalar):
    """The modified potential.

    This is based on equation 2.5 from the oxDNA paper."""
    return 1 - a * (theta - theta0) ** 2


def v_smooth(x: jd_types.ARR_OR_SCALAR, b: jd_types.Scalar, x_c: jd_types.Scalar) -> jd_types.ARR_OR_SCALAR:
    """The smooth potential.

    This is based on equation 2.6 from the oxDNA paper."""
    return b * (x_c - x) ** 2
