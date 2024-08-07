import jax.numpy as jnp

import jax_dna.utils.types as jd_types

# https://ora.ox.ac.uk/objects/uuid:b2415bb2-7975-4f59-b5e2-8c022b4a3719/files/mdcac62bc9133143fc05070ed20048c50
# Section 2.4.1


def v_fene(r: jd_types.ARR_OR_SCALAR, eps: float, r0: float, delt: float) -> jd_types.ARR_OR_SCALAR:
    """The FENE potential.

    This is based on equation 2.1 from the oxDNA paper.
    """
    x = (r - r0) ** 2 / delt**2
    return -eps / 2.0 * jnp.log(1 - x)


def v_morse(
    r: jd_types.ARR_OR_SCALAR,
    eps: float,
    r0: float,
    a: float,
) -> jd_types.ARR_OR_SCALAR:
    """The Morse potential.

    This is based on equation 2.2 from the oxDNA paper.
    """
    x = -(r - r0) * a
    return eps * (1 - jnp.exp(x)) ** 2


def v_harmonic(
    r: jd_types.ARR_OR_SCALAR,
    k: float,
    r0: float,
) -> jd_types.ARR_OR_SCALAR:
    """The Harmonic potential.

    This is based on equation 2.3 from the oxDNA paper.
    """
    return k / 2 * (r - r0) ** 2


def v_lj(r, eps, sigma):
    x = (sigma / r) ** 12 - (sigma / r) ** 6
    return 4 * eps * x


def v_mod(theta, a, theta0):
    return 1 - a * (theta - theta0) ** 2


def v_smooth(x, b, x_c):
    return b * (x_c - x) ** 2
