"""Base energy functions for DNA1 model.

These functions are based on the oxDNA1 model paper found here:
https://ora.ox.ac.uk/objects/uuid:b2415bb2-7975-4f59-b5e2-8c022b4a3719/files/mdcac62bc9133143fc05070ed20048c50
"""

import jax.numpy as jnp

import jax_dna.energy.potentials as jd_potentials
import jax_dna.utils.types as typ


def f1(
    r: typ.ARR_OR_SCALAR,
    r_low: typ.Scalar,
    r_high: typ.Scalar,
    r_c_low: typ.Scalar,
    r_c_high: typ.Scalar,
    eps: typ.Scalar,
    a: typ.Scalar,
    r0: typ.Scalar,
    r_c: typ.Scalar,
    b_low: typ.Scalar,
    b_high: typ.Scalar,
) -> typ.ARR_OR_SCALAR:
    """The radial part of the stacking and hydrogen-bonding potentials.

    This is based on equation 2.7 from the oxDNA paper.
    """
    oob = jnp.where(
        (r_c_low < r) & (r < r_low),
        eps * jd_potentials.v_smooth(r, b_low, r_c_low),
        jnp.where((r_high < r) & (r < r_c_high), eps * jd_potentials.v_smooth(r, b_high, r_c_high), 0.0),
    )
    return jnp.where(
        (r_low < r) & (r < r_high), jd_potentials.v_morse(r, eps, r0, a) - jd_potentials.v_morse(r_c, eps, r0, a), oob
    )


def f2(
    r: typ.ARR_OR_SCALAR,
    r_low: typ.Scalar,
    r_high: typ.Scalar,
    r_c_low: typ.Scalar,
    r_c_high: typ.Scalar,
    k: typ.Scalar,
    r0: typ.Scalar,
    r_c: typ.Scalar,
    b_low: typ.Scalar,
    b_high: typ.Scalar,
) -> typ.ARR_OR_SCALAR:
    """The radial part of the cross-stacking and coaxial stacking potentials.

    This is based on equation 2.8 from the oxDNA paper.
    """
    oob = jnp.where(
        (r_c_low < r) & (r < r_low),
        k * jd_potentials.v_smooth(r, b_low, r_c_low),
        jnp.where((r_high < r) & (r < r_c_high), k * jd_potentials.v_smooth(r, b_high, r_c_high), 0.0),
    )
    return jnp.where(
        (r_low < r) & (r < r_high), jd_potentials.v_harmonic(r, k, r0) - jd_potentials.v_harmonic(r_c, k, r0), oob
    )


def f3(
    r: typ.ARR_OR_SCALAR,
    r_star: typ.Scalar,
    r_c: typ.Scalar,
    eps: typ.Scalar,
    sigma: typ.Scalar,
    b: typ.Scalar,
) -> typ.ARR_OR_SCALAR:
    """The radial part of the excluded volume potential.

    This is based on equation 2.9 from the oxDNA paper.
    """
    oob = jnp.where((r_star < r) & (r < r_c), eps * jd_potentials.v_smooth(r, b, r_c), 0.0)
    return jnp.where(r < r_star, jd_potentials.v_lj(r, eps, sigma), oob)


def f4(
    theta: typ.ARR_OR_SCALAR,
    theta0: typ.Scalar,
    delta_theta_star: typ.Scalar,
    delta_theta_c: typ.Scalar,
    a: typ.Scalar,
    b: typ.Scalar,
) -> typ.ARR_OR_SCALAR:
    """The angular modulation factor used in stacking, hydrogen-bonding, cross-stacking and coaxial stacking.

    This is based on equation 2.10 from the oxDNA paper.
    """
    oob = jnp.where(
        (theta0 - delta_theta_c < theta) & (theta < theta0 - delta_theta_star),
        jd_potentials.v_smooth(theta, b, theta0 - delta_theta_c),
        jnp.where(
            (theta0 + delta_theta_star < theta) & (theta < theta0 + delta_theta_c),
            jd_potentials.v_smooth(theta, b, theta0 + delta_theta_c),
            0.0,
        ),
    )
    return jnp.where(
        (theta0 - delta_theta_star < theta) & (theta < theta0 + delta_theta_star),
        jd_potentials.v_mod(theta, a, theta0),
        oob,
    )


def f5(
    x: typ.ARR_OR_SCALAR,
    x_star: typ.Scalar,
    x_c: typ.Scalar,
    a: typ.Scalar,
    b: typ.Scalar,
) -> typ.ARR_OR_SCALAR:
    """Another modulating term which is used to impose right-handedness (effectively a one-sided modulation).

    This is based on equation 2.11 from the oxDNA paper.
    """
    return jnp.where(
        x > 0.0,
        1.0,
        jnp.where(
            (x_star < x) & (x < 0.0),
            jd_potentials.v_mod(x, a, 0),
            jnp.where((x_c < x) & (x < x_star), jd_potentials.v_smooth(x, b, x_c), 0.0),
        ),
    )
