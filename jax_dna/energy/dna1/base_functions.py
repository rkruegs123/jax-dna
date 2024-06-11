import jax.numpy as jnp

import jax_dna.energy.potentials as jd_potentials


def f1(
    r,
    r_low,
    r_high,
    r_c_low,
    r_c_high,  # thresholding/smoothing parameters
    eps,
    a,
    r0,
    r_c,  # morse parameters
    b_low,
    b_high,  # smoothing parameters
):
    oob = jnp.where(
        (r_c_low < r) & (r < r_low),
        eps * jd_potentials.v_smooth(r, b_low, r_c_low),
        jnp.where((r_high < r) & (r < r_c_high), eps * jd_potentials.v_smooth(r, b_high, r_c_high), 0.0),
    )
    return jnp.where(
        (r_low < r) & (r < r_high), jd_potentials.v_morse(r, eps, r0, a) - jd_potentials.v_morse(r_c, eps, r0, a), oob
    )


def f2(
    r,
    r_low,
    r_high,
    r_c_low,
    r_c_high,  # thresholding/smoothing parameters
    k,
    r0,
    r_c,  # harmonic parameters
    b_low,
    b_high,  # smoothing parameters
):
    oob = jnp.where(
        (r_c_low < r) & (r < r_low),
        k * jd_potentials.v_smooth(r, b_low, r_c_low),
        jnp.where((r_high < r) & (r < r_c_high), k * jd_potentials.v_smooth(r, b_high, r_c_high), 0.0),
    )
    return jnp.where(
        (r_low < r) & (r < r_high), jd_potentials.v_harmonic(r, k, r0) - jd_potentials.v_harmonic(r_c, k, r0), oob
    )


def f3(
    # thresholding/smoothing parameters
    r,
    r_star,
    r_c,
    # lj parameters
    eps,
    sigma,
    # smoothing parameters
    b,
):
    oob = jnp.where((r_star < r) & (r < r_c), eps * jd_potentials.v_smooth(r, b, r_c), 0.0)
    return jnp.where(r < r_star, jd_potentials.v_lj(r, eps, sigma), oob)


def f4(
    theta,
    theta0,
    delta_theta_star,
    delta_theta_c,  # thresholding/smoothing parameters
    a,  # mod parameters
    b,  # smoothing parameters
):
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
    x,
    x_star,
    x_c,  # thresholding/smoothing parameters
    a,  # mod parameters
    b,  # smoothing parameters
):
    return jnp.where(
        x > 0.0,
        1.0,
        jnp.where(
            (x_star < x) & (x < 0.0),
            jd_potentials.v_mod(x, a, 0),
            jnp.where((x_c < x) & (x < x_star), jd_potentials.v_smooth(x, b, x_c), 0.0),
        ),
    )


def f6(theta, a, b):
    cond = theta >= b
    val = a / 2 * (theta - b) ** 2
    return jnp.where(cond, val, 0.0)
