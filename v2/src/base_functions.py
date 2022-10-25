import pdb
import jax.numpy as jnp
import numpy as np
from functools import wraps
from typing import Callable, Tuple

from jax_md import util

f32 = util.f32
f64 = util.f64
Array = util.Array


# Pairwise potentials

def _v_fene(r, eps, r0, delt): # Note: named as helper as we make `v_fene` the name of the parameterized potential
    x = (r - r0)**2 / delt**2
    # Note: if `x` is too big, we will easily try to take the log of negatives, which will yield `nan`
    return -eps / 2.0 * jnp.log(1 - x)

def v_morse(r, eps, r0, a):
    x = -(r - r0) * a
    return eps * (1 - jnp.exp(x))**2

def v_harmonic(r, k, r0):
    return k / 2 * (r - r0)**2

def v_lj(r, eps, sigma):
    x = (sigma / r)**12 - (sigma / r)**6
    return 4 * eps * x

def v_mod(theta, a, theta0):
    return 1 - a*(theta - theta0)**2



# Functional forms
def f1(r, r_low, r_high, r_c_low, r_c_high, # thresholding/smoothing parameters
       eps, a, r0, r_c, # morse parameters
       b_low, b_high, # smoothing parameters
):

    oob = jnp.where(r_c_low < r & r < r_low,
                    eps * v_smooth(r, b_low, r_c_low),
                    jnp.where(r_high < r & r < r_c_high,
                              eps * v_smooth(r, b_high, r_c_high),
                              0.0))
    return jnp.where(r_low < r & r < r_high,
                     v_morse(r, eps, r0, a) - v_morse(r_c, eps, r0, a),
                     oob)


def f2(r, r_low, r_high, r_c_low, r_c_high, # thresholding/smoothing parameters
       k, r0, r_c, # harmonic parameters
       b_low, b_high # smoothing parameters
):
    oob = jnp.where(r_c_low < r & r < r_low,
                    k * v_smooth(r, b_low, r_c_low),
                    jnp.where(r_high < r & r < r_c_high,
                              k * v_smooth(r, b_high, r_c_high),
                              0.0))
    return jnp.where(r_low < r & r < r_high,
                     v_harmonic(r, k, r0) - v_harmonic(r_c, k, r0),
                     oob)


def f3(r, r_star, r_c, # thresholding/smoothing parameters
       eps, sigma, # lj parameters
       b # smoothing parameters
):
    oob = jnp.where(r_star < r & r < r_c,
                    eps * v_smooth(r, b, r_c),
                    0.0)
    return jnp.where(r < r_star,
                     v_lj(r, eps, sigma),
                     oob)

def f4(theta, theta0, delta_theta_star, delta_theta_c, # thresholding/smoothing parameters
       a, # mod parameters
       b # smoothing parameters
):
    oob = jnp.where(theta0 - delta_theta_c < theta & theta < theta0 - delta_theta_star,
                    v_smooth(theta, b, theta0 - delta_theta_c),
                    jnp.where(theta0 + delta_theta_star < theta & theta < theta0 + delta_theta_c,
                              v_smooth(theta, b, theta0 + delta_theta_c),
                              0.0))
    return jnp.where(theta0 - delta_theta_star < theta & theta < theta0 + delta_theta_star,
                     v_mod(theta, a, theta0),
                     oob)


# Note: for stacking, e.g. x = cos(phi)
def f5(x, x_star, x_c, # thresholding/smoothing parameters
       a, # mod parameters
       b # smoothing parameters
):
    return jnp.where(x > 0.0,
                     1.0,
                     jnp.where(x_star < x & x < 0.0,
                               v_mod(x, a, 0),
                               jnp.where(x_c < x & x < x_star,
                                         v_smooth(x, b, x_c),
                                         0.0)))



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import get_params


    params = get_params.get_default_params()

    # Check FENE
    fene_eps = params['fene']['eps_backbone']
    fene_delta = params['fene']['delta_backbone']
    fene_r0 = params['fene']['r0_backbone']
    rs = np.linspace(0, 5)
    unsmoothed = [_v_fene(r, fene_eps, fene_r0, fene_delta) for r in rs]
    smoothed = [v_fene(r, fene_eps, fene_r0, fene_delta) for r in rs]

    overlapping = 0.250
    plt.plot(rs, smoothed, label="smoothed", alpha=overlapping, lw=2)
    plt.plot(rs, unsmoothed, label="unsmoothed", alpha=overlapping, lw=2)
    plt.show()
