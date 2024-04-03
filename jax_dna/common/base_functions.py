import pdb
import jax.numpy as jnp

from jax import jit



# Pairwise potentials

@jit
def v_fene(r, eps, r0, delt):
    """
    The vanilla FENE potential. Used in a smoothed FENE potential in oxDNA.
    """
    x = (r - r0)**2 / delt**2
    # Note: if `x` is too big, we will easily try to take the log of negatives, which will yield `nan`
    return -eps / 2.0 * jnp.log(1 - x)

@jit
def v_morse(r, eps, r0, a):
    x = -(r - r0) * a
    return eps * (1 - jnp.exp(x))**2

@jit
def v_harmonic(r, k, r0):
    return k / 2 * (r - r0)**2

@jit
def v_lj(r, eps, sigma):
    x = (sigma / r)**12 - (sigma / r)**6
    return 4 * eps * x

@jit
def v_mod(theta, a, theta0):
    return 1 - a*(theta - theta0)**2

@jit
def v_smooth(x, b, x_c):
    return b*(x_c - x)**2

# Functional forms

@jit
def f1(r, r_low, r_high, r_c_low, r_c_high, # thresholding/smoothing parameters
       eps, a, r0, r_c, # morse parameters
       b_low, b_high, # smoothing parameters
):

    oob = jnp.where((r_c_low < r) & (r < r_low),
                    eps * v_smooth(r, b_low, r_c_low),
                    jnp.where((r_high < r) & (r < r_c_high),
                              eps * v_smooth(r, b_high, r_c_high),
                              0.0))
    return jnp.where((r_low < r) & (r < r_high),
                     v_morse(r, eps, r0, a) - v_morse(r_c, eps, r0, a),
                     oob)

@jit
def f2(r, r_low, r_high, r_c_low, r_c_high, # thresholding/smoothing parameters
       k, r0, r_c, # harmonic parameters
       b_low, b_high # smoothing parameters
):
    oob = jnp.where((r_c_low < r) & (r < r_low),
                    k * v_smooth(r, b_low, r_c_low),
                    jnp.where((r_high < r) & (r < r_c_high),
                              k * v_smooth(r, b_high, r_c_high),
                              0.0))
    return jnp.where((r_low < r) & (r < r_high),
                     v_harmonic(r, k, r0) - v_harmonic(r_c, k, r0),
                     oob)

@jit
def f3(r, r_star, r_c, # thresholding/smoothing parameters
       eps, sigma, # lj parameters
       b # smoothing parameters
):
    oob = jnp.where((r_star < r) & (r < r_c),
                    eps * v_smooth(r, b, r_c),
                    0.0)
    return jnp.where(r < r_star,
                     v_lj(r, eps, sigma),
                     oob)

@jit
def f4(theta, theta0, delta_theta_star, delta_theta_c, # thresholding/smoothing parameters
       a, # mod parameters
       b # smoothing parameters
):
    oob = jnp.where((theta0 - delta_theta_c < theta) & (theta < theta0 - delta_theta_star),
                    v_smooth(theta, b, theta0 - delta_theta_c),
                    jnp.where((theta0 + delta_theta_star < theta) & (theta < theta0 + delta_theta_c),
                              v_smooth(theta, b, theta0 + delta_theta_c),
                              0.0))
    return jnp.where((theta0 - delta_theta_star < theta) & (theta < theta0 + delta_theta_star),
                     v_mod(theta, a, theta0),
                     oob)


# Note: for stacking, e.g. x = cos(phi)
@jit
def f5(x, x_star, x_c, # thresholding/smoothing parameters
       a, # mod parameters
       b # smoothing parameters
):
    return jnp.where(x > 0.0,
                     1.0,
                     jnp.where((x_star < x) & (x < 0.0),
                               v_mod(x, a, 0),
                               jnp.where((x_c < x) & (x < x_star),
                                         v_smooth(x, b, x_c),
                                         0.0)))


def f6(theta, a, b):
    cond = (theta >= b)
    val = a/2 * (theta - b)**2
    return jnp.where(cond, val, 0.0)

if __name__ == "__main__":
    pass
