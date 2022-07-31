from functools import partial
import jax.numpy as jnp
import toml
from jax_md import energy
import pdb

from utils import get_params


PARAMS = get_params()

# Define individual potentials
# FIXME: Could use ones from JAX-MD when appropriate (e.g. morse, harmonic). Could just add ones to JAX-MD that are missing (e.g. FENE)

# FIXME: need some initial positions from Megan
def v_fene(r, eps=PARAMS["fene"]["eps_backbone"],
           r0=PARAMS["fene"]["r0_backbone"], delt=PARAMS["fene"]["delta_backbone"]):
    x = (r - r0)**2 / delt**2
    # Note: if `x` is too big, we will easily try to take the log of negatives, wihch will yield `nan`
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

def v_smooth(x, b, x_c):
    return b*(x_c - x)**2


# Define functional forms
# FIXME: Do cutoff with Carl's method. Likely don't need r_c_low and r_c_high
def f1(r, r_low, r_high, r_c_low, r_c_high, # thresholding/smoothing parameters
       eps, a, r0, r_c, # morse parameters
       b_low, b_high, # smoothing parameters
):

    return jnp.where(jnp.logical_and(jnp.less(r_low, r), jnp.less(r, r_high)),
                     v_morse(r, eps, r0, a) - v_morse(r_c, eps, r0, a),
                     jnp.where(jnp.logical_and(jnp.less(r_c_low, r), jnp.less(r, r_low)),
                               eps * v_smooth(r, b_low, r_c_low),
                               jnp.where(jnp.logical_and(jnp.less(r_high, r), jnp.less(r, r_c_high)),
                                         eps * v_smooth(r, b_high, r_c_high),
                                         0.0)))

    """
    if r_low < r and r < r_high:
        return v_morse(r, eps, r0, a) - v_morse(r_c, eps, r0, a)
    elif r_c_low < r and r < r_low:
        return eps * v_smooth(r, b_low, r_c_low)
    elif r_high < r and r < r_c_high:
        return eps * v_smooth(r, b_high, r_c_high)
    else:
        return 0.0
    """


def f2(r, r_low, r_high, r_c_low, r_c_high, # thresholding/smoothing parameters
       k, r0, r_c, # harmonic parameters
       b_low, b_high # smoothing parameters
):
    return jnp.where(jnp.logical_and(jnp.less(r_low, r), jnp.less(r, r_high)),
                     v_harmonic(r, k, r0) - v_harmonic(r_c, k, r0),
                     jnp.where(jnp.logical_and(jnp.less(r_c_low, r), jnp.less(r, r_low)),
                               k * v_smooth(r, b_low, r_c_low),
                               jnp.where(jnp.logical_and(jnp.less(r_high, r), jnp.less(r, r_c_high)),
                                         k * v_smooth(r, b_high, r_c_high),
                                         0.0)))

    """
    if r_low < r and r < r_high:
        return v_harmonic(r, k, r0) - v_harmonic(r_c, k, r0)
    elif r_c_low < r and r < r_low:
        return k * v_smooth(r, b_low, r_c_low)
    elif r_high < r and r < r_c_high:
        return k * v_smooth(r, b_high, r_c_high)
    else:
        return 0.0
    """


def get_f3(r_star, r_c,
           eps, sigma):
    return energy.multiplicative_isotropic_cutoff(lambda r: v_lj(r, eps, sigma),
                                                  r_onset=r_star, r_cutoff=r_c)

def f3(r, r_star, r_c, # thresholding/smoothing parameters
       eps, sigma, # lj parameters
       b # smoothing parameters
):
    return jnp.where(jnp.less(r, r_star),
                     v_lj(r, eps, sigma),
                     jnp.where(jnp.logical_and(jnp.less(r_star, r), jnp.less(r, r_c)),
                     # jnp.where(jnp.less(r_star, r) and jnp.less(r, r_c), # throws an error
                               eps * v_smooth(r, b, r_c),
                               jnp.zeros(r.shape[0])))
"""
# From Sam
return jnp.where(r < r_star,
v_lj(r, eps, sigma),
jnp.where(r < r_c, # note that other condition is implicit
eps * v_smooth(r, b r_c),
0))

# Note that if you *do* need a logical and, you can use `(r > r_star) & (r < r_c)`
# pythonic `and` always tries to coerce arguments into boolean
# Jax has overridden & -- it is (1) jittable, (2) element wise, and (3) doesn't force serial execution (and therefore GPU->CPU data transfer)
"""

"""
if r < r_star:
    return v_lj(r, eps, sigma)
elif r_star < r and r < r_c:
    return eps * v_smooth(r, b, r_c)
else:
    return 0.0
"""




def f4(theta, theta0, delta_theta_star, delta_theta_c, # thresholding/smoothing parameters
       a, # mod parameters
       b # smoothing parameters
):
    return jnp.where(jnp.logical_and(
        jnp.less(theta0 - delta_theta_star, theta),
        jnp.less(theta, theta0 + delta_theta_star)), v_mod(theta, a, theta0),
              jnp.where(jnp.logical_and(
                  jnp.less(theta0 - delta_theta_c, theta),
                  jnp.less(theta, theta0 - delta_theta_star)), v_smooth(theta, b, theta0 - delta_theta_c),
                  jnp.where(jnp.logical_and(
                      jnp.less(theta0 + delta_theta_star, theta),
                      jnp.less(theta, theta0 + delta_theta_c)), v_smooth(theta, b, theta0 + delta_theta_c),
                      0.0)))
    """
    if theta0 - delta_theta_star < theta and theta < theta0 + delta_theta_star:
        return v_mod(theta, a, theta0)
    elif theta0 - delta_theta_c < theta and theta < theta0 - delta_theta_star:
        return v_smooth(theta, b, theta0 - delta_theta_c)
    elif theta0 + delta_theta_star < theta and theta < theta0 + delta_theta_c:
        return v_smooth(theta, b, theta0 + delta_theta_c)
    else:
        return 0.0
    """

# FIXME: Confirm with megan that phi should be x in def of f5.
# Note: for stacking, e.g. x = cos(phi)
def f5(x, x_star, x_c, # thresholding/smoothing parameters
       a, # mod parameters
       b # smoothing parameters
):
    return jnp.where(jnp.greater(x, 0.0), 1.0,
                     jnp.where(jnp.logical_and(jnp.less(x_star, x), jnp.less(x, 0.0)),
                               v_mod(x, a, 0),
                               jnp.where(jnp.logical_and(jnp.less(x_c, x), jnp.less(x, x_star)),
                                         v_smooth(x, b, x_c), 0.0)))
    """
    if x > 0:
        return 1.0
    elif x_star < x and x < 0:
        return v_mod(x, a, 0)
    elif x_c < x and x < x_star:
        return v_smooth(x, b, x_c)
    else:
        return 0.0
    """



"""
f3_base = get_f3(r_star=PARAMS["excluded_volume"]["dr_star_base"],
                 r_c=PARAMS["excluded_volume"]["dr_c_base"],
                 eps=PARAMS["excluded_volume"]["eps_exc"],
                 sigma=PARAMS["excluded_volume"]["sigma_base"])
"""
f3_base = partial(f3,
                  r_star=PARAMS["excluded_volume"]["dr_star_base"],
                  r_c=PARAMS["excluded_volume"]["dr_c_base"],
                  eps=PARAMS["excluded_volume"]["eps_exc"],
                  sigma=PARAMS["excluded_volume"]["sigma_base"],
                  b=PARAMS["excluded_volume"]["b_base"])
f3_back_base = partial(f3,
                       r_star=PARAMS["excluded_volume"]["dr_star_back_base"],
                       r_c=PARAMS["excluded_volume"]["dr_c_back_base"],
                       eps=PARAMS["excluded_volume"]["eps_exc"],
                       sigma=PARAMS["excluded_volume"]["sigma_back_base"],
                       b=PARAMS["excluded_volume"]["b_back_base"])
f3_base_back = partial(f3,
                       r_star=PARAMS["excluded_volume"]["dr_star_base_back"],
                       r_c=PARAMS["excluded_volume"]["dr_c_base_back"],
                       eps=PARAMS["excluded_volume"]["eps_exc"],
                       sigma=PARAMS["excluded_volume"]["sigma_base_back"],
                       b=PARAMS["excluded_volume"]["b_base_back"])
def exc_vol_bonded(dr_base, dr_back_base, dr_base_back):

    r_base = jnp.linalg.norm(dr_base, axis=1)
    r_back_base = jnp.linalg.norm(dr_back_base, axis=1)
    r_base_back = jnp.linalg.norm(dr_base_back, axis=1)

    # FIXME: need to add rc's and b's
    # Note: r_c must be greater than r*

    term1 = f3_base(r_base)
    term2 = f3_back_base(r_back_base)
    term3 = f3_base_back(r_base_back)

    return term1 + term2 + term3


f1_dr_stack = partial(f1,
                      r_low=PARAMS["stacking"]["dr_low_stack"],
                      r_high=PARAMS["stacking"]["dr_high_stack"],
                      r_c_low=PARAMS["stacking"]["dr_c_low_stack"],
                      r_c_high=PARAMS["stacking"]["dr_c_high_stack"],
                      eps=PARAMS["stacking"]["eps_stack"],
                      a=PARAMS["stacking"]["a_stack"],
                      r0=PARAMS["stacking"]["dr0_stack"],
                      r_c=PARAMS["stacking"]["dr_c_stack"],
                      b_low=PARAMS["stacking"]["b_low_stack"],
                      b_high=PARAMS["stacking"]["b_high_stack"])
f4_theta_4 = partial(f4,
                     theta0=PARAMS["stacking"]["theta0_stack_4"],
                     delta_theta_star=PARAMS["stacking"]["delta_theta_star_stack_4"],
                     delta_theta_c=PARAMS["stacking"]["delta_theta_4_c"],
                     a=PARAMS["stacking"]["a_stack_4"],
                     b=PARAMS["stacking"]["b_theta_4"])
f4_theta_5p = partial(f4,
                      theta0=PARAMS["stacking"]["theta0_stack_5"],
                      delta_theta_star=PARAMS["stacking"]["delta_theta_star_stack_5"],
                      delta_theta_c=PARAMS["stacking"]["delta_theta_5_c"],
                      a=PARAMS["stacking"]["a_stack_5"],
                      b=PARAMS["stacking"]["b_theta_5"])
f4_theta_6p = partial(f4,
                      theta0=PARAMS["stacking"]["theta0_stack_6"],
                      delta_theta_star=PARAMS["stacking"]["delta_theta_star_stack_6"],
                      delta_theta_c=PARAMS["stacking"]["delta_theta_6_c"],
                      a=PARAMS["stacking"]["a_stack_6"],
                      b=PARAMS["stacking"]["b_theta_6"])
def stacking(dr_stack, theta4, theta5, theta6):
    # need dr_stack, theta_4, theta_5, theta_6, phi1, and phi2
    # theta_4: angle between base normal vectors
    # theta_5: angle between base normal and line passing throug stacking
    # theta_6: theta_5 but with the other base normal
    # note: for above, really just need dr_stack and base normals

    r_stack = jnp.linalg.norm(dr_stack, axis=1)

    term1 = f1_dr_stack(r_stack)



    # for the phi's, we also need dr_backbone and the cross product of the normal and backbone-base vectors
    # phi1:
    raise NotImplementedError



if __name__ == "__main__":
    pass
