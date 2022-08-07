from functools import partial
import jax.numpy as jnp
import toml
from jax_md import energy
import pdb

from utils import get_params


# FIXME: pass around better
TEMP = 300 # Kelvin
PARAMS = get_params(t=TEMP)
FENE_PARAMS = PARAMS["fene"]
EXC_VOL_PARAMS = PARAMS["excluded_volume"]
STACK_PARAMS = PARAMS["stacking"]
HB_PARAMS = PARAMS["hydrogen_bonding"]
CROSS_PARAMS = PARAMS["cross_stacking"]
COAX_PARAMS = PARAMS["coaxial_stacking"]


# Define individual potentials
# FIXME: Could use ones from JAX-MD when appropriate (e.g. morse, harmonic). Could just add ones to JAX-MD that are missing (e.g. FENE)

# FIXME: need some initial positions from Megan
def _v_fene(r, eps=FENE_PARAMS["eps_backbone"],
           r0=FENE_PARAMS["r0_backbone"], delt=FENE_PARAMS["delta_backbone"]):
    x = (r - r0)**2 / delt**2
    # Note: if `x` is too big, we will easily try to take the log of negatives, wihch will yield `nan`
    return -eps / 2.0 * jnp.log(1 - x)

def v_fene(r):

    # return _v_fene(r)
    # Thresholded version -- analogous to allowing broken bakcbone from oxDNA
    rbackr0 = r - FENE_PARAMS["r0_backbone"]

    return jnp.where(jnp.abs(rbackr0) >= FENE_PARAMS["delta_backbone"],
                     1.0e12,
                     _v_fene(r))


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
f3_base = get_f3(r_star=EXC_VOL_PARAMS["dr_star_base"],
                 r_c=EXC_VOL_PARAMS["dr_c_base"],
                 eps=EXC_VOL_PARAMS["eps_exc"],
                 sigma=EXC_VOL_PARAMS["sigma_base"])
"""
f3_base_exc_vol = partial(f3,
                          r_star=EXC_VOL_PARAMS["dr_star_base"],
                          r_c=EXC_VOL_PARAMS["dr_c_base"],
                          eps=EXC_VOL_PARAMS["eps_exc"],
                          sigma=EXC_VOL_PARAMS["sigma_base"],
                          b=EXC_VOL_PARAMS["b_base"])
f3_back_base_exc_vol = partial(f3,
                               r_star=EXC_VOL_PARAMS["dr_star_back_base"],
                               r_c=EXC_VOL_PARAMS["dr_c_back_base"],
                               eps=EXC_VOL_PARAMS["eps_exc"],
                               sigma=EXC_VOL_PARAMS["sigma_back_base"],
                               b=EXC_VOL_PARAMS["b_back_base"])
f3_base_back_exc_vol = partial(f3,
                               r_star=EXC_VOL_PARAMS["dr_star_base_back"],
                               r_c=EXC_VOL_PARAMS["dr_c_base_back"],
                               eps=EXC_VOL_PARAMS["eps_exc"],
                               sigma=EXC_VOL_PARAMS["sigma_base_back"],
                               b=EXC_VOL_PARAMS["b_base_back"])
def exc_vol_bonded(dr_base, dr_back_base, dr_base_back):
    # Note: r_c must be greater than r*
    r_base = jnp.linalg.norm(dr_base, axis=1)
    r_back_base = jnp.linalg.norm(dr_back_base, axis=1)
    r_base_back = jnp.linalg.norm(dr_base_back, axis=1)

    return f3_base_exc_vol(r_base) + f3_back_base_exc_vol(r_back_base) \
        + f3_base_back_exc_vol(r_base_back)

f3_back_exc_vol = partial(f3,
                          r_star=EXC_VOL_PARAMS["dr_star_backbone"],
                          r_c=EXC_VOL_PARAMS["dr_c_backbone"],
                          eps=EXC_VOL_PARAMS["eps_exc"],
                          sigma=EXC_VOL_PARAMS["sigma_backbone"],
                          b=EXC_VOL_PARAMS["b_backbone"])

# FIXME: lots of duplicated computation with exc_vol_bonded
# Should (a) compute the r's outside these functions, and (b) have a base that we then add on to
def exc_vol_unbonded(dr_base, dr_backbone, dr_back_base, dr_base_back):
    r_back = jnp.linalg.norm(dr_backbone, axis=1)
    r_base = jnp.linalg.norm(dr_base, axis=1)
    r_back_base = jnp.linalg.norm(dr_back_base, axis=1)
    r_base_back = jnp.linalg.norm(dr_base_back, axis=1)
    return f3_back_exc_vol(r_back) + f3_base_exc_vol(r_base) \
        + f3_back_base_exc_vol(r_back_base) + f3_base_back_exc_vol(r_base_back)

f1_dr_stack = partial(f1,
                      r_low=STACK_PARAMS["dr_low_stack"],
                      r_high=STACK_PARAMS["dr_high_stack"],
                      r_c_low=STACK_PARAMS["dr_c_low_stack"],
                      r_c_high=STACK_PARAMS["dr_c_high_stack"],
                      eps=STACK_PARAMS["eps_stack"],
                      a=STACK_PARAMS["a_stack"],
                      r0=STACK_PARAMS["dr0_stack"],
                      r_c=STACK_PARAMS["dr_c_stack"],
                      b_low=STACK_PARAMS["b_low_stack"],
                      b_high=STACK_PARAMS["b_high_stack"])
f4_theta_4_stack = partial(f4,
                           theta0=STACK_PARAMS["theta0_stack_4"],
                           delta_theta_star=STACK_PARAMS["delta_theta_star_stack_4"],
                           delta_theta_c=STACK_PARAMS["delta_theta_4_c"],
                           a=STACK_PARAMS["a_stack_4"],
                           b=STACK_PARAMS["b_theta_4"])
f4_theta_5p_stack = partial(f4,
                            theta0=STACK_PARAMS["theta0_stack_5"],
                            delta_theta_star=STACK_PARAMS["delta_theta_star_stack_5"],
                            delta_theta_c=STACK_PARAMS["delta_theta_5_c"],
                            a=STACK_PARAMS["a_stack_5"],
                            b=STACK_PARAMS["b_theta_5"])
f4_theta_6p_stack = partial(f4,
                            theta0=STACK_PARAMS["theta0_stack_6"],
                            delta_theta_star=STACK_PARAMS["delta_theta_star_stack_6"],
                            delta_theta_c=STACK_PARAMS["delta_theta_6_c"],
                            a=STACK_PARAMS["a_stack_6"],
                            b=STACK_PARAMS["b_theta_6"])
f5_neg_cosphi1_stack = partial(f5,
                               x_star=STACK_PARAMS["neg_cos_phi1_star_stack"],
                               x_c=STACK_PARAMS["neg_cos_phi1_c"],
                               a=STACK_PARAMS["a_stack_1"],
                               b=STACK_PARAMS["b_neg_cos_phi1"])
f5_neg_cosphi2_stack = partial(f5,
                               x_star=STACK_PARAMS["neg_cos_phi2_star_stack"],
                               x_c=STACK_PARAMS["neg_cos_phi2_c"],
                               a=STACK_PARAMS["a_stack_2"],
                               b=STACK_PARAMS["b_neg_cos_phi2"])
def stacking(dr_stack, theta4, theta5, theta6, cosphi1, cosphi2):
    # need dr_stack, theta_4, theta_5, theta_6, phi1, and phi2
    # theta_4: angle between base normal vectors
    # theta_5: angle between base normal and line passing throug stacking
    # theta_6: theta_5 but with the other base normal
    # note: for above, really just need dr_stack and base normals

    r_stack = jnp.linalg.norm(dr_stack, axis=1)

    return f1_dr_stack(r_stack) * f4_theta_4_stack(theta4) \
        * f4_theta_5p_stack(theta5) * f4_theta_6p_stack(theta6) \
        * f5_neg_cosphi1_stack(-cosphi1) * f5_neg_cosphi2_stack(-cosphi2)


f1_dr_hb = partial(f1,
                   r_low=HB_PARAMS["dr_low_hb"],
                   r_high=HB_PARAMS["dr_high_hb"],
                   r_c_low=HB_PARAMS["dr_c_low_hb"],
                   r_c_high=HB_PARAMS["dr_c_high_hb"],
                   eps=HB_PARAMS["eps_hb"],
                   a=HB_PARAMS["a_hb"],
                   r0=HB_PARAMS["dr0_hb"],
                   r_c=HB_PARAMS["dr_c_hb"],
                   b_low=HB_PARAMS["b_low_hb"],
                   b_high=HB_PARAMS["b_high_hb"])
f4_theta_1_hb = partial(f4,
                        theta0=HB_PARAMS["theta0_hb_1"],
                        delta_theta_star=HB_PARAMS["delta_theta_star_hb_1"],
                        delta_theta_c=HB_PARAMS["delta_theta_1_c"],
                        a=HB_PARAMS["a_hb_1"],
                        b=HB_PARAMS["b_theta_1"])
f4_theta_2_hb = partial(f4,
                        theta0=HB_PARAMS["theta0_hb_2"],
                        delta_theta_star=HB_PARAMS["delta_theta_star_hb_2"],
                        delta_theta_c=HB_PARAMS["delta_theta_2_c"],
                        a=HB_PARAMS["a_hb_2"],
                        b=HB_PARAMS["b_theta_2"])
f4_theta_3_hb = partial(f4,
                        theta0=HB_PARAMS["theta0_hb_3"],
                        delta_theta_star=HB_PARAMS["delta_theta_star_hb_3"],
                        delta_theta_c=HB_PARAMS["delta_theta_3_c"],
                        a=HB_PARAMS["a_hb_3"],
                        b=HB_PARAMS["b_theta_3"])
f4_theta_4_hb = partial(f4,
                        theta0=HB_PARAMS["theta0_hb_4"],
                        delta_theta_star=HB_PARAMS["delta_theta_star_hb_4"],
                        delta_theta_c=HB_PARAMS["delta_theta_4_c"],
                        a=HB_PARAMS["a_hb_4"],
                        b=HB_PARAMS["b_theta_4"])
f4_theta_7_hb = partial(f4,
                        theta0=HB_PARAMS["theta0_hb_7"],
                        delta_theta_star=HB_PARAMS["delta_theta_star_hb_7"],
                        delta_theta_c=HB_PARAMS["delta_theta_7_c"],
                        a=HB_PARAMS["a_hb_7"],
                        b=HB_PARAMS["b_theta_7"])
f4_theta_8_hb = partial(f4,
                        theta0=HB_PARAMS["theta0_hb_8"],
                        delta_theta_star=HB_PARAMS["delta_theta_star_hb_8"],
                        delta_theta_c=HB_PARAMS["delta_theta_8_c"],
                        a=HB_PARAMS["a_hb_8"],
                        b=HB_PARAMS["b_theta_8"])
def hydrogen_bonding(dr_hb, theta1, theta2, theta3, theta4, theta7, theta8):
    r_hb = jnp.linalg.norm(dr_hb, axis=1)
    return f1_dr_hb(r_hb) * f4_theta_1_hb(theta1) * f4_theta_2_hb(theta2) \
        * f4_theta_3_hb(theta3) * f4_theta_4_hb(theta4) * f4_theta_7_hb(theta7) \
        * f4_theta_8_hb(theta8)


# Cross Stacking
f2_dr_cross = partial(f2,
                      r_low=CROSS_PARAMS["dr_low_cross"],
                      r_high=CROSS_PARAMS["dr_high_cross"],
                      r_c_low=CROSS_PARAMS["dr_c_low_cross"],
                      r_c_high=CROSS_PARAMS["dr_c_high_cross"],
                      k=CROSS_PARAMS["k"],
                      r0=CROSS_PARAMS["r0_cross"],
                      r_c=CROSS_PARAMS["dr_c_cross"],
                      b_low=CROSS_PARAMS["b_low_cross"],
                      b_high=CROSS_PARAMS["b_high_cross"])
f4_theta_1_cross = partial(f4,
                           theta0=CROSS_PARAMS["theta0_cross_1"],
                           delta_theta_star=CROSS_PARAMS["delta_theta_star_cross_1"],
                           delta_theta_c=CROSS_PARAMS["delta_theta_1_c"],
                           a=CROSS_PARAMS["a_cross_1"],
                           b=CROSS_PARAMS["b_theta_1"])
f4_theta_2_cross = partial(f4,
                           theta0=CROSS_PARAMS["theta0_cross_2"],
                           delta_theta_star=CROSS_PARAMS["delta_theta_star_cross_2"],
                           delta_theta_c=CROSS_PARAMS["delta_theta_2_c"],
                           a=CROSS_PARAMS["a_cross_2"],
                           b=CROSS_PARAMS["b_theta_2"])
f4_theta_3_cross = partial(f4,
                           theta0=CROSS_PARAMS["theta0_cross_3"],
                           delta_theta_star=CROSS_PARAMS["delta_theta_star_cross_3"],
                           delta_theta_c=CROSS_PARAMS["delta_theta_3_c"],
                           a=CROSS_PARAMS["a_cross_3"],
                           b=CROSS_PARAMS["b_theta_3"])
f4_theta_4_cross = partial(f4,
                           theta0=CROSS_PARAMS["theta0_cross_4"],
                           delta_theta_star=CROSS_PARAMS["delta_theta_star_cross_4"],
                           delta_theta_c=CROSS_PARAMS["delta_theta_4_c"],
                           a=CROSS_PARAMS["a_cross_4"],
                           b=CROSS_PARAMS["b_theta_4"])
f4_theta_7_cross = partial(f4,
                           theta0=CROSS_PARAMS["theta0_cross_7"],
                           delta_theta_star=CROSS_PARAMS["delta_theta_star_cross_7"],
                           delta_theta_c=CROSS_PARAMS["delta_theta_7_c"],
                           a=CROSS_PARAMS["a_cross_7"],
                           b=CROSS_PARAMS["b_theta_7"])
f4_theta_8_cross = partial(f4,
                           theta0=CROSS_PARAMS["theta0_cross_8"],
                           delta_theta_star=CROSS_PARAMS["delta_theta_star_cross_8"],
                           delta_theta_c=CROSS_PARAMS["delta_theta_8_c"],
                           a=CROSS_PARAMS["a_cross_8"],
                           b=CROSS_PARAMS["b_theta_8"])

def cross_stacking(dr_hb, theta1, theta2, theta3, theta4, theta7, theta8):
    r_hb = jnp.linalg.norm(dr_hb, axis=1) # FIXME: repeated computation. May just want to do in the energy function and pass along
    return f2_dr_cross(r_hb) * f4_theta_1_cross(theta1) \
        * f4_theta_2_cross(theta2) * f4_theta_3_cross(theta3) \
        * (f4_theta_4_cross(theta4) + f4_theta_4_cross(jnp.pi - theta4)) \
        * (f4_theta_7_cross(theta7) + f4_theta_7_cross(jnp.pi - theta7)) \
        * (f4_theta_8_cross(theta8) + f4_theta_8_cross(jnp.pi - theta8))


# Coaxial stacking
f2_dr_coax = partial(f2, # FIXME: naming, and many like this. Should be f2_dr_stack_coax
                     r_low=COAX_PARAMS["dr_low_coax"],
                     r_high=COAX_PARAMS["dr_high_coax"],
                     r_c_low=COAX_PARAMS["dr_c_low_coax"],
                     r_c_high=COAX_PARAMS["dr_c_high_coax"],
                     k=COAX_PARAMS["k_coax"],
                     r0=COAX_PARAMS["dr0_coax"],
                     r_c=COAX_PARAMS["dr_c_coax"],
                     b_low=COAX_PARAMS["b_low_coax"],
                     b_high=COAX_PARAMS["b_high_coax"])
f4_theta_4_coax = partial(f4,
                          theta0=COAX_PARAMS["theta0_coax_4"],
                          delta_theta_star=COAX_PARAMS["delta_theta_star_coax_4"],
                          delta_theta_c=COAX_PARAMS["delta_theta_4_c"],
                          a=COAX_PARAMS["a_coax_4"],
                          b=COAX_PARAMS["b_theta_4"])
f4_theta_1_coax = partial(f4,
                          theta0=COAX_PARAMS["theta0_coax_1"],
                          delta_theta_star=COAX_PARAMS["delta_theta_star_coax_1"],
                          delta_theta_c=COAX_PARAMS["delta_theta_1_c"],
                          a=COAX_PARAMS["a_coax_1"],
                          b=COAX_PARAMS["b_theta_1"])
f4_theta_5_coax = partial(f4,
                          theta0=COAX_PARAMS["theta0_coax_5"],
                          delta_theta_star=COAX_PARAMS["delta_theta_star_coax_5"],
                          delta_theta_c=COAX_PARAMS["delta_theta_5_c"],
                          a=COAX_PARAMS["a_coax_5"],
                          b=COAX_PARAMS["b_theta_5"])
f4_theta_6_coax = partial(f4,
                          theta0=COAX_PARAMS["theta0_coax_6"],
                          delta_theta_star=COAX_PARAMS["delta_theta_star_coax_6"],
                          delta_theta_c=COAX_PARAMS["delta_theta_6_c"],
                          a=COAX_PARAMS["a_coax_6"],
                          b=COAX_PARAMS["b_theta_6"])
f5_cosphi3_coax = partial(f5,
                          x_star=COAX_PARAMS["cos_phi3_star_coax"],
                          x_c=COAX_PARAMS["cos_phi3_c"],
                          a=COAX_PARAMS["a_coax_3p"],
                          b=COAX_PARAMS["b_cos_phi3"])
f5_cosphi4_coax = partial(f5,
                          x_star=COAX_PARAMS["cos_phi4_star_coax"],
                          x_c=COAX_PARAMS["cos_phi4_c"],
                          a=COAX_PARAMS["a_coax_4p"],
                          b=COAX_PARAMS["b_cos_phi4"])
def coaxial_stacking(dr_stack, theta4, theta1, theta5, theta6, cosphi3, cosphi4):
    r_stack = jnp.linalg.norm(dr_stack, axis=1)
    return f2_dr_coax(r_stack) * f4_theta_4_coax(theta4) \
        * (f4_theta_1_coax(theta1) + f4_theta_1_coax(2 * jnp.pi - theta1)) \
        * (f4_theta_5_coax(theta5) + f4_theta_5_coax(jnp.pi - theta5)) \
        * (f4_theta_6_coax(theta6) + f4_theta_6_coax(jnp.pi - theta6)) \
        * f5_cosphi3_coax(cosphi3) * f5_cosphi4_coax(cosphi4)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    """
    xs = np.linspace(0.65, 0.85, 50)
    ys = v_fene(xs)
    pdb.set_trace()
    plt.plot(xs, ys)
    plt.show()
    """


    # Test making FENE smooth beyond valid range
    """
    _mbf_fmax = 1000.0 # max_backbone_force, must be > 0
    FENE_EPS = FENE_PARAMS["eps_backbone"]
    FENE_DELTA = FENE_PARAMS["delta_backbone"]
    FENE_DELTA2 = FENE_PARAMS["delta_backbone"]**2
    _mbf_xmax = (-FENE_EPS + np.sqrt(FENE_EPS * FENE_EPS + 4.0 * _mbf_fmax * _mbf_fmax * FENE_DELTA2)) / (2.0 * _mbf_fmax) # from DNAInteraction.cpp::get_settings()

    _mbf_finf = 0.4 # default value. Could also take as argument. Must be > 0


    _use_mbf = False
    def fene_smooth(r):

        rbackr0 = r - FENE_PARAMS["r0_backbone"]

        if _use_mbf and np.abs(rbackr0) > _mbf_xmax:
            fene_xmax = -(FENE_EPS / 2.0) * np.log(1.0 - np.sqrt(_mbf_xmax) / FENE_DELTA2)
            long_xmax = (_mbf_fmax - _mbf_finf) * _mbf_xmax * np.log(_mbf_xmax) + _mbf_finf * _mbf_xmax
            energy = (_mbf_fmax - _mbf_finf) * _mbf_xmax * np.log(np.abs(rbackr0)) + _mbf_finf * np.abs(rbackr0) - long_xmax + fene_xmax
            return energy
        elif np.abs(rbackr0) >= FENE_DELTA:
            return 1.0e12
        else:
            return v_fene(r)


    max_dr = 0.5
    xs = np.linspace(FENE_PARAMS["r0_backbone"] - max_dr, FENE_PARAMS["r0_backbone"] + max_dr, 50)
    ys = list()
    for x in xs:
        ys.append(fene_smooth(x))

    plt.plot(xs, ys)
    plt.show()
    """

    # Test subterms of stacking for a given configuration

    import matplotlib.pyplot as plt
    from jax_md import space
    from jax_md import rigid_body
    import jax.numpy as jnp

    from utils import read_config
    from utils import Q_to_base_normal, Q_to_cross_prod, Q_to_back_base
    from utils import com_to_backbone, com_to_stacking, com_to_hb


    body, box_size = read_config("data/polyA_10bp/equilibrated.dat")
    displacement, shift = space.periodic(box_size)
    d = space.map_bond(partial(displacement))

    nbs_i = np.array(list(range(9)))
    nbs_j = np.array(list(range(1, 10)))

    base_site = jnp.array(
        [com_to_hb, 0.0, 0.0]
    )
    stack_site = jnp.array(
        [com_to_stacking, 0.0, 0.0]
    )
    back_site = jnp.array(
        [com_to_backbone, 0.0, 0.0]
    )

    Q = body.orientation
    back_sites = body.center + rigid_body.quaternion_rotate(Q, back_site) # (N, 3)
    stack_sites = body.center + rigid_body.quaternion_rotate(Q, stack_site)
    base_sites = body.center + rigid_body.quaternion_rotate(Q, base_site)

    dr_back = d(back_sites[nbs_i], back_sites[nbs_j])
    dr_stack = d(stack_sites[nbs_i], stack_sites[nbs_j])
    base_normals = Q_to_base_normal(Q)
    cross_prods = Q_to_cross_prod(Q)
    theta4 = jnp.arccos(jnp.einsum('ij, ij->i', base_normals[nbs_i], base_normals[nbs_j]))
    theta5 = jnp.pi - jnp.arccos(jnp.einsum('ij, ij->i', dr_stack, base_normals[nbs_j]) / jnp.linalg.norm(dr_stack, axis=1))
    theta6 = jnp.pi - jnp.arccos(jnp.einsum('ij, ij->i', base_normals[nbs_i], dr_stack) / jnp.linalg.norm(dr_stack, axis=1)) # NOTE: WE CHANGED THIS

    dr_back2 = d(back_sites[nbs_j], back_sites[nbs_i]) # same as dr_back but in the opposite direction
    # cosphi1 = jnp.einsum('ij, ij->i', cross_prods[nbs_i], dr_back) / jnp.linalg.norm(dr_back, axis=1) # orig. not OK. (goes to 0)
    # cosphi1 = -jnp.einsum('ij, ij->i', cross_prods[nbs_i], dr_back2) / jnp.linalg.norm(dr_back2, axis=1) # not OK (goes to 0)
    # cosphi1 = -jnp.einsum('ij, ij->i', cross_prods[nbs_j], dr_back2) / jnp.linalg.norm(dr_back2, axis=1) # not OK. (goes to 0)
    # cosphi1 = -jnp.einsum('ij, ij->i', cross_prods[nbs_i], dr_back2) / jnp.linalg.norm(dr_back, axis=1) # not OK (goes to 0)
    # cosphi1 = jnp.einsum('ij, ij->i', cross_prods[nbs_i], dr_back2) / jnp.linalg.norm(dr_back2, axis=1) # OK (goes to 1)
    cosphi1 = -jnp.einsum('ij, ij->i', cross_prods[nbs_i], dr_back) / jnp.linalg.norm(dr_back, axis=1) # OK (goes to 1)
    # cosphi1 = jnp.einsum('ij, ij->i', cross_prods[nbs_i], dr_back2) / jnp.linalg.norm(dr_back2, axis=1) # OK (goes to 1)
    # cosphi1 = jnp.einsum('ij, ij->i', cross_prods[nbs_j], dr_back2) / jnp.linalg.norm(dr_back2, axis=1) # OK. (goes to 1)

    # cosphi2 = jnp.einsum('ij, ij->i', cross_prods[nbs_j], dr_back) / jnp.linalg.norm(dr_back, axis=1) # orig. not OK (goes to 0)
    # cosphi2 = jnp.einsum('ij, ij->i', cross_prods[nbs_j], dr_back2) / jnp.linalg.norm(dr_back2, axis=1) # OK (goes to 1)
    # cosphi2 = -jnp.einsum('ij, ij->i', cross_prods[nbs_j], dr_back2) / jnp.linalg.norm(dr_back2, axis=1) # not OK (goes to 0)
    # cosphi2 = jnp.einsum('ij, ij->i', cross_prods[nbs_i], dr_back2) / jnp.linalg.norm(dr_back2, axis=1) # OK (goes to 1)
    # cosphi2 = -jnp.einsum('ij, ij->i', cross_prods[nbs_i], dr_back2) / jnp.linalg.norm(dr_back2, axis=1) # not OK (goes to 0)
    cosphi2 = -jnp.einsum('ij, ij->i', cross_prods[nbs_j], dr_back) / jnp.linalg.norm(dr_back, axis=1) # OK (goes to 1)
    # cosphi2 = jnp.einsum('ij, ij->i', cross_prods[nbs_i], dr_back) / jnp.linalg.norm(dr_back, axis=1) # not OK (goes to 0)
    # cosphi2 = -jnp.einsum('ij, ij->i', cross_prods[nbs_i], dr_back) / jnp.linalg.norm(dr_back, axis=1) # OK (goes to 1). Note, however, that things like this are identical to cosphi1

    r_stack = jnp.linalg.norm(dr_stack, axis=1) # Would happen in `stacking`
    """
    # Note: all negative
    f1_dr_stack_vals = f1_dr_stack(r_stack)

    plt.hist(f1_dr_stack_vals)
    plt.show()
    """

    """
    # Note: all positive between 0 and 1
    f4_theta_4_stack_vals = f4_theta_4_stack(theta4)
    plt.hist(f4_theta_4_stack_vals)
    plt.show()
    """

    """
    # Note: all positive bewteen 0 and 1
    f4_theta_5p_stack_vals = f4_theta_5p_stack(theta5)
    plt.hist(f4_theta_5p_stack_vals)
    plt.show()
    """

    """
    # NOTE: Once we corrected by doing jnp.pi - (original theta6), we went from ~0 to ~1!
    f4_theta_6p_stack_vals = f4_theta_6p_stack(theta6)
    plt.hist(f4_theta_6p_stack_vals)
    plt.show()
    """


    """
    # NOTE: Once we corrected by taking a negative
    f5_neg_cosphi1_stack_vals = f5_neg_cosphi1_stack(-cosphi1)
    plt.hist(f5_neg_cosphi1_stack_vals)
    plt.show()
    """

    """
    # NOTE: Also, corrected by taking a negative
    f5_neg_cosphi2_stack_vals = f5_neg_cosphi2_stack(-cosphi2)
    plt.hist(f5_neg_cosphi2_stack_vals)
    plt.show()
    """


    # Major note: I think all things are up to this point such that "labelled nucleotide is in the 3' direction"
    # Next: simulate. If good, send and respond to Megan
