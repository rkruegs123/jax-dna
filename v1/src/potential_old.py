from functools import partial
import jax.numpy as jnp
import toml
from jax_md import energy
import pdb

from jax.tree_util import Partial # FIXME: only update v_fene for now...
from jax.config import config
config.update("jax_enable_x64", True)



# Define functional forms
# FIXME: Do cutoff with Carl's method. Likely don't need r_c_low and r_c_high
def f1(r, r_low, r_high, r_c_low, r_c_high, # thresholding/smoothing parameters
       eps, a, r0, r_c, # morse parameters
       b_low, b_high, # smoothing parameters
):
    return jnp.where(r_low < r & r < r_high,
                     v_morse(r, eps, r0, a) - v_morse(r_c, eps, r0, a),
                     jnp.where(jnp.logical_and(jnp.less(r_c_low, r), jnp.less(r, r_low)),
                               eps * v_smooth(r, b_low, r_c_low),
                               jnp.where(jnp.logical_and(jnp.less(r_high, r), jnp.less(r, r_c_high)),
                                         eps * v_smooth(r, b_high, r_c_high),
                                         0.0)))

    """
    # Our own
    def _sub(_r):
        if r_low < _r and _r < r_high:
            return v_morse(_r, eps, r0, a) - v_morse(r_c, eps, r0, a)
        elif r_c_low < _r and _r < r_low:
            return eps * v_smooth(_r, b_low, r_c_low)
        elif r_high < _r and _r < r_c_high:
            return eps * v_smooth(_r, b_high, r_c_high)
        else:
            return 0.0
    """

    """
    # from DNAInteraction.cpp::_f1
    def _sub(_r):
        val = 0.0
        shift = eps * onp.sqrt(1 - onp.exp(-(r_c - r0) * a))
        if _r < r_c_high:
            if _r > r_high:
                val = eps * b_high * onp.sqrt(_r - r_c_high)
            elif _r > r_low:
                tmp = 1 - onp.exp(-(_r - r0) * a)
                val = eps * onp.sqrt(tmp) - shift
            elif _r > r_c_low:
                val = eps * b_low * onp.sqrt(_r - r_c_low)
        return val


    return jnp.array([_sub(r1) for r1 in r])
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

## FIXME: Could use ones from JAX-MD when appropriate (e.g. morse, harmonic). Could just add ones to JAX-MD that are missing (e.g. FENE)
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

def v_smooth(x, b, x_c):
    return b*(x_c - x)**2



def get_potentials(params):

    FENE_PARAMS = params["fene"]
    EXC_VOL_PARAMS = params["excluded_volume"]
    STACK_PARAMS = params["stacking"]
    HB_PARAMS = params["hydrogen_bonding"]
    CROSS_PARAMS = params["cross_stacking"]
    COAX_PARAMS = params["coaxial_stacking"]


    # Define individual potentials
    v_fene = Partial(_v_fene, eps=FENE_PARAMS["eps_backbone"], r0=FENE_PARAMS["r0_backbone"],
                     delt=FENE_PARAMS["delta_backbone"])

    """
    v_fene_helper = partial(_v_fene, eps=FENE_PARAMS["eps_backbone"], r0=FENE_PARAMS["r0_backbone"],
                            delt=FENE_PARAMS["delta_backbone"])
    def v_fene(r):
        # Thresholded version -- analogous to allowing broken bakcbone from oxDNA
        rbackr0 = r - FENE_PARAMS["r0_backbone"]

        return jnp.where(jnp.abs(rbackr0) >= FENE_PARAMS["delta_backbone"],
                         1.0e12,
                         v_fene_helper(r))
    """




    """
    f3_base = get_f3(r_star=EXC_VOL_PARAMS["dr_star_base"],
                     r_c=EXC_VOL_PARAMS["dr_c_base"],
                     eps=EXC_VOL_PARAMS["eps_exc"],
                     sigma=EXC_VOL_PARAMS["sigma_base"])
    """
    f3_base_exc_vol = Partial(f3,
                              r_star=EXC_VOL_PARAMS["dr_star_base"],
                              r_c=EXC_VOL_PARAMS["dr_c_base"],
                              eps=EXC_VOL_PARAMS["eps_exc"],
                              sigma=EXC_VOL_PARAMS["sigma_base"],
                              b=EXC_VOL_PARAMS["b_base"])
    f3_back_base_exc_vol = Partial(f3,
                                   r_star=EXC_VOL_PARAMS["dr_star_back_base"],
                                   r_c=EXC_VOL_PARAMS["dr_c_back_base"],
                                   eps=EXC_VOL_PARAMS["eps_exc"],
                                   sigma=EXC_VOL_PARAMS["sigma_back_base"],
                                   b=EXC_VOL_PARAMS["b_back_base"])
    f3_base_back_exc_vol = Partial(f3,
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

    f3_back_exc_vol = Partial(f3,
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

    f1_dr_stack = Partial(f1,
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
    f4_theta_4_stack = Partial(f4,
                               theta0=STACK_PARAMS["theta0_stack_4"],
                               delta_theta_star=STACK_PARAMS["delta_theta_star_stack_4"],
                               delta_theta_c=STACK_PARAMS["delta_theta_4_c"],
                               a=STACK_PARAMS["a_stack_4"],
                               b=STACK_PARAMS["b_theta_4"])
    f4_theta_5p_stack = Partial(f4,
                                theta0=STACK_PARAMS["theta0_stack_5"],
                                delta_theta_star=STACK_PARAMS["delta_theta_star_stack_5"],
                                delta_theta_c=STACK_PARAMS["delta_theta_5_c"],
                                a=STACK_PARAMS["a_stack_5"],
                                b=STACK_PARAMS["b_theta_5"])
    f4_theta_6p_stack = Partial(f4,
                                theta0=STACK_PARAMS["theta0_stack_6"],
                                delta_theta_star=STACK_PARAMS["delta_theta_star_stack_6"],
                                delta_theta_c=STACK_PARAMS["delta_theta_6_c"],
                                a=STACK_PARAMS["a_stack_6"],
                                b=STACK_PARAMS["b_theta_6"])
    f5_neg_cosphi1_stack = Partial(f5,
                                   x_star=STACK_PARAMS["neg_cos_phi1_star_stack"],
                                   x_c=STACK_PARAMS["neg_cos_phi1_c"],
                                   a=STACK_PARAMS["a_stack_1"],
                                   b=STACK_PARAMS["b_neg_cos_phi1"])
    f5_neg_cosphi2_stack = Partial(f5,
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

    return v_fene, exc_vol_bonded, stacking, exc_vol_unbonded, hydrogen_bonding, cross_stacking, coaxial_stacking


if __name__ == "__main__":
    pass
