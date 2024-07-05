import jax.numpy as jnp

# https://ora.ox.ac.uk/objects/uuid:b2415bb2-7975-4f59-b5e2-8c022b4a3719/files/mdcac62bc9133143fc05070ed20048c50
# Section 2.4.1

def v_fene(r, eps, r0, delt):
    """
    The vanilla FENE potential. Used in a smoothed FENE potential in oxDNA.
    """
    x = (r - r0) ** 2 / delt**2
    # Note: if `x` is too big, we will easily try to take the log of negatives, which will yield `nan`
    return -eps / 2.0 * jnp.log(1 - x)


def v_morse(r, eps, r0, a):
    x = -(r - r0) * a
    return eps * (1 - jnp.exp(x)) ** 2


def v_harmonic(r, k, r0):
    return k / 2 * (r - r0) ** 2


def v_lj(r, eps, sigma):
    x = (sigma / r) ** 12 - (sigma / r) ** 6
    return 4 * eps * x


def v_mod(theta, a, theta0):
    return 1 - a * (theta - theta0) ** 2


def v_smooth(x, b, x_c):
    return b * (x_c - x) ** 2
