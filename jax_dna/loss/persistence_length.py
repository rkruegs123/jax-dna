# ruff: noqa
# fmt: off
import pdb
import numpy as onp

import jax.numpy as jnp
from jax import vmap
from jax_md import rigid_body, util
from jax_md.rigid_body import RigidBody, Quaternion

from jax_dna.common import utils

from jax.config import config
config.update("jax_enable_x64", True)


Array = util.Array

# Tom's thesis: 130-150 base pairs, 475 A, 47.5 nm (note: there is a typo b/w A and nm)
TARGET_PERSISTENCE_LENGTH_DSDNA = 47.5 # nm


"""
def vector_autocorrelate(arr):
    n_vectors = arr.shape[0]

    # correlate each component indipendently
    acorr = jnp.array([jnp.correlate(arr[:,i], arr[:,i], 'full') for i in jnp.arange(3)])[:, n_vectors-1:] # we should  really vmap over this, but for simplicity, we unroll a for loop for now

    # sum the correlations for each component
    acorr = jnp.sum(acorr, axis = 0)

    # divide by the number of values actually measured and return
    acorr /= (n_vectors - jnp.arange(n_vectors))

    return acorr
"""

def vector_autocorrelate(arr):
    max_dist = arr.shape[0]

    def window_correlations(i):
        li = arr[i]
        i_correlation_fn = lambda j: jnp.where(j >= i, jnp.dot(li, arr[j]), 0.0)
        i_correlations = vmap(i_correlation_fn)(jnp.arange(max_dist))
        i_correlations = jnp.roll(i_correlations, -i)
        return i_correlations

        all_correlations += i_correlations

    all_correlations = vmap(window_correlations)(jnp.arange(max_dist))
    all_correlations = jnp.sum(all_correlations, axis=0)

    all_correlations /= jnp.arange(max_dist, 0, -1)
    return all_correlations


def compute_l_vector(quartet, system: RigidBody, base_sites: Array):
    # Base pair 1 is comprised of a1 and b1. Base pair #2 is a2, b2.
    # i.e. a1 is H-bonded to b1, a2 is h-bonded to b2
    a1, b1, a2, b2 = quartet # a1, b1, a2, b2 are the indices of the relevant nucleotides

    # get midpoints for each base pair
    mp1 = (base_sites[b1] + base_sites[a1]) / 2.
    mp2 = (base_sites[b2] + base_sites[a2]) / 2.

    # get vector between midpoint
    l = mp2 - mp1
    l0 = jnp.linalg.norm(l)
    l /= l0

    return l, l0

# vector autocorrelate: https://stackoverflow.com/questions/48844295/computing-autocorrelation-of-vectors-with-numpy
def get_correlation_curve(system: RigidBody, base_quartets: Array, base_site):
    base_sites = system.center + rigid_body.quaternion_rotate(system.orientation, base_site)
    get_all_l_vectors = vmap(compute_l_vector, in_axes = [0, None, None])
    all_l_vectors, l0_vals = get_all_l_vectors(base_quartets, system, base_sites)
    autocorr = vector_autocorrelate(all_l_vectors)
    return autocorr, jnp.mean(l0_vals)

def persistence_length_fit(autocorr, l0_av):
    y = jnp.log(autocorr)
    x = jnp.arange(autocorr.shape[0])
    x = jnp.stack([jnp.ones_like(x), x], axis=1)

    # fit line
    fit_ = jnp.linalg.lstsq(x, y)

    # extract slope and compute Lp
    slope = fit_[0][1] # slope = -l0_av / Lp
    Lp = -l0_av/slope

    offset = fit_[0][0] # slope = -l0_av / Lp

    return Lp, offset

def get_persistence_length_loss_fn(base_quartets, com_to_hb, target_lp=TARGET_PERSISTENCE_LENGTH_DSDNA):

    base_site = jnp.array([com_to_hb, 0.0, 0.0])

    def compute_lp(body):
        correlation_curve, l0_avg = get_correlation_curve(body, base_quartets, base_site)
        lp, _ = persistence_length_fit(correlation_curve, l0_avg) # oxDNA units

        lp_nm = lp * utils.nm_per_oxdna_length # nanometers
        return lp_nm

    def loss_fn(body):
        lp_nm = compute_lp(body)
        return (lp_nm - target_lp)**2

    return compute_lp, loss_fn
