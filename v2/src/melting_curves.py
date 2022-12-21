import pdb
import pickle
from pathlib import Path
import numpy as onp
import matplotlib.pyplot as plt
from copy import deepcopy

from jax import jit
import jax.numpy as jnp

from metadynamics import cv
import metadynamics.utils as md_utils
import metadynamics.energy as md_energy



def toms_correction(phi):
    t1 = 1 + 1/(2*phi)
    t2_arg = (1+(1/(2*phi)))**2 - 1
    t2 = onp.sqrt(t2_arg)
    f_inf = t1 - t2
    return f_inf


# 320 K is oxDNA 1 melting temperature for 8 bps
# 300, 310, 320, 330, 340
# compute f_inf for each temperature and fit a sigmoid as the melting temperature

if __name__ == "__main__":
    # bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/metad_2022-12-16_21-41-38")
    bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/metad_2022-12-18_01-31-31")
    # bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/metad_2022-12-20_06-31-49")
    centers = pickle.load(open(bpath / "centers.pkl", "rb"))
    widths = pickle.load(open(bpath / "widths.pkl", "rb"))
    heights = pickle.load(open(bpath / "heights.pkl", "rb"))

    pdb.set_trace()


    repulsive_wall_fn = md_utils.get_repulsive_wall_fn(d_critical=15.0, wall_strength=1000.0)
    repulsive_wall_fn = jit(repulsive_wall_fn)

    n_bp_lo = -1
    n_bp_hi = 8
    num_n_bp_samples = 50
    sample_n_bps = onp.linspace(n_bp_lo, n_bp_hi, num_n_bp_samples)
    distances_lo = 0.0
    distances_hi = 20.0
    num_distances_samples = 200
    sample_distances = onp.linspace(distances_lo, distances_hi, num_distances_samples)


    """
    n_bp_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for n_bp in n_bp_to_test:
        vals_at_zero = [repulsive_wall_fn(heights, centers, widths, n_bp, d) for d in sample_distances]
        plt.plot(vals_at_zero, label=f"{n_bp}")
    plt.xticks(list(range(len(sample_distances)))[::20], list(onp.round(sample_distances, 2))[::20])
    plt.legend()
    plt.show()
    pdb.set_trace()
    """



    b, a = onp.meshgrid(sample_n_bps, sample_distances)
    vals = onp.empty((b.shape))
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            vals[i, j] = repulsive_wall_fn(heights, centers, widths, b[i, j], a[i, j]) # FIXME: maybe swap a and b?
    l_a = a.min()
    r_a = a.max()
    l_b = b.min()
    r_b = b.max()


    bound_threshold = 0.4

    d_n_bp = (n_bp_hi - n_bp_lo) / (num_n_bp_samples-1)
    idx_limit = int(jnp.ceil((bound_threshold - n_bp_lo) / d_n_bp))

    vals = jnp.array(vals)
    # pdb.set_trace()
    # cutoff = 1e-2
    # vals = jnp.where(vals < cutoff, jnp.inf, -vals) # should be the free energy
    # vals = -vals
    kt = 0.1
    beta = 1/kt
    # probs = jnp.exp(-beta*vals)
    offset = 0 # Note: wlil really want a continuous way to get the offset. maybe just the max of the biases, or  the max minus a little bit. could also approximate given the stride and the deposit height
    offset_vals = vals - offset
    scaled_probs = jnp.exp(beta*offset_vals)


    trunc_vals = vals[:, :idx_limit]
    trunc_a = a[:, :idx_limit]
    trunc_b = b[:, :idx_limit]

    pdb.set_trace()

    unbound_probs_at_distance = jnp.trapz(
        # probs[:, :idx_limit],
        scaled_probs[:, :idx_limit],
        dx=d_n_bp,
        axis=1
    )
    unbound_free_energies_at_distance = -kt * jnp.log(unbound_probs_at_distance)

    wall = 10.5

    d_distance = (distances_hi - distances_lo) / (num_distances_samples-1)
    wall_idx = int(jnp.ceil((wall - distances_lo) / d_distance))


    stitch_point = sample_distances[wall_idx]
    stitch_point_fe = unbound_free_energies_at_distance[wall_idx]

    # correction: free energy at the wall + 2*kT*ln(edge_distance)
    # e.g. say we are correcting at x > stitch_point, then we have:
    # free energy @ stitch_point + 2*kT*ln(stitch_point) - 2*kT*ln(x)
    # do this correction for all x > stitch_point pu
    corrected_unbound_free_energies_at_distance = onp.array(unbound_free_energies_at_distance)
    for i in range(len(unbound_free_energies_at_distance)):
        if sample_distances[i] > stitch_point:
            corrected_f = stitch_point_fe + 2*kt*jnp.log(stitch_point) - 2*kt*jnp.log(sample_distances[i])
            corrected_unbound_free_energies_at_distance[i] = float(corrected_f)

    plt.plot(corrected_unbound_free_energies_at_distance)
    plt.xticks(list(range(len(sample_distances)))[::40], list(onp.round(sample_distances, 2))[::40])
    plt.show()

    # convert this back to probabilities
    corrected_unbound_probs_at_distance = jnp.exp(-beta*corrected_unbound_free_energies_at_distance)
    unnormalizeld_unbound_prob = jnp.trapz(corrected_unbound_probs_at_distance, dx=d_distance)


    bound_probs_at_distance = jnp.trapz(
        # probs[:, :idx_limit],
        scaled_probs[:, idx_limit:],
        dx=d_n_bp,
        axis=1
    )
    unnormalized_bound_prob = jnp.trapz(bound_probs_at_distance[:wall_idx], dx=d_distance)

    bound_over_unbound = unnormalized_bound_prob / unnormalizeld_unbound_prob
    f_inf = toms_correction(phi=bound_over_unbound) # estimate for current temperature

    pdb.set_trace()

    # TODO: Tom's correction






    pdb.set_trace()

    # l_val, r_val = onp.abs(vals).min(), onp.abs(vals).max()

    figure, axes = plt.subplots()
    # c = axes.pcolormesh(a, b, vals, cmap='cool', vmin=l_val, vmax=r_val)
    c = axes.pcolormesh(a, b, vals, cmap='cool')
    # c = axes.pcolormesh(trunc_a, trunc_b, trunc_vals, cmap='cool')
    # axes.axis([l_a, r_a, l_b, r_b])
    figure.colorbar(c)
    plt.xlabel("Interstrand Distance")
    plt.ylabel("# Base Pairs")

    plt.show()



    pdb.set_trace()

    print("done")
