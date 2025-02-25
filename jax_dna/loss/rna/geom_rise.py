import functools
import pdb

from jax import jit, vmap
import jax.numpy as jnp

from jax_dna.loss.rna.utils import fit_plane


def get_mean_rise(base_sites, n, first_base, last_base):

    # compute hel_axis
    # i1A = first_base = offset
    # i1B = n_bp - offset - 1 = last_base

    ## First base pair
    i1A = first_base
    i1B = n-first_base-1
    first_midp = (base_sites[i1A] + base_sites[i1B]) / 2

    ## Second base pair
    i2A = last_base
    i2B = n-last_base-1
    last_midp = (base_sites[i2A] + base_sites[i2B]) / 2

    # Get normalized displacement
    r0N = last_midp - first_midp
    hel_axis = r0N / jnp.linalg.norm(r0N)


    # Now, compute the rises
    # n_bps = last_base - first_base
    n_bps = last_base - first_base - 1

    def single_rise(bp_idx):
        i = first_base + bp_idx

        first_midp = (base_sites[i] + base_sites[n-i-1]) / 2
        second_midp = (base_sites[i+1] + base_sites[n-(i+1)-1]) / 2

        dr = second_midp - first_midp
        return jnp.dot(dr, hel_axis)
    rises = vmap(single_rise)(jnp.arange(n_bps))

    return rises.mean()
    # return rises


if __name__ == "__main__":
    from pathlib import Path
    from jax_dna.common import utils, topology, trajectory
    import numpy as onp
    import matplotlib.pyplot as plt

    basedir = Path("data/test-data/rna2-13bp-md")

    top_path = basedir / "sys.top"
    top_info = topology.TopologyInfo(
        top_path,
        reverse_direction=False,
        is_rna=True,
    )

    traj_path = basedir / "output.dat"
    traj_info = trajectory.TrajectoryInfo(
        top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False
    )
    traj_states = traj_info.get_states()
    traj_states = utils.tree_stack(traj_states)

    def compute_body_rise(body, first_base, last_base):
        n = body.center.shape[0]

        Q = body.orientation
        back_base_vectors = utils.Q_to_back_base(Q) # space frame, normalized

        com_to_hb = 0.4

        base_sites = body.center + com_to_hb * back_base_vectors

        return get_mean_rise(base_sites, n, first_base, last_base)


    # first = 1
    # last = 12
    offset = 1
    n_bp = 13
    first = offset
    # last = n_bp-offset-2
    last = n_bp-offset-1 # -1 for whatever reason (rather than -2) in geom vs. lp code

    # all_rises = vmap(compute_body_rise, (0, None, None))(traj_states, 1, 12)
    # s0_rises = compute_body_rise(traj_states[0], first, last)
    all_rises = vmap(compute_body_rise, (0, None, None))(traj_states, first, last)
    all_rises = all_rises*utils.nm_per_oxrna_length*10

    mean_rise = all_rises.mean()
    print(f"Mean rise: {mean_rise}")
