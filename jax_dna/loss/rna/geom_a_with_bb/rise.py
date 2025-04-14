# ruff: noqa
import functools
import pdb

import jax.numpy as jnp
from jax import jit, vmap

from jax_dna.loss.rna.geom_a_with_bb import utils as rna_loss_utils


def get_mean_rise(body, first_base, last_base):
    n = body.center.shape[0]
    _, stack_sites, _, _, _, _ = rna_loss_utils.get_site_positions(body)

    plane_vector = rna_loss_utils.get_plane_vector(stack_sites, n, first_base, last_base)

    pos_as, pos_bs = rna_loss_utils.get_pos_abs(stack_sites, n, first_base, last_base)

    def compute_rise(i):
        midp_a = 0.5 * (pos_as[i] + pos_bs[i])
        midp_b = 0.5 * (pos_as[i + 1] + pos_bs[i + 1])

        da = jnp.dot(plane_vector, midp_a)
        db = jnp.dot(plane_vector, midp_b)
        return db - da

    all_rises = vmap(compute_rise)(jnp.arange(pos_as.shape[0] - 1))
    return all_rises.mean()


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as onp

    from jax_dna.common import topology, trajectory, utils

    basedir = Path("data/test-data/rna2-13bp-md")

    top_path = basedir / "sys.top"
    top_info = topology.TopologyInfo(
        top_path,
        reverse_direction=False,
        is_rna=True,
    )

    traj_path = basedir / "output.dat"
    traj_info = trajectory.TrajectoryInfo(top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
    traj_states = traj_info.get_states()
    traj_states = utils.tree_stack(traj_states)

    offset = 1
    n_bp = 13
    first = offset
    last = n_bp - offset - 2

    all_rises = vmap(get_mean_rise, (0, None, None))(traj_states, first, last)

    mean_rise = all_rises.mean()
    print(f"Mean rise (oxDNA units): {mean_rise}")
    conversion = 8.4
    print(f"Mean rise (A): {mean_rise*conversion}")
