# ruff: noqa
import functools
import pdb

import jax.numpy as jnp
from jax import jit, vmap

from jax_dna.loss.rna.utils import fit_plane


def get_rises(stack_sites, n, first_base, last_base):
    back_poses = []

    def get_back_positions(i):
        nt11 = i
        nt12 = n - i - 1

        nt21 = i + 1
        nt22 = n - (i + 1) - 1

        back_pos1 = stack_sites[nt11] - stack_sites[nt21]
        back_pos2 = stack_sites[nt12] - stack_sites[nt22]
        return jnp.array([back_pos1, back_pos2])

    back_poses = vmap(get_back_positions)(jnp.arange(first_base, last_base))
    back_poses = back_poses.reshape(-1, 3)

    # now we try to fit a plane through all these points
    plane_vector = fit_plane(back_poses)

    midp_first_base = (stack_sites[first_base] + stack_sites[n - first_base - 1]) / 2

    midp_last_base = (stack_sites[last_base] + stack_sites[n - last_base - 1]) / 2
    guess = (
        -midp_first_base + midp_last_base
    )  # vector pointing from midpoint of the first bp to the last bp, an estimate of the vector
    guess = guess / jnp.linalg.norm(guess)

    # Check if plane vector is pointing in opposite direction
    plane_vector = jnp.where(jnp.dot(guess, plane_vector) < 0, -1.0 * plane_vector, plane_vector)

    """
    if (onp.rad2deg(math.acos(np.dot(plane_vector,guess/my_norm(guess)))) > 20):
        # print 'Warning, guess vector and plane vector have angles:', np.rad2deg(math.acos(np.dot(guess/my_norm(guess),plane_vector)))
        pdb.set_trace()
        pass
    """

    # Now, compute the rises
    n_bps = last_base - first_base

    def single_rise(bp_idx):
        i = first_base + bp_idx

        midp = (stack_sites[i] + stack_sites[n - i - 1]) / 2
        midp_ip1 = (stack_sites[i + 1] + stack_sites[n - (i + 1) - 1]) / 2

        midp_proj = jnp.dot(plane_vector, midp)
        midp_ip1_proj = jnp.dot(plane_vector, midp_ip1)

        rise = midp_ip1_proj - midp_proj
        return rise

    rises = vmap(single_rise)(jnp.arange(n_bps))

    return rises.mean()


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

    def compute_body_rise(body, first_base, last_base):
        n = body.center.shape[0]

        Q = body.orientation
        back_base_vectors = utils.Q_to_back_base(Q)  # space frame, normalized

        com_to_backbone_x = -0.4
        com_to_backbone_y = 0.2
        com_to_stacking = 0.34

        stack_sites = body.center + com_to_stacking * back_base_vectors

        return get_rises(stack_sites, n, first_base, last_base)

    # first = 1
    # last = 12
    offset = 1
    n_bp = 13
    first = offset
    last = n_bp - offset - 2

    # all_rises = vmap(compute_body_rise, (0, None, None))(traj_states, 1, 12)
    all_rises = vmap(compute_body_rise, (0, None, None))(traj_states, first, last)
    pdb.set_trace()
    all_rises = all_rises * utils.nm_per_oxrna_length

    running_avg = onp.cumsum(all_rises) / onp.arange(1, all_rises.shape[0] + 1)
    plt.plot(running_avg)
    plt.show()
    plt.close()

    mean_rise = all_rises.mean()
    print(f"Mean rise: {mean_rise}")
