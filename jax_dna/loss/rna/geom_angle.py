import functools
import pdb

from jax import jit, vmap
import jax.numpy as jnp

from jax_dna.loss.rna.utils import fit_plane


def get_angles(base_sites, back_sites, n, first_base, last_base):

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


    # Now, compute the angles
    # n_bps = last_base - first_base
    n_bps = last_base - first_base - 1

    def single_angle(bp_idx):
        i = first_base + bp_idx

        first_dr = (back_sites[i] - back_sites[n-i-1])
        first_dr_norm = first_dr / jnp.linalg.norm(first_dr)

        second_dr = (back_sites[i+1] - back_sites[n-(i+1)-1])
        second_dr_norm = second_dr / jnp.linalg.norm(second_dr)

        first_dr_norm = first_dr_norm - jnp.dot(hel_axis, first_dr_norm) * hel_axis
        first_dr_norm = first_dr_norm / jnp.linalg.norm(first_dr_norm)

        second_dr_norm = second_dr_norm - jnp.dot(hel_axis, second_dr_norm) * hel_axis
        second_dr_norm = second_dr_norm / jnp.linalg.norm(second_dr_norm)

        return jnp.acos(jnp.dot(first_dr_norm, second_dr_norm))
    angles = vmap(single_angle)(jnp.arange(n_bps))

    return angles.mean()


if __name__ == "__main__":
    from pathlib import Path
    from jax_dna.common import utils, topology, trajectory
    import numpy as onp
    import matplotlib.pyplot as plt

    basedir = Path("data/templates/rna2-13bp-md")

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

    def compute_body_angle(body, first_base, last_base):
        n = body.center.shape[0]

        Q = body.orientation
        back_base_vectors = utils.Q_to_back_base(Q) # space frame, normalized
        base_normals = utils.Q_to_base_normal(Q) # space frame, normalized

        com_to_backbone_x = -0.4
        com_to_backbone_y = 0.2
        back_sites = body.center + com_to_backbone_x*back_base_vectors + com_to_backbone_y*base_normals

        com_to_hb = 0.4
        base_sites = body.center + com_to_hb * back_base_vectors

        return get_angles(base_sites, back_sites, n, first_base, last_base)


    # first = 1
    # last = 12
    offset = 1
    n_bp = 13
    first = offset
    # last = n_bp-offset-2
    last = n_bp-offset-1 # -1 for whatever reason (rather than -2) in geom vs. lp code

    # s0_angles = compute_body_angle(traj_states[0], first, last)
    all_angles = vmap(compute_body_angle, (0, None, None))(traj_states, first, last)

    mean_angle = all_angles.mean()
    print(f"Mean angle (rad): {mean_angle}")
    print(f"Mean angle (deg): {180.0 * mean_angle / jnp.pi}")
    n_bp_per_turn = 2.0 * jnp.pi / mean_angle
    print(f"Pitch (# bp / turn): {n_bp_per_turn}")
