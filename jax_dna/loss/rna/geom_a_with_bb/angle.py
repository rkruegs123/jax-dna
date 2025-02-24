import functools
import pdb

from jax import jit, vmap
import jax.numpy as jnp

from jax_dna.loss.rna.geom_a_with_bb import utils as rna_loss_utils



def get_mean_angle(body, first_base, last_base):
    n = body.center.shape[0]
    back_sites, stack_sites, base_sites = rna_loss_utils.get_site_positions(body)

    plane_vector = rna_loss_utils.get_plane_vector(stack_sites, n, first_base, last_base)

    pos_as, pos_bs = rna_loss_utils.get_pos_abs(stack_sites, n, first_base, last_base)

    def compute_angle(i):
        vec_bp = -pos_as[i] + pos_bs[i]
        vec_bp_norm = vec_bp / jnp.linalg.norm(vec_bp)

        vec_bp_2 = -pos_as[i+1] + pos_bs[i+1]
        vec_bp_2 = vec_bp_2 / jnp.linalg.norm(vec_bp_2)

        vec_bp_norm = vec_bp_norm - plane_vector * jnp.dot(vec_bp_norm, plane_vector)
        vec_bp_norm = vec_bp_norm / jnp.linalg.norm(vec_bp_norm)

        vec_bp_2 = vec_bp_2 - plane_vector * jnp.dot(vec_bp_2, plane_vector)
        vec_bp_2 = vec_bp_2 / jnp.linalg.norm(vec_bp_2)

        angle_rad = jnp.acos(jnp.dot(vec_bp_norm, vec_bp_2))
        return angle_rad
    all_angles_rad = vmap(compute_angle)(jnp.arange(pos_as.shape[0] - 1))
    return all_angles_rad.mean()



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


    offset = 1
    n_bp = 13
    first = offset
    last = n_bp-offset-2

    s0 = traj_states[0]

    s0_angle_rad = get_mean_angle(s0, first, last)
    s0_angle_deg = jnp.rad2deg(s0_angle_rad)

    all_angles_rad = vmap(get_mean_angle, (0, None, None))(traj_states, first, last)

    mean_angle = all_angles_rad.mean()
    print(f"Mean angle (rad): {mean_angle}")
    print(f"Mean angle (deg): {180.0 * mean_angle / jnp.pi}")
    n_bp_per_turn = 2.0 * jnp.pi / mean_angle
    print(f"Pitch (# bp / turn): {n_bp_per_turn}")
