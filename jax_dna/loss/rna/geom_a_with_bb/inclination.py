import functools
import pdb

from jax import jit, vmap
import jax.numpy as jnp

from jax_dna.loss.rna.geom_a_with_bb import utils as rna_loss_utils



def get_mean_inclination(body, first_base, last_base):
    n = body.center.shape[0]
    back_sites, stack_sites, _, a1s, _, _ = rna_loss_utils.get_site_positions(body)

    plane_vector = rna_loss_utils.get_plane_vector(stack_sites, n, first_base, last_base)

    def compute_inclination(i):

        pos_a = back_sites[i]
        pos_b = back_sites[n-1-i]

        a1_a = a1s[i]
        a1_b = -a1s[n-1-i]

        inclination_vector = pos_a - pos_b
        inclination_vector = inclination_vector / jnp.linalg.norm(inclination_vector)

        # angle = jnp.rad2deg(jnp.acos(jnp.dot(inclination_vector, plane_vector)))
        angle_a = jnp.rad2deg(jnp.acos(jnp.dot(a1_a, plane_vector)))
        angle_b = jnp.rad2deg(jnp.acos(jnp.dot(a1_b, plane_vector)))

        inclination = 0.5*(angle_a + angle_b)
        return 90-inclination
    all_inclinations = vmap(compute_inclination)(jnp.arange(last_base - first_base))
    return all_inclinations.mean()



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

    all_inclinations = vmap(get_mean_inclination, (0, None, None))(traj_states, first, last)

    mean_inclination = all_inclinations.mean()
    print(f"Mean inclination (deg): {mean_inclination}")
