import pdb
import numpy as onp
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap, jit
from jax_md import space

from jax_dna.common import utils, topology, trajectory

@jit
def get_site_positions(body):
    Q = body.orientation
    back_base_vectors = utils.Q_to_back_base(Q) # space frame, normalized
    base_normals = utils.Q_to_base_normal(Q) # space frame, normalized
    cross_prods = utils.Q_to_cross_prod(Q) # space frame, normalized

    # RNA values
    """
    com_to_backbone_x = -0.4
    com_to_backbone_y = 0.2
    com_to_stacking = 0.34
    com_to_hb = 0.4
    back_sites = body.center + com_to_backbone_x*back_base_vectors + com_to_backbone_y*base_normals
    stack_sites = body.center + com_to_stacking * back_base_vectors
    base_sites = body.center + com_to_hb * back_base_vectors
    """

    # In code (like DNA1)
    com_to_backbone = -0.4
    com_to_stacking = 0.34
    com_to_hb = 0.4
    back_sites = body.center + com_to_backbone*back_base_vectors
    stack_sites = body.center + com_to_stacking * back_base_vectors
    base_sites = body.center + com_to_hb * back_base_vectors

    return back_sites, stack_sites, base_sites

@jit
def fit_plane(points):
    """Fits plane through points, return normal to the plane."""

    n_points = points.shape[0]
    rc = jnp.sum(points, axis=0) / n_points
    points_centered = points - rc

    As = vmap(lambda point: jnp.kron(point, point).reshape(3, 3))(points_centered)
    A = jnp.sum(As, axis=0)
    vals, vecs = jnp.linalg.eigh(A)
    return vecs[:, 0]

@partial(jax.jit, static_argnums=(5, 6))
def get_localized_axis(body, back_sites, stack_sites, base_sites, base_id, down_length, up_length):

    n = body.center.shape[0]

    def get_base_plane_nuc_info(i):
        midpoint_A = 0.5*(stack_sites[i] + stack_sites[n-i-1])
        midpoint_B = 0.5*(stack_sites[n-i-1-1] + stack_sites[i+1])

        guess = -midpoint_A + midpoint_B

        p1 = back_sites[i] - back_sites[i+1]
        p2 = back_sites[n-i-1] - back_sites[n-i-1-1]
        p3 = midpoint_A - midpoint_B
        return guess, jnp.array([p1, p2, p3])

    n_base_plane_idxs = 1 + down_length + up_length
    base_plane_idxs = base_id-down_length + jnp.arange(n_base_plane_idxs)

    # base_plane_idxs = jnp.arange(base_id-down_length, base_id+up_length+1)
    guesses, plane_points = vmap(get_base_plane_nuc_info)(base_plane_idxs)
    plane_points = plane_points.reshape(-1, 3)

    mean_guess = jnp.mean(guesses, axis=0)
    mean_guess = mean_guess / jnp.linalg.norm(mean_guess)
    plane_vector = fit_plane(plane_points)

    plane_vector = jnp.where(jnp.dot(mean_guess, plane_vector) < 0, -1.0*plane_vector, plane_vector)

    return plane_vector / jnp.linalg.norm(plane_vector)

def get_cosines(body, cosines, cosines_counter, base_start, down_neigh, up_neigh):

    back_sites, stack_sites, base_sites = get_site_positions(body)
    l_0 = get_localized_axis(body, back_sites, stack_sites, base_sites, base_start, down_neigh, up_neigh)

    for i in range(base_start,base_start + len(cosines)):
        l_i = get_localized_axis(body, back_sites, stack_sites, base_sites, i, down_neigh, up_neigh)
        cosines[i - base_start] += onp.dot( l_0, l_i  )
        cosines_counter[i - base_start] += 1

def run():
    basedir = Path("output") / "test-rna-lp"
    assert(basedir.exists())
    top_path = basedir / "sys.top"
    traj_path = basedir / "output.dat"

    top_info = topology.TopologyInfo(top_path, reverse_direction=False, is_rna=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq, is_rna=True), dtype=jnp.float64)
    traj_info = trajectory.TrajectoryInfo(
        top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
    traj_states = traj_info.get_states()

    ## Practice computing the local helical axis
    offset = 4
    down_neigh = 1
    up_neigh = 1

    body = traj_states[0]
    n = body.center.shape[0]
    n_bp = int(n // 2)
    cosines = [0.] * (n_bp - offset * 2 )
    cosines_counter = [0] * len(cosines)
    base_start = offset

    n_traj_states = len(traj_states)

    for idx in tqdm(range(n_traj_states)):
        body = traj_states[idx]
        get_cosines(body, cosines, cosines_counter, offset, 1, 1)

    for i in range(len(cosines)):
        print(f"{i} {cosines[i] / float(cosines_counter[i])}")



if __name__ == "__main__":
    run()
