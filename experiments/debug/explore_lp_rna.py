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


def get_end_j(body, bp1_idx, bp2_idx, displacement_fn):

    n = body.center.shape[0]

    back_sites, stack_sites, base_sites = get_site_positions(body)

    nt11 = bp1_idx
    nt12 = n - 1 - bp1_idx

    nt21 = bp2_idx
    nt22 = n - 1 - bp2_idx

    first_midpos = (base_sites[nt11] + base_sites[nt12]) / 2.0
    last_midpos = (base_sites[nt21] + base_sites[nt22]) / 2.0

    r0N = displacement_fn(last_midpos, first_midpos)
    return onp.sqrt(onp.dot(r0N, r0N)) / float((bp2_idx - bp1_idx))



def get_cosines_jax(body, base_start, down_neigh, up_neigh, max_dist):

    back_sites, stack_sites, base_sites = get_site_positions(body)
    l_0 = get_localized_axis(body, back_sites, stack_sites, base_sites, base_start, down_neigh, up_neigh)

    corr_dists = jnp.arange(max_dist) # FIXME: max_dist is really one too big. It's really one more than the max dist
    def get_corr(dist):
        i = base_start + dist
        l_i = get_localized_axis(body, back_sites, stack_sites, base_sites, i, down_neigh, up_neigh)
        corr = jnp.dot(l_0, l_i)
        count = 1
        return corr, count
    body_corrs_jax, body_corr_counters_jax = vmap(get_corr)(corr_dists)

    return body_corrs_jax, body_corr_counters_jax



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
    box_size = traj_info.box_size

    n_traj_states = len(traj_states)

    displacement_fn, _ = space.periodic(box_size)

    end_distances = list()
    for idx in tqdm(range(n_traj_states)):
        body = traj_states[idx]
        get_cosines(body, cosines, cosines_counter, offset, 1, 1)
        # get_cosines_jax(body, cosines, cosines_counter, offset, 1, 1)

        end_dist = get_end_j(body, offset, n_bp - offset, displacement_fn)
        end_distances.append(end_dist)

    r2 = onp.mean(onp.array(end_distances)**2)
    pdb.set_trace()

    max_dist = len(cosines)
    traj_states = utils.tree_stack(traj_states)
    all_cosines_jax, all_cosines_counter_jax = vmap(get_cosines_jax, (0, None, None, None, None))(traj_states, offset, 1, 1, max_dist)
    cosines_jax = onp.sum(all_cosines_jax, axis=0)
    cosines_counter_jax = onp.sum(all_cosines_counter_jax, axis=0)

    for i in range(len(cosines)):
        # print(f"{i} {cosines[i] / float(cosines_counter[i])}")
        print(f"{i} {cosines_jax[i] / float(cosines_counter_jax[i])}")



if __name__ == "__main__":
    run()
