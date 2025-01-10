import pdb
import numpy as onp
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap, jit, lax
from jax_md import space

from jax_dna.common import utils, topology, trajectory
from jax_dna.loss import persistence_length as jd_lp

@jit
def get_site_positions(body, is_rna=True):
    Q = body.orientation
    back_base_vectors = utils.Q_to_back_base(Q) # space frame, normalized
    base_normals = utils.Q_to_base_normal(Q) # space frame, normalized
    cross_prods = utils.Q_to_cross_prod(Q) # space frame, normalized

    if is_rna:
        # RNA values
        com_to_backbone_x = -0.4
        com_to_backbone_y = 0.2
        com_to_stacking = 0.34
        com_to_hb = 0.4
        back_sites = body.center + com_to_backbone_x*back_base_vectors + com_to_backbone_y*base_normals
        stack_sites = body.center + com_to_stacking * back_base_vectors
        base_sites = body.center + com_to_hb * back_base_vectors
    else:
        # In code (like DNA1)
        com_to_backbone = -0.4
        com_to_stacking = 0.34
        com_to_hb = 0.4
        back_sites = body.center + com_to_backbone*back_base_vectors
        stack_sites = body.center + com_to_stacking * back_base_vectors
        base_sites = body.center + com_to_hb * back_base_vectors

    return back_sites, stack_sites, base_sites




displacement_fn, shift_fn = space.free()



def calculate_groove_distance(back_sites, nucA_id, nucB_id):
    n = back_sites.shape[0]
    n_bp = n // 2
    # pos_back_A = s._strands[sidA]._nucleotides[nucA_id].get_pos_back()
    pos_back_A = back_sites[nucA_id]
    if nucB_id - 1 >= 0:
        # pos_back_B_left = s._strands[sidB]._nucleotides[nucB_id-1].get_pos_back()
        pos_back_B_left = back_sites[n_bp+nucB_id-1]
    else:
        pos_back_B_left = None
    # if nucB_id + 1 < len( s._strands[sidB]._nucleotides  ):
    if nucB_id + 1 < n_bp:
        # pos_back_B_right = s._strands[sidB]._nucleotides[nucB_id+1].get_pos_back()
        pos_back_B_right = back_sites[n_bp+nucB_id+1]
    else:
        pos_back_B_right = None

    # pos_back_B1 = s._strands[sidB]._nucleotides[nucB_id].get_pos_back()
    pos_back_B1 = back_sites[n_bp+nucB_id]

    candidate_distance = []

    if not (pos_back_B_left is None):
        plane_norm = pos_back_B1 - pos_back_B_left
        plane_norm = plane_norm / onp.linalg.norm(plane_norm)
        val_B = onp.dot(plane_norm, pos_back_A)
        t = val_B - onp.dot(pos_back_B1,plane_norm)
        if t > 0 or abs(t) > onp.linalg.norm(pos_back_B1 - pos_back_B_left):
            t = 0
            intersection_point = pos_back_B1
        else:
            intersection_point = pos_back_B1 + t * plane_norm
        candidate_distance.append(onp.linalg.norm(intersection_point - pos_back_A))
    if not (pos_back_B_right is None):
        plane_norm = pos_back_B1 - pos_back_B_right
        plane_norm = plane_norm / onp.linalg.norm(plane_norm)
        val_B = onp.dot(plane_norm, pos_back_A)
        t = val_B - onp.dot(pos_back_B1, plane_norm)
        if t > 0 or abs(t) > onp.linalg.norm(pos_back_B1 - pos_back_B_right):
            t = 0
            intersection_point = pos_back_B1
        else:
            intersection_point = pos_back_B1 + t * plane_norm
        candidate_distance.append(onp.linalg.norm(intersection_point - pos_back_A))

    return min(candidate_distance)


@jit
def calculate_groove_distance_jax(back_sites, nucA_id, nucB_id):
    n = back_sites.shape[0]
    n_bp = n // 2
    pos_back_A = back_sites[nucA_id]
    pos_back_B1 = back_sites[n_bp+nucB_id]

    candidate_distance = list()

    valid_left_pos = (nucB_id - 1 >= 0)
    pos_back_B_left = back_sites[n_bp+nucB_id-1]

    plane_norm = pos_back_B1 - pos_back_B_left
    plane_norm = plane_norm / jnp.linalg.norm(plane_norm)
    val_B = jnp.dot(plane_norm, pos_back_A)
    t = val_B - jnp.dot(pos_back_B1, plane_norm)

    intersection_point = jnp.where(
        (t > 0) | (jnp.abs(t) > jnp.linalg.norm(pos_back_B1 - pos_back_B_left)),
        pos_back_B1,
        pos_back_B1 + t * plane_norm
    )

    left_distance = jnp.linalg.norm(intersection_point - pos_back_A)
    MAX = 1e6
    left_distance = jnp.where(valid_left_pos, left_distance, MAX)
    # candidate_distance.append(left_distance)


    valid_right_pos = (nucB_id + 1 < n_bp)
    pos_back_B_right = back_sites[n_bp+nucB_id+1]
    plane_norm = pos_back_B1 - pos_back_B_right
    plane_norm = plane_norm / jnp.linalg.norm(plane_norm)
    val_B = jnp.dot(plane_norm, pos_back_A)
    t = val_B - jnp.dot(pos_back_B1, plane_norm)
    intersection_point = jnp.where(
        (t > 0) | (jnp.abs(t) > jnp.linalg.norm(pos_back_B1 - pos_back_B_right)),
        pos_back_B1,
        pos_back_B1 + t * plane_norm
    )

    right_distance = jnp.linalg.norm(intersection_point - pos_back_A)
    right_distance = jnp.where(valid_right_pos, right_distance, MAX)
    # candidate_distance.append(right_distance)

    candidate_distances = jnp.array([left_distance, right_distance])

    return jnp.min(candidate_distances)


def single(body, offset):
    n_bp = body.center.shape[0] // 2
    back_sites, stack_sites, base_sites = get_site_positions(body)

    all_small_grooves = list()
    all_big_grooves = list()
    for j in range(offset, n_bp-offset-2):

        # Compute mmgrooves_for_nuc(body, j, 0)
        A_bpos = back_sites[j]
        distances = list()
        for strand2_idx in range(n_bp, 2*n_bp):
            B_bpos = back_sites[strand2_idx]
            dist = space.distance(displacement_fn(A_bpos, B_bpos))
            distances.append(dist)

        local_max = list()
        local_mins = list()
        better_l_mins = list()

        for i in range(len(distances)-2):
            if distances[i+1] < distances[i] and distances[i+1] < distances[i+2]:
                local_mins.append([i+1, distances[i+1]])
                better_l_mins.append([i+1, calculate_groove_distance(back_sites, j, i+1)])
            elif distances[i+1] > distances[i] and distances[i+1] > distances[i+2]:
                local_max.append([i+1, distances[i+1]])

        opposite = -j - 1 + n_bp

        if len(local_mins) > 2:
            print('Detected more than 2 local mins....?')

        grooves = [0, 0]
        for i, val in better_l_mins:
            if i < opposite:
                grooves[0] = val
            else:
                grooves[1] = val
        if len(local_mins) > 2:
            grooves[0] = grooves[1] = 0

        gr = deepcopy(grooves)
        if(gr[0] > 0): # TODO: can mask for this
            all_small_grooves.append(gr[0])
        if(gr[1] > 0): # TODO: can mask for this
            all_big_grooves.append(gr[1])

    return all_small_grooves, all_big_grooves



def single_jax(body, offset):
    n_bp = body.center.shape[0] // 2
    back_sites, stack_sites, base_sites = get_site_positions(body)

    all_small_grooves = list()
    all_big_grooves = list()

    n_valid_small_grooves = 0
    n_valid_big_grooves = 0
    small_groove_sm = 0.0
    big_groove_sm = 0.0


    for j in range(offset, n_bp-offset-2):

        # Compute mmgrooves_for_nuc(body, j, 0)
        A_bpos = back_sites[j]

        a_dist_fn = lambda strand2_idx: space.distance(displacement_fn(A_bpos, back_sites[strand2_idx]))
        distances = vmap(a_dist_fn)(jnp.arange(n_bp, 2*n_bp))
        distances = onp.array(distances)

        opposite = n_bp - j - 1


        # PARITY WITH ORIGINAL IN JAX

        distances = jnp.array(distances)
        n_distances = distances.shape[0]

        def detect_grooves(carry, i):
            small_groove, big_groove, n_local_mins = carry

            strand_cond = (i+1 < opposite)
            local_min_cond = (distances[i+1] < distances[i]) & (distances[i+1] < distances[i+2])
            val = calculate_groove_distance_jax(back_sites, j, i+1)

            n_local_mins += jnp.where(local_min_cond, 1, 0)

            small_groove = jnp.where(
                n_local_mins > 2, 0,
                jnp.where(
                    strand_cond & local_min_cond,
                    val, small_groove
                )
            )

            big_groove = jnp.where(
                n_local_mins > 2, 0,
                jnp.where(
                    jnp.logical_not(strand_cond) & local_min_cond,
                    val, big_groove
                )
            )

            return (small_groove, big_groove, n_local_mins), None
        (small_groove, big_groove, n_local_mins), _ = lax.scan(detect_grooves, (0, 0, 0), jnp.arange(n_distances-2))
        grooves = [small_groove, big_groove]



        # ORIGINAL
        """
        n_local_mins = 0
        better_l_mins = list()

        for i in range(len(distances)-2):
            if distances[i+1] < distances[i] and distances[i+1] < distances[i+2]:
                n_local_mins += 1
                better_l_mins.append([i+1, calculate_groove_distance_jax(back_sites, j, i+1)])



        grooves = [0, 0]
        for i, val in better_l_mins:
            if i < opposite:
                grooves[0] = val
            else:
                grooves[1] = val

        # RK ADDED. Optional.
        # if n_local_mins == 2 and (grooves[0] == 0 or grooves[1] == 0):
        #     grooves = [0, 0]

        if n_local_mins > 2:
            print('Detected more than 2 local mins....?')
            grooves[0] = grooves[1] = 0
        """



        gr = deepcopy(grooves)
        if(gr[0] > 0): # TODO: can mask for this
            all_small_grooves.append(gr[0])
        if(gr[1] > 0): # TODO: can mask for this
            all_big_grooves.append(gr[1])

    return all_small_grooves, all_big_grooves




def run():
    basedir = Path("/home/ryan/Documents/harvard/research/petr-scripts/grooves/gr/ds-40bp-rna")
    # basedir = Path("output") / "test-rna-lp"
    assert(basedir.exists())
    top_path = basedir / "sys_rna.top"
    traj_path = basedir / "output.dat"

    top_info = topology.TopologyInfo(top_path, reverse_direction=False, is_rna=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq, is_rna=True), dtype=jnp.float64)
    traj_info = trajectory.TrajectoryInfo(
        top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
    traj_states = traj_info.get_states()

    ## Practice computing the local helical axis
    offset = 4
    scale = 1

    body = traj_states[0]
    n = body.center.shape[0]
    n_bp = int(n // 2)

    n_traj_states = len(traj_states)

    all_small_grooves, all_big_grooves = list(), list()
    all_small_grooves_jax, all_big_grooves_jax = list(), list()
    for idx in tqdm(range(n_traj_states)):
        body = traj_states[idx]
        idx_small_grooves, idx_big_grooves = single(body, offset)

        # all_small_grooves.append(idx_small_grooves)
        all_small_grooves += idx_small_grooves
        # all_big_grooves.append(idx_big_grooves)
        all_big_grooves += idx_big_grooves

        idx_small_grooves_jax, idx_big_grooves_jax = single_jax(body, offset)
        all_small_grooves_jax += idx_small_grooves_jax
        all_big_grooves_jax += idx_big_grooves_jax


    print('#Small_groove (+/- std) big_groove (+/- std)')
    small_mean = onp.mean(all_small_grooves)
    small_std = onp.std(all_small_grooves)
    big_mean = onp.mean(all_big_grooves)
    big_std = onp.std(all_big_grooves)
    print(f"{small_mean} {small_std} {big_mean} {big_std}")
    scale = 8.518
    print(f"{scale*(small_mean-0.7)} {scale*small_std} {scale*(big_mean-0.7)} {scale*big_std}")


    small_mean_jax = onp.mean(all_small_grooves_jax)
    small_std_jax = onp.std(all_small_grooves_jax)
    big_mean_jax = onp.mean(all_big_grooves_jax)
    big_std_jax = onp.std(all_big_grooves_jax)
    print(f"{scale*(small_mean_jax-0.7)} {scale*small_std_jax} {scale*(big_mean_jax-0.7)} {scale*big_std_jax}")






if __name__ == "__main__":
    run()
