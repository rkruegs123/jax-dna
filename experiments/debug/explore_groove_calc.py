import pdb
import numpy as onp
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import random

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


        """
        The below isn't perfect, but it raises a point about what should really be going on -- you shoulud count the number of major and minor groove mins, and each one is only valid if there is one min. Just becaues there are two total mins doesn't mean that one is a minor groove and the other is a major groove.
        """
        # RK ADDED. Optional.
        # if len(local_mins) == 2 and (grooves[0] == 0 or grooves[1] == 0):
        #    grooves = [0, 0]

        gr = deepcopy(grooves)
        if(gr[0] > 0): # TODO: can mask for this
            all_small_grooves.append(gr[0])
        if(gr[1] > 0): # TODO: can mask for this
            all_big_grooves.append(gr[1])

    return all_small_grooves, all_big_grooves


@partial(jit, static_argnums=1)
def single_jax(body, offset, petrs_way=True):
    n_bp = body.center.shape[0] // 2
    # n_bp = 40
    back_sites, stack_sites, base_sites = get_site_positions(body)


    @jit
    def get_major_minor_grooves(j):
        # Compute mmgrooves_for_nuc(body, j, 0)
        A_bpos = back_sites[j]

        a_dist_fn = lambda strand2_idx: space.distance(displacement_fn(A_bpos, back_sites[strand2_idx]))
        distances = vmap(a_dist_fn)(jnp.arange(n_bp, 2*n_bp))

        opposite = n_bp - j - 1

        n_distances = distances.shape[0]

        def detect_grooves(carry, i):
            # small_groove, big_groove, n_local_mins = carry
            small_groove, big_groove, n_local_s1_mins, n_local_s2_mins = carry

            strand_cond = (i+1 < opposite)
            local_min_cond = (distances[i+1] < distances[i]) & (distances[i+1] < distances[i+2])
            val = calculate_groove_distance_jax(back_sites, j, i+1)


            # n_local_mins += jnp.where(local_min_cond, 1, 0)
            local_min_s1_cond = local_min_cond & strand_cond
            n_local_s1_mins += jnp.where(local_min_s1_cond, 1, 0)
            local_min_s2_cond = local_min_cond & jnp.logical_not(strand_cond)
            n_local_s2_mins += jnp.where(local_min_s2_cond, 1, 0)

            n_local_mins = n_local_s1_mins + n_local_s2_mins

            if petrs_way:
                small_groove = jnp.where(
                    n_local_mins > 2, 0,
                    jnp.where(
                        local_min_s1_cond,
                        val, small_groove
                    )
                )

                big_groove = jnp.where(
                    n_local_mins > 2, 0,
                    jnp.where(
                        local_min_s2_cond,
                        val, big_groove
                    )
                )
            else:
                small_groove = jnp.where(
                    n_local_s1_mins > 1, 0, # Note the inequality
                    jnp.where(
                        local_min_s1_cond,
                        val, small_groove
                    )
                )

                big_groove = jnp.where(
                    n_local_s2_mins > 1, 0, # Note the inequality
                    jnp.where(
                        local_min_s2_cond,
                        val, big_groove
                    )
                )

            return (small_groove, big_groove, n_local_s1_mins, n_local_s2_mins), None
        (small_groove, big_groove, _, _), _ = lax.scan(detect_grooves, (0, 0, 0, 0), jnp.arange(n_distances-2))

        small_groove_is_valid = (small_groove != 0)
        big_groove_is_valid = (big_groove != 0)

        return small_groove, big_groove, small_groove_is_valid, big_groove_is_valid

    all_small_grooves, all_big_grooves, valid_small_grooves, valid_big_grooves = vmap(get_major_minor_grooves)(jnp.arange(offset, n_bp-offset-2))
    return all_small_grooves, all_big_grooves, valid_small_grooves.astype(jnp.int32), valid_big_grooves.astype(jnp.int32)




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
    # scale = 1 # this isn't used, so just using this variable name for the angstrom scale.
    scale = 8.518

    body = traj_states[0]
    n = body.center.shape[0]
    n_bp = int(n // 2)

    n_traj_states = len(traj_states)

    all_small_grooves, all_big_grooves = list(), list()
    all_small_grooves_jax, all_big_grooves_jax = list(), list()

    all_idx_small_grooves_jax, all_idx_big_grooves_jax, all_idx_small_grooves_valid, all_idx_big_grooves_valid = list(), list(), list(), list()

    all_avg_small_groove_per_state = list()
    all_avg_big_groove_per_state = list()

    MAX_TRAJ_STATES = 1000
    n_eval_states = min(n_traj_states, MAX_TRAJ_STATES)

    # for idx in tqdm(range(n_traj_states)):
    compute_mean_every = 10
    running_avg_small = list()
    running_avg_big = list()
    running_avg_idxs = list()
    for idx in tqdm(range(n_eval_states)):
        body = traj_states[idx]
        idx_small_grooves, idx_big_grooves = single(body, offset)

        all_small_grooves += idx_small_grooves
        all_big_grooves += idx_big_grooves

        idx_small_grooves_jax, idx_big_grooves_jax, idx_small_grooves_valid, idx_big_grooves_valid = single_jax(body, offset)
        all_idx_small_grooves_jax.append(idx_small_grooves_jax)
        all_idx_big_grooves_jax.append(idx_big_grooves_jax)
        all_idx_small_grooves_valid.append(idx_small_grooves_valid)
        all_idx_big_grooves_valid.append(idx_big_grooves_valid)

        all_small_grooves_jax += list(idx_small_grooves_jax[idx_small_grooves_valid.nonzero()[0]])
        all_big_grooves_jax += list(idx_big_grooves_jax[idx_big_grooves_valid.nonzero()[0]])


        nonzero_idx_small = idx_small_grooves_valid.nonzero()[0]
        n_nonzero_small = nonzero_idx_small.shape[0]
        if n_nonzero_small != 0:
            state_mean_small_groove = idx_small_grooves_jax[idx_small_grooves_valid.nonzero()[0]].mean()
            all_avg_small_groove_per_state.append(state_mean_small_groove)

        nonzero_idx_big = idx_big_grooves_valid.nonzero()[0]
        n_nonzero_big = nonzero_idx_big.shape[0]
        if n_nonzero_big != 0:
            state_mean_big_groove = idx_big_grooves_jax[idx_big_grooves_valid.nonzero()[0]].mean()
            all_avg_big_groove_per_state.append(state_mean_big_groove)

        if idx % compute_mean_every == 0 and idx:
            running_avg_idxs.append(idx)

            small_mean_scaled = scale*(onp.mean(all_small_grooves)-0.7)
            running_avg_small.append(small_mean_scaled)
            big_mean_scaled = scale*(onp.mean(all_big_grooves)-0.7)
            running_avg_big.append(big_mean_scaled)

    plt.plot(running_avg_idxs, running_avg_small, label="Minor")
    plt.plot(running_avg_idxs, running_avg_big, label="Major")
    plt.xlabel("State Index")
    plt.ylabel("Groove Width (A)")
    plt.legend()
    plt.show()
    plt.close()


    print('\nSmall_groove\t(+/- std)\tbig_groove\t(+/- std)')
    small_mean = onp.mean(all_small_grooves)
    small_std = onp.std(all_small_grooves)
    big_mean = onp.mean(all_big_grooves)
    big_std = onp.std(all_big_grooves)
    print(f"\nReference (unscaled):")
    print(f"- {small_mean}\t{small_std}\t{big_mean}\t{big_std}")

    print(f"\nReference (scaled):")
    print(f"- {scale*(small_mean-0.7)}\t{scale*small_std}\t{scale*(big_mean-0.7)}\t{scale*big_std}")


    small_mean_jax = onp.mean(all_small_grooves_jax)
    small_std_jax = onp.std(all_small_grooves_jax)
    big_mean_jax = onp.mean(all_big_grooves_jax)
    big_std_jax = onp.std(all_big_grooves_jax)
    print(f"\nComputed (scaled):")
    print(f"- {scale*(small_mean_jax-0.7)}\t{scale*small_std_jax}\t{scale*(big_mean_jax-0.7)}\t{scale*big_std_jax}")

    print(f"\nComputed (state averages, scaled):")
    print(f"- {scale*(onp.mean(all_avg_small_groove_per_state)-0.7)}\tNA\t\t\t{scale*(onp.mean(all_avg_big_groove_per_state)-0.7)}\tNA")

    # Reconstruct as expected value
    all_idx_small_grooves_jax = jnp.array(all_idx_small_grooves_jax)
    all_idx_big_grooves_jax = jnp.array(all_idx_big_grooves_jax)
    all_idx_small_grooves_valid = jnp.array(all_idx_small_grooves_valid)
    all_idx_big_grooves_valid = jnp.array(all_idx_big_grooves_valid)
    mock_energies = jnp.array([random.uniform(-20.0, -10.0) for _ in range(n_eval_states)])

    def compute_boltz(energy):
        new_energy = energy
        ref_energy = energy
        kt = 1.0
        beta = 1 / kt
        return jnp.exp(-beta * new_energy) / jnp.exp(-beta * ref_energy)
    state_boltzs = vmap(compute_boltz)(mock_energies)

    def get_extended_state_boltzs(state_boltz, state_grooves_valid):
        return state_boltz*state_grooves_valid

    ## Small groove
    extended_state_boltzs = vmap(get_extended_state_boltzs)(state_boltzs, all_idx_small_grooves_valid)
    denom = extended_state_boltzs.sum()
    extended_state_weights = extended_state_boltzs / denom

    small_mean_jax = jnp.multiply(extended_state_weights, all_idx_small_grooves_jax).sum()

    ## Big groove
    extended_state_boltzs = vmap(get_extended_state_boltzs)(state_boltzs, all_idx_big_grooves_valid)
    denom = extended_state_boltzs.sum()
    extended_state_weights = extended_state_boltzs / denom

    big_mean_jax = jnp.multiply(extended_state_weights, all_idx_big_grooves_jax).sum()

    print(f"\nComputed (reconstructed via expectation, scaled):")
    print(f"- {scale*(small_mean_jax-0.7)}\tNA\t\t\t{scale*(big_mean_jax-0.7)}\tNA")


    """
    Note: the philosophy here is that we treat each valid sample as it's own "state", but assign identical weights to all valid samples that are from the same state where that weight is the weight correpsonding to the state it comes from
    """


if __name__ == "__main__":
    run()
