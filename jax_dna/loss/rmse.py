import pdb
import unittest
import pandas as pd
from pathlib import Path
import numpy as onp
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

import jax
from jax import vmap
import jax.numpy as jnp
from jax_md import rigid_body, util, space

from jax_dna.common import utils, topology, trajectory
from jax_dna.dna1 import model as model1
from jax_dna.dna2 import model as model2

jax.config.update("jax_enable_x64", True)




def svd_align(ref_coords, coords):

    n_nt = coords.shape[1]
    indexes = jnp.arange(n_nt)

    ref_center = jnp.zeros(3)

    av1 = ref_center
    av2 = jnp.mean(coords[0][indexes], axis=0)
    coords = coords.at[0].set(coords[0] - av2)
    # coords[0] = coords[0] - av2

    # correlation matrix
    a = jnp.dot(jnp.transpose(coords[0][indexes]), ref_coords - av1)
    u, _, vt = jnp.linalg.svd(a)
    rot = jnp.transpose(jnp.dot(jnp.transpose(vt), jnp.transpose(u)))

    # check if we have found a reflection
    found_reflection = (jnp.linalg.det(rot) < 0)
    vt = jnp.where(found_reflection, vt.at[2].set(-vt[2]), vt)
    rot = jnp.where(found_reflection,
                    jnp.transpose(jnp.dot(jnp.transpose(vt), jnp.transpose(u))),
                    rot)
    tran = av1
    return  (jnp.dot(coords[0], rot) + tran,
             jnp.dot(coords[1], rot),
             jnp.dot(coords[2], rot))


def compute_fluc_sq(state, target_positions):
    back_base_vectors = utils.Q_to_back_base(state.orientation) # a1s
    base_normals = utils.Q_to_base_normal(state.orientation) # a3

    conf = jnp.asarray([state.center, back_base_vectors, base_normals])
    aligned_conf = svd_align(target_positions, conf)[0]
    fluc_sq = jnp.power(jnp.linalg.norm(aligned_conf - target_positions, axis=1), 2)
    return fluc_sq


def compute_rmses(traj_states, target_state, top_info):

    n_states = traj_states.center.shape[0]

    # Center the target state
    target_state = target_state.set(center=target_state.center - jnp.mean(target_state.center, axis=0))

    # Compute squared fluctuations
    """
    all_fluc_sqs = list()
    for c_idx in tqdm(range(n_states)):

        ## JAX-DNA calculation, full jax
        state = traj_states[c_idx]
        fluc_sq = compute_fluc_sq(state, target_state.center)
        all_fluc_sqs.append(fluc_sq)
    all_fluc_sqs = jnp.array(all_fluc_sqs)
    """
    all_fluc_sqs = vmap(compute_fluc_sq, (0, None))(traj_states, target_state.center)

    # Average
    rmsds = jnp.sqrt(jnp.mean(all_fluc_sqs, axis=1)) * utils.nm_per_oxdna_length
    rmsfs = jnp.sqrt(jnp.mean(all_fluc_sqs, axis=0)) * utils.nm_per_oxdna_length

    return (rmsds, rmsfs)



def compute_mean_structure(traj_states):
    n_states = traj_states.center.shape[0]

    s0 = traj_states[0]
    n_nt = s0.center.shape[0]
    s0_centered = s0.set(center=s0.center - jnp.mean(s0.center, axis=0))


    # states_to_align = traj_states[1:]

    @jax.jit
    def align_state(state):
        back_base_vectors = utils.Q_to_back_base(state.orientation) # a1s
        base_normals = utils.Q_to_base_normal(state.orientation) # a3
        conf = jnp.asarray([state.center, back_base_vectors, base_normals])
        aligned_conf = svd_align(s0_centered.center, conf)
        return aligned_conf

    state_sum = None
    # for state in states_to_align:
    # for s_idx in tqdm(range(n_states-1), desc="Aligning"):
    for s_idx in tqdm(range(n_states), desc="Aligning"):
        # state = states_to_align[s_idx]
        state = traj_states[s_idx]
        aligned = align_state(state)
        aligned = onp.array(aligned)
        if state_sum is None:
            state_sum = aligned
        else:
            state_sum += aligned

    mean_state = state_sum / n_states
    pos, a1s, a3s = mean_state

    a1s = onp.array([v/onp.linalg.norm(v) for v in a1s])
    a3s = onp.array([v/onp.linalg.norm(v) for v in a3s])

    # Convert to rigid body
    R = onp.empty((n_nt, 3), dtype=onp.float64)
    quat = onp.empty((n_nt, 4), dtype=onp.float64)
    for nt_idx in range(n_nt):
        back_base_vector = a1s[nt_idx]
        base_normal = a3s[nt_idx]
        com = pos[nt_idx]

        alpha, beta, gamma = trajectory.principal_axes_to_euler_angles(
            back_base_vector,
            onp.cross(base_normal, back_base_vector),
            base_normal
        )

        q0, q1, q2, q3 = trajectory.euler_angles_to_quaternion(alpha, beta, gamma)

        R[nt_idx, :] = com
        quat[nt_idx, :] = onp.array([q0, q1, q2, q3])

    R = jnp.array(R, dtype=jnp.float64)
    quat = jnp.array(quat, dtype=jnp.float64)
    body = rigid_body.RigidBody(R, rigid_body.Quaternion(quat))

    return body, (pos, a1s, a3s)




class TestRMSE(unittest.TestCase):
    test_data_basedir = Path("data/test-data")

    def test_rmse(self):

        from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs
        from oxDNA_analysis_tools.deviations import deviations

        test_cases = [
            (self.test_data_basedir / f"simple-helix", model1.com_to_hb)
        ]

        for basedir, com_to_hb in test_cases:

            top_path = basedir / "generated.top"
            traj_path = basedir / "output.dat"
            target_path = basedir / "start.conf"


            ## Compute RMSDs using OAT
            ti_ref, di_ref = describe(None, str(target_path))
            ti_trj, di_trj = describe(None, str(traj_path))

            ref_conf = get_confs(ti_ref, di_ref, 0, 1)[0]
            RMSDs_oat, RMSFs_oat = deviations(di_trj, ti_trj, ref_conf, indexes=[], ncpus=1)

            ## Compute RMSDs using JAX
            top_info = topology.TopologyInfo(top_path, reverse_direction=False)
            n = len(top_info.seq)

            traj_info = trajectory.TrajectoryInfo(
                top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
            traj_states = traj_info.get_states()
            traj_states = utils.tree_stack(traj_states)

            target_info = trajectory.TrajectoryInfo(
                top_info, read_from_file=True, traj_path=target_path, reverse_direction=False)
            target_state = target_info.get_states()[0]

            RMSDs_jax, RMSFs_jax = compute_rmses(traj_states, target_state, top_info)

            ## Check for equality
            assert(onp.allclose(RMSDs_jax, RMSDs_oat))
            assert(onp.allclose(RMSFs_jax, RMSFs_oat))

        return

    def test_mean(self):
        from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs
        from oxDNA_analysis_tools.mean import mean

        test_cases = [
            (self.test_data_basedir / f"simple-helix", model1.com_to_hb)
        ]

        for basedir, com_to_hb in test_cases:

            top_path = basedir / "generated.top"
            traj_path = basedir / "output.dat"


            ## Compute Mean structure using OAT
            ti_trj, di_trj = describe(None, str(traj_path))
            ref_conf = get_confs(ti_trj, di_trj, 0, 1)[0]

            mean_conf_oat = mean(di_trj, ti_trj, ref_conf, indexes=[], ncpus=1)

            ## Compute RMSDs using JAX
            top_info = topology.TopologyInfo(top_path, reverse_direction=False)
            n = len(top_info.seq)


            traj_info = trajectory.TrajectoryInfo(
                top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False
            )
            traj_states = traj_info.get_states()
            traj_states = utils.tree_stack(traj_states)

            mean_body_calc, (calc_pos, calc_a1s, calc_a3s) = compute_mean_structure(traj_states)

            assert(onp.allclose(calc_a1s, mean_conf_oat.a1s))
            assert(onp.allclose(calc_a3s, mean_conf_oat.a3s))
            assert(onp.allclose(calc_pos, mean_conf_oat.positions))


        return

if __name__ == "__main__":
    unittest.main()
