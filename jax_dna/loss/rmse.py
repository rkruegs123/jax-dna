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



from oxDNA_analysis_tools.UTILS.RyeReader import describe, inbox, get_confs
from oxDNA_analysis_tools.align import svd_align



def my_svd_align_jax(ref_coords, coords, indexes):
    ref_center = jnp.zeros(3)

    av1 = ref_center
    av2 = jnp.mean(coords[0][indexes], axis=0)
    coords[0] = coords[0] - av2

    # correlation matrix
    a = jnp.dot(onp.transpose(coords[0][indexes]), ref_coords - av1)
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


def single_mfs(state, target_positions, indices):
    back_base_vectors = utils.Q_to_back_base(state.orientation) # a1s
    base_normals = utils.Q_to_base_normal(state.orientation) # a3

    conf = onp.asarray([state.center, back_base_vectors, base_normals])
    aligned_conf = my_svd_align_jax(target_positions[indices], conf, indices)[0]
    MF = jnp.power(jnp.linalg.norm(aligned_conf - target_positions, axis=1), 2)
    return MF




def compute_rmses(traj_path, target_path, top_path, displacement_fn):
    ## Processing for OAT
    ti_ref, di_ref = describe(None, str(target_path))
    top_info, traj_info = describe(None, str(traj_path))

    mean_conf = get_confs(ti_ref, di_ref, 0, 1)[0]

    indexes = list(range(top_info.nbases))

    mean_conf = inbox(mean_conf)
    ref_cms = onp.mean(mean_conf.positions[indexes], axis=0)
    mean_conf.positions -= ref_cms

    ## Processing for JAX-DNA
    top_info_jdna = topology.TopologyInfo(top_path, reverse_direction=False)
    n = len(top_info_jdna.seq)

    traj_info_jdna = trajectory.TrajectoryInfo(
        top_info_jdna, read_from_file=True, traj_path=traj_path, reverse_direction=False)
    traj_states_jdna = traj_info_jdna.get_states()

    target_info_jdna = trajectory.TrajectoryInfo(
        top_info_jdna, read_from_file=True, traj_path=target_path, reverse_direction=False)
    target_state_jdna = target_info_jdna.get_states()[0]


    MFs = list()
    MFs_jdna = list()
    MFs_jdna_full = list()
    for c_idx in tqdm(range(traj_info.nconfs)):

        ## OAT calculation
        conf = get_confs(top_info, traj_info, c_idx, 1)[0]
        conf = inbox(conf, center=True)
        conf = onp.asarray([conf.positions, conf.a1s, conf.a3s])
        aligned_conf = svd_align(mean_conf.positions[indexes], deepcopy(conf), indexes, ref_center=onp.zeros(3))[0]
        MF = onp.power(onp.linalg.norm(aligned_conf - mean_conf.positions, axis=1), 2)
        MFs.append(MF)

        ## JAX-DNA calculation, full jax
        state = traj_states_jdna[c_idx]
        MFs_jax = single_mfs(state, jnp.array(mean_conf.positions), jnp.array(indexes))
        MFs_jdna_full.append(MFs_jax)

        assert(onp.allclose(MFs_jax, MF))

    MFs = onp.array(MFs)

    RMSDs = onp.sqrt(onp.mean(MFs, axis=1)) * 0.8518
    pdb.set_trace()
    RMSFs = onp.sqrt(onp.mean(MFs, axis=0)) * 0.8518

    return (RMSDs, RMSFs)




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
            top_info = topology.TopologyInfo(top_path, reverse_direction=False)
            n = len(top_info.seq)

            traj_path = basedir / "output.dat"
            traj_info = trajectory.TrajectoryInfo(
                top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
            traj_states = traj_info.get_states()

            target_path = basedir / "start.conf"
            target_info = trajectory.TrajectoryInfo(
                top_info, read_from_file=True, traj_path=target_path, reverse_direction=False)
            target_state = target_info.get_states()[0]

            displacement_fn, _ = space.periodic(traj_info.box_size)


            ## Compute RMSDs using OAT

            ti_ref, di_ref = describe(None, str(target_path))
            ti_trj, di_trj = describe(None, str(traj_path))

            ref_conf = get_confs(ti_ref, di_ref, 0, 1)[0]
            RMSDs_default, RMSFs_default = deviations(di_trj, ti_trj, ref_conf, indexes=[], ncpus=1)
            RMSDs_my, RMSFs_my = compute_rmses(traj_path, target_path, top_path, displacement_fn)
            assert((RMSDs_my == RMSDs_default).all())
            assert((RMSFs_my == RMSFs_default).all())





        return

if __name__ == "__main__":
    unittest.main()

    """
    compute RMSD directly, no RMSF. Then change error checking appropriately.

    vmap

    remove indicies. Assume we care about  the whole structure every time.

    put into real functionaltiy/API
    """
