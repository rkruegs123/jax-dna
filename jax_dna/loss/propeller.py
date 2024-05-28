# ruff: noqa
# fmt: off
import pdb
import unittest
from pathlib import Path
import numpy as onp

import jax.numpy as jnp
from jax import vmap
from jax_md import util, rigid_body

from jax_dna.common import utils, trajectory, topology

from jax.config import config
config.update("jax_enable_x64", True)


# Tom's these, botrtom page 57. Potentially averaged from ref 162.
# Note: there seem to be conflicting values on this. Some other citations/values are the following:
# - https://people.bu.edu/mfk/restricted566/dnastructure.pdf -- 12.6 (Table 1, 15.0 deg)
TARGET_PROPELLER_TWIST = 21.7


def compute_single_propeller_twist(bp: util.Array, base_normals: util.Array):
    """
    Computes the propeller twist of a base pair. The propeller
    twist is defined as the angle between the normal vectors
    of h-bonded bases

    Args:
    - bp: a 2-dimensional array containing the indices of the h-bonded nucleotides
    - base_normals: the base normal vectors of the entire body
    """

    # get the normal vectors of the h-bonded bases
    bp1, bp2 = bp
    nv1 = base_normals[bp1]
    nv2 = base_normals[bp2]

    # compute angle between base normal vectors
    theta = jnp.arccos(utils.clamp(jnp.dot(nv1, nv2)))
    return theta

def get_all_p_twists(body: rigid_body.RigidBody, base_pairs: util.Array):
    base_normals = utils.Q_to_base_normal(body.orientation)
    all_p_twists_rad = vmap(compute_single_propeller_twist, (0, None))(base_pairs, base_normals)
    all_p_twists_deg = 180.0 - (all_p_twists_rad * 180.0 / jnp.pi)
    return all_p_twists_deg


def get_propeller_loss_fn(base_pairs, target_propeller_twist=TARGET_PROPELLER_TWIST):
    def get_avg_p_twist(body: rigid_body.RigidBody):
        all_p_twists_deg = get_all_p_twists(body, base_pairs)
        avg_p_twist_deg = jnp.mean(all_p_twists_deg)
        return avg_p_twist_deg

    def loss_fn(body: rigid_body.RigidBody):
        avg_p_twist_deg = get_avg_p_twist(body)
        return (avg_p_twist_deg - target_propeller_twist)**2

    return get_avg_p_twist, loss_fn

class TestPropeller(unittest.TestCase):
    test_data_basedir = Path("data/test-data")

    def check_avg_propeller(self, basedir, top_fname, traj_fname, base_pairs,
                            num_states_to_eval=40, tol_degrees=2):
        # Load the system
        top_path = basedir / "generated.top"
        top_info = topology.TopologyInfo(top_path, reverse_direction=False)

        traj_path = basedir / "output.dat"
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
        traj_states = traj_info.get_states()

        states_to_eval = traj_states[-num_states_to_eval:]
        all_p_twists = list()
        for body in states_to_eval:
            all_p_twists_deg = get_all_p_twists(body, base_pairs)
            avg_p_twist_deg = jnp.mean(all_p_twists_deg)
            all_p_twists.append(avg_p_twist_deg)

        mean_p_twist = onp.mean(all_p_twists)
        self.assertTrue(onp.abs(mean_p_twist - TARGET_PROPELLER_TWIST) < tol_degrees)


    def test_propeller(self):
        simple_helix_bps = jnp.array([[1, 14], [2, 13], [3, 12],
                                      [4, 11], [5, 10], [6, 9]])
        simple_helix_test = (self.test_data_basedir / "simple-helix",
                             "generated.top", "output.dat",
                             simple_helix_bps)
        propeller_tests = [simple_helix_test]
        for basedir, top_fname, traj_fname, base_pairs in propeller_tests:
            self.check_avg_propeller(basedir, top_fname, traj_fname, base_pairs)

if __name__ == "__main__":
    unittest.main()
