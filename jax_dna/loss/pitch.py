# ruff: noqa
# fmt: off
import pdb
import unittest
import pandas as pd
from pathlib import Path
import numpy as onp
import matplotlib.pyplot as plt

import jax
from jax import vmap
import jax.numpy as jnp
from jax_md import rigid_body, util, space

from jax_dna.common import utils, topology, trajectory
from jax_dna.dna1 import model as model1
from jax_dna.dna2 import model as model2

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)


# DNA Structure and Function, R. Sinden, 1st ed
# Table 1.3, Pg 27
TARGET_AVG_PITCH = 10.5


def compute_single_pitch(quartet, base_sites: util.Array, displacement_fn):
    """
    Computes the pitch of a pair of base pairs.

    The pitch is defined as the angle between the projections
    of two base-base vectors in a plane perpendicular to the
    helical axis for two contiguous base pairs.

    Args:
    - quartet: a 4-dimensional array where the first two
    elements define the first base pair and the second two
    elements define the second base pair
    - base_sites: an array of base site positions for the
    entire system
    """

    # a1 is h-bonded to b1, a2 is h-bonded to b2
    a1, b1, a2, b2 = quartet

    # get base-base vectors for each base pair, 1 and 2
    bb1 = displacement_fn(base_sites[b1], base_sites[a1])
    bb2 = displacement_fn(base_sites[b2], base_sites[a2])

    # get "average" helical axis
    a2a1 = displacement_fn(base_sites[a1], base_sites[a2])
    b2b1 = displacement_fn(base_sites[b1], base_sites[b2])
    local_helix = 0.5 * (a2a1 + b2b1)
    local_helix_dir = local_helix / jnp.linalg.norm(local_helix)

    # project each of the base-base vectors onto the plane perpendicular to the helical axis
    bb1_projected = displacement_fn(bb1, jnp.dot(bb1, local_helix_dir) * local_helix_dir)
    bb2_projected = displacement_fn(bb2, jnp.dot(bb2, local_helix_dir) * local_helix_dir)

    bb1_projected_dir = bb1_projected / jnp.linalg.norm(bb1_projected)
    bb2_projected_dir = bb2_projected / jnp.linalg.norm(bb2_projected)

    # find the angle between the projections of the base-base vectors in the plane perpendicular to the "local/average" helical axis
    theta = jnp.arccos(utils.clamp(jnp.dot(bb1_projected_dir, bb2_projected_dir)))
    return theta

def get_all_pitches(body, quartets, displacement_fn, com_to_hb):
    # Construct the base site position in the body frame
    base_site_bf = jnp.array([com_to_hb, 0.0, 0.0])

    # Compute the space-frame base sites
    base_sites = body.center + rigid_body.quaternion_rotate(
        body.orientation, base_site_bf)

    # Compute the pitches for all quartets
    all_pitches = vmap(compute_single_pitch, (0, None, None))(
        quartets, base_sites, displacement_fn)

    return all_pitches


def get_pitch_loss_fn(quartets, displacement_fn, com_to_hb,
                      target_avg_pitch=TARGET_AVG_PITCH):

    n_quartets = quartets.shape[0]


    def get_avg_pitch(body: rigid_body.RigidBody):
        pitches = get_all_pitches(body, quartets, displacement_fn, com_to_hb)
        num_turns = jnp.sum(pitches) / (2*jnp.pi)
        avg_pitch = (n_quartets+1) / num_turns
        return avg_pitch

    def loss_fn(body: rigid_body.RigidBody):
        avg_pitch = get_avg_pitch(body)
        return (target_avg_pitch - avg_pitch)**2

    return get_avg_pitch, loss_fn

class TestPitch(unittest.TestCase):
    test_data_basedir = Path("data/test-data")

    def check_pitches(self, basedir, top_fname, traj_fname, pitches_fname,
                      quartets, tol_places=4):
        top_path = basedir / "generated.top"
        traj_path = basedir / "output.dat"
        pitch_path = basedir / "pitch.dat"

        # Load the ground truth pitches
        oxdna_pitches = pd.read_csv(
            pitch_path,
            names=["q1", "q2", "q3",
                   "q4", "q5", "q6"],
            delim_whitespace=True)
        oxdna_pitches = oxdna_pitches.iloc[1: , :] # drop the first line

        # Compute pitches
        # note: for simplicity we do not read in the reverse direction
        # so that we don't have to reverse the quartet indices
        top_info = topology.TopologyInfo(top_path, reverse_direction=False)

        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
        displacement_fn, _ = space.periodic(traj_info.box_size)
        traj_states = traj_info.get_states()
        computed_pitches = [
            # note: we assume DNA1
            get_all_pitches(body, quartets, displacement_fn, model1.com_to_hb)
            for body in traj_states
        ]

        # Check for equality
        for i, (idx, row) in enumerate(oxdna_pitches.iterrows()):
            ith_oxdna_pitches = row.to_numpy()
            ith_computed_pitches = computed_pitches[i]
            for oxdna_pitch, computed_pitch in zip(ith_oxdna_pitches, ith_computed_pitches):
                self.assertAlmostEqual(oxdna_pitch, computed_pitch, places=tol_places)

    def test_pitch(self):
        simple_helix_quartets = jnp.array([
            [1, 14, 2, 13], [2, 13, 3, 12],
            [3, 12, 4, 11], [4, 11, 5, 10],
            [5, 10, 6, 9], [6, 9, 7, 8]])
        simple_helix_test = (self.test_data_basedir / "simple-helix",
                             "generated.top", "output.dat", "pitch.dat",
                             simple_helix_quartets)
        simple_helix_test_oxdna2 = (self.test_data_basedir / "simple-helix-oxdna2",
                                    "generated.top", "output.dat", "pitch.dat",
                                    simple_helix_quartets)
        pitch_tests = [simple_helix_test, simple_helix_test_oxdna2]
        for basedir, top_fname, traj_fname, pitches_fname, quartets in pitch_tests:
            self.check_pitches(basedir, top_fname, traj_fname, pitches_fname, quartets)

    def test_long_duplex(self):
        # for n_bp in [30, 60, 80]:
        # for n_bp in [60, 80]:
        for n_bp in [60]:
            basedir = self.test_data_basedir / f"simple-helix-{n_bp}bp"
            # basedir = self.test_data_basedir / f"simple-helix-{n_bp}bp-oxdna2"
            top_path = basedir / "sys.top"
            top_info = topology.TopologyInfo(top_path, reverse_direction=False)
            n = len(top_info.seq)

            traj_path = basedir / "output.dat"
            traj_info = trajectory.TrajectoryInfo(
                top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
            displacement_fn, _ = space.periodic(traj_info.box_size)
            traj_states = traj_info.get_states()

            print(f"----- Number of base pairs: {n_bp} -----")

            for n_skip_quartets in [3, 5, 10]:

                quartets = utils.get_all_quartets(n_nucs_per_strand=n // 2)
                quartets = quartets[n_skip_quartets:-n_skip_quartets]
                n_quartets = quartets.shape[0]

                computed_pitches = [
                    get_all_pitches(body, quartets, displacement_fn, model1.com_to_hb)
                    for body in traj_states
                ]

                state_avg_pitches = list()
                for pitches in computed_pitches:
                    num_turns = jnp.sum(pitches) / (2*jnp.pi)
                    state_avg_pitch = (n_quartets+1) / num_turns
                    state_avg_pitches.append(state_avg_pitch)

                avg_pitch = onp.mean(state_avg_pitches)
                print(f"- Average pitch (skipping {n_skip_quartets} quartets): {avg_pitch}")

                running_avg = onp.cumsum(state_avg_pitches) / onp.arange(1, len(computed_pitches) + 1)
                plt.plot(running_avg)
                plt.title(f"{n_bp} base pairs, {n_skip_quartets} skipped quartets")
                plt.xlabel("num state")
                plt.ylabel("avg. pitch")
                plt.show()
                plt.close()

        return

    def test_oxdna2(self):
        basedir = self.test_data_basedir / f"simple-helix-60bp-oxdna2"
        top_path = basedir / "sys.top"
        top_info = topology.TopologyInfo(top_path, reverse_direction=False)
        n = len(top_info.seq)

        traj_path = basedir / "output.dat"
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
        displacement_fn, _ = space.periodic(traj_info.box_size)
        traj_states = traj_info.get_states()


        for n_skip_quartets in [3, 5, 10]:

            quartets = utils.get_all_quartets(n_nucs_per_strand=n // 2)
            quartets = quartets[n_skip_quartets:-n_skip_quartets]
            n_quartets = quartets.shape[0]

            computed_pitches = [
                get_all_pitches(body, quartets, displacement_fn, model1.com_to_hb)
                for body in traj_states
            ]

            state_avg_pitches = list()
            for pitches in computed_pitches:
                num_turns = jnp.sum(pitches) / (2*jnp.pi)
                state_avg_pitch = (n_quartets+1) / num_turns
                state_avg_pitches.append(state_avg_pitch)

            avg_pitch = onp.mean(state_avg_pitches)
            print(f"- Average pitch (skipping {n_skip_quartets} quartets): {avg_pitch}")

            running_avg = onp.cumsum(state_avg_pitches) / onp.arange(1, len(computed_pitches) + 1)
            plt.plot(running_avg)
            plt.title(f"{n_skip_quartets} skipped quartets")
            plt.xlabel("num state")
            plt.ylabel("avg. pitch")
            plt.show()
            plt.close()

        return

if __name__ == "__main__":
    unittest.main()
