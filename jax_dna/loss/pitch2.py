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



def compute_angle(quartet, back_sites: util.Array, local_helix_dir, displacement_fn):
    # a1 is h-bonded to b1, a2 is h-bonded to b2
    a1, b1, a2, b2 = quartet

    bb1 = displacement_fn(back_sites[b1], back_sites[a1])
    bb1_dir = bb1 / jnp.linalg.norm(bb1)
    bb2 = displacement_fn(back_sites[b2], back_sites[a2])
    bb2_dir = bb2 / jnp.linalg.norm(bb2)

    bb1_proj = displacement_fn(bb1, jnp.dot(local_helix_dir, bb1) * local_helix_dir)
    bb1_proj_dir = bb1_proj / jnp.linalg.norm(bb1_proj)
    bb2_proj = displacement_fn(bb2, jnp.dot(local_helix_dir, bb2) * local_helix_dir)
    bb2_proj_dir = bb2_proj / jnp.linalg.norm(bb2_proj)

    theta = jnp.arccos(utils.clamp(jnp.dot(bb1_proj_dir, bb2_proj_dir)))
    return theta

def compute_local_helical_axis(quartet, base_sites: util.Array, displacement_fn):
    # a1 is h-bonded to b1, a2 is h-bonded to b2
    a1, b1, a2, b2 = quartet

    midp_a1a2 = (base_sites[a1] + base_sites[b1]) / 2.0
    midp_b1b2 = (base_sites[a2] + base_sites[b2]) / 2.0

    dr = displacement_fn(midp_b1b2, midp_a1a2)
    return dr / jnp.linalg.norm(dr)


def get_all_angles(body, quartets, displacement_fn, com_to_hb, com_to_backbone_x, com_to_backbone_y):

    Q = body.orientation
    back_base_vectors = utils.Q_to_back_base(Q)
    cross_prods = utils.Q_to_cross_prod(Q)

    # Construct the base and back site positions in the space frame
    base_sites = body.center + com_to_hb * back_base_vectors
    # back_sites = body.center + com_to_backbone * back_base_vectors
    back_sites = body.center + com_to_backbone_x*back_base_vectors + com_to_backbone_y*cross_prods

    local_helical_axes = vmap(compute_local_helical_axis, (0, None, None))(quartets, base_sites, displacement_fn)
    angles = vmap(compute_angle, (0, None, 0, None))(quartets, back_sites, local_helical_axes, displacement_fn)

    return angles



class TestPitch(unittest.TestCase):
    test_data_basedir = Path("data/test-data")

    def test_pitch2(self):

        test_cases = [
            # (self.test_data_basedir / f"simple-helix-60bp", model1.com_to_hb, model1.com_to_backbone, 0.0),
            # (self.test_data_basedir / f"simple-helix-60bp-oxdna2", model2.com_to_hb, model2.com_to_backbone_x, model2.com_to_backbone_y)
            (self.test_data_basedir / f"simple-helix-60bp-oxdna2", model2.com_to_hb, model1.com_to_backbone, 0.0)
        ]

        for basedir, com_to_hb, com_to_backbone_x, com_to_backbone_y in test_cases:

            top_path = basedir / "sys.top"
            top_info = topology.TopologyInfo(top_path, reverse_direction=False)
            n = len(top_info.seq)

            traj_path = basedir / "output.dat"
            traj_info = trajectory.TrajectoryInfo(
                top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
            displacement_fn, _ = space.periodic(traj_info.box_size)
            traj_states = traj_info.get_states()

            for offset in [4]:

                quartets = utils.get_all_quartets(n_nucs_per_strand=n // 2)
                quartets = quartets[offset:-offset-1]
                n_quartets = quartets.shape[0]

                computed_angles = [
                    get_all_angles(body, quartets, displacement_fn, com_to_hb, com_to_backbone_x, com_to_backbone_y)
                    for body in traj_states
                ]
                state_avg_angles = [onp.mean(angles) for angles in computed_angles]

                avg_pitch = 2*onp.pi / onp.mean(state_avg_angles)

                print(f"- Average pitch (skipping {(offset+1)*2} quartets): {avg_pitch}")

                running_avg_angles = onp.cumsum(state_avg_angles) / onp.arange(1, len(state_avg_angles) + 1)
                running_avg_pitches = 2*onp.pi / running_avg_angles
                plt.plot(running_avg_pitches)
                plt.title(f"{(offset+1)*2} skipped quartets")
                plt.xlabel("num state")
                plt.ylabel("avg. pitch")
                plt.show()
                plt.close()

        return

if __name__ == "__main__":
    unittest.main()
