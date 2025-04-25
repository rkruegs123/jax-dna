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



def compute_rise(quartet, base_sites: util.Array, local_helix_dir, displacement_fn):
    # a1 is h-bonded to b1, a2 is h-bonded to b2
    a1, b1, a2, b2 = quartet


    midp1 = (base_sites[a1] + base_sites[b1]) / 2.0
    midp2 = (base_sites[a2] + base_sites[b2]) / 2.0

    dr = displacement_fn(midp2, midp1)
    rise = jnp.dot(dr, local_helix_dir)
    return rise


def compute_local_helical_axis(quartet, base_sites: util.Array, displacement_fn):
    # a1 is h-bonded to b1, a2 is h-bonded to b2
    a1, b1, a2, b2 = quartet

    midp_a1a2 = (base_sites[a1] + base_sites[b1]) / 2.0
    midp_b1b2 = (base_sites[a2] + base_sites[b2]) / 2.0

    dr = displacement_fn(midp_b1b2, midp_a1a2)
    return dr / jnp.linalg.norm(dr)


def get_avg_rises(body, quartets, displacement_fn, com_to_hb):

    Q = body.orientation
    back_base_vectors = utils.Q_to_back_base(Q)
    cross_prods = utils.Q_to_cross_prod(Q)

    # Construct the base and back site positions in the space frame
    base_sites = body.center + com_to_hb * back_base_vectors

    local_helical_axes = vmap(compute_local_helical_axis, (0, None, None))(quartets, base_sites, displacement_fn)
    rises = vmap(compute_rise, (0, None, 0, None))(quartets, base_sites, local_helical_axes, displacement_fn)

    return jnp.mean(rises)




class TestRise(unittest.TestCase):
    test_data_basedir = Path("data/test-data")

    def test_rise(self):

        test_cases = [
            (self.test_data_basedir / f"simple-helix-60bp-oxdna2", model2.com_to_hb),
            (self.test_data_basedir / f"simple-helix-60bp", model1.com_to_hb)
        ]

        for basedir, com_to_hb in test_cases:

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

                computed_rises_ang = onp.array([
                    get_avg_rises(body, quartets, displacement_fn, com_to_hb) * utils.ang_per_oxdna_length
                    for body in traj_states
                ])
                avg_rise_ang = jnp.mean(computed_rises_ang)

                print(f"- Average rise in A (skipping {(offset+1)*2} quartets): {avg_rise_ang}")

                running_avg_rises_ang = onp.cumsum(computed_rises_ang) / onp.arange(1, len(computed_rises_ang) + 1)
                plt.plot(running_avg_rises_ang)
                plt.title(f"{(offset+1)*2} skipped quartets")
                plt.xlabel("num state")
                plt.ylabel("avg. rise (A)")
                plt.show()
                plt.close()

        return

if __name__ == "__main__":
    unittest.main()
