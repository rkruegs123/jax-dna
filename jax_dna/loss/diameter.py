import pdb
import unittest
import pandas as pd
from pathlib import Path
import numpy as onp
import matplotlib.pyplot as plt

from jax import vmap
import jax.numpy as jnp
from jax_md import rigid_body, util, space

from jax_dna.common import utils, topology, trajectory
from jax_dna.dna1 import model as model1
from jax_dna.dna2 import model as model2

from jax.config import config
config.update("jax_enable_x64", True)



def compute_diameter(bp, back_sites: util.Array, displacement_fn, sigma_backbone):
    # a1 is h-bonded to b1, a2 is h-bonded to b2
    a1, b1 = bp
    dr = displacement_fn(back_sites[a1], back_sites[b1])
    return space.distance(dr) + sigma_backbone



def get_avg_diameters(body, bps, displacement_fn, com_to_backbone_x, com_to_backbone_y, sigma_backbone):

    Q = body.orientation
    back_base_vectors = utils.Q_to_back_base(Q)
    cross_prods = utils.Q_to_cross_prod(Q)
    back_sites = body.center + com_to_backbone_x*back_base_vectors + com_to_backbone_y*cross_prods

    diameters = vmap(compute_diameter, (0, None, None, None))(bps, back_sites, displacement_fn, sigma_backbone)

    return jnp.mean(diameters)




class TestDiameter(unittest.TestCase):
    test_data_basedir = Path("data/test-data")

    sigma_backbone = 0.70

    def test_diameter(self):

        test_cases = [
            # (self.test_data_basedir / f"simple-helix-60bp-oxdna2", model2.com_to_backbone_x, model2.com_to_backbone_y),
            (self.test_data_basedir / f"simple-helix-60bp-oxdna2", model1.com_to_backbone, 0.0), # note that even for DNA2 we still use the DNA1 backbone site for calculation
            (self.test_data_basedir / f"simple-helix-60bp", model1.com_to_backbone, 0.0)
        ]

        for basedir, com_to_backbone_x, com_to_backbone_y in test_cases:

            top_path = basedir / "sys.top"
            top_info = topology.TopologyInfo(top_path, reverse_direction=False)
            n = len(top_info.seq)

            traj_path = basedir / "output.dat"
            traj_info = trajectory.TrajectoryInfo(
                top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
            displacement_fn, _ = space.periodic(traj_info.box_size)
            traj_states = traj_info.get_states()

            for offset in [4]:

                bps = utils.get_all_bps(n_nucs_per_strand=n // 2)
                bps = bps[offset:-offset-1]

                computed_diameters_ang = onp.array([
                    get_avg_diameters(body, bps, displacement_fn, com_to_backbone_x, com_to_backbone_y, self.sigma_backbone) * utils.ang_per_oxdna_length
                    for body in traj_states
                ])
                avg_diameter_ang = jnp.mean(computed_diameters_ang)

                print(f"- Average diameter in A (skipping {(offset+1)*2} bps): {avg_diameter_ang}")

                running_avg_diameters_ang = onp.cumsum(computed_diameters_ang) / onp.arange(1, len(computed_diameters_ang) + 1)
                plt.plot(running_avg_diameters_ang)
                plt.title(f"{(offset+1)*2} skipped bps")
                plt.xlabel("num state")
                plt.ylabel("avg. diameter (A)")
                plt.show()
                plt.close()

        return

if __name__ == "__main__":
    unittest.main()
