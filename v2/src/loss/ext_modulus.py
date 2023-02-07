import pdb
from tqdm import tqdm

from jax.config import config as jax_config
import jax.numpy as jnp
import numpy as onp
from jax import jit, vmap

from jax_md import simulate
from jax_md import space
from jax_md import util
from jax_md import rigid_body
from jax_md.rigid_body import RigidBody, Quaternion

import sys
sys.path.append("v2/src/")
# pdb.set_trace()

from utils import DEFAULT_TEMP
from utils import back_site, stack_site, base_site
from utils import get_one_hot
from utils import clamp


from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo
from loader.get_params import get_default_params

from jax.config import config
config.update("jax_enable_x64", True)

Array=util.Array
FLAGS = jax_config.FLAGS
DYNAMICS_STEPS = 100

f32 = util.f32
f64 = util.f64

DTYPE = [f32]
if FLAGS.jax_enable_x64:
    DTYPE += [f64]


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from pathlib import Path


    from jaxopt import GaussNewton
    from utils import get_kt, DEFAULT_TEMP

    kT = get_kt(t=DEFAULT_TEMP)

    forces = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    lens = jnp.array([33.47837202, 34.63343703, 35.13880103, 35.48323747,
                      35.73831831, 35.95160141, 36.12325301, 36.27638711])


    def coth(x):
        return 1 / jnp.tanh(x)

    def WLC(coeffs, x_data, force_data, kT):
        # coeffs = [L0, Lp, K]
        y = ((force_data * coeffs[0]**2)/(coeffs[1]*kT))**(1/2)
        residual = x_data - coeffs[0] * (1 + force_data/coeffs[2] - kT/(2*force_data) * (1 + y*coth(y)))
        return residual


    """
    1 unit of force      	4.863x10 − 11 {\displaystyle ^{-11}} N = 48.63 pN
    1 unit of length 	        8.518x10 − 10 {\displaystyle ^{-10}} m = 0.8518 nm
    1 nm is 10 A

    True K -- 2166 pN, 44.540407 simulation force units
    True L0 -- 339.6 A, 39.86851 simulation length units
    True Lps -- 431.0 A, 50.59873 simulation length units

    """
    x_init = jnp.array([39.87, 50.60, 44.54]) # initialize to the true values
    # x_init = jnp.array([45.0, 55.0, 40.0]) # initialize to close-to-true values

    gn = GaussNewton(residual_fun=WLC)
    gn_sol = gn.run(x_init, x_data=lens, force_data=forces, kT=kT).params



    pdb.set_trace()





    """
    dir_end1=(100, 119)
    dir_end2=(9, 210)
    dir_force_axis=jnp.array([0, 0, 1])
    basedir = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/elastic-mod-sims-2-6/bak")
    force_to_dir = {
        0.05: "langevin_2023-02-06_04-00-50_n10000000",
        0.10: "langevin_2023-02-06_04-01-03_n10000000",
        0.15: "langevin_2023-02-06_04-01-18_n10000000",
        0.20: "langevin_2023-02-06_04-01-49_n10000000",
        0.25: "langevin_2023-02-06_04-01-39_n10000000",
        0.30: "langevin_2023-02-06_04-05-20_n10000000",
        0.35: "langevin_2023-02-06_04-01-45_n10000000",
        0.40: "langevin_2023-02-06_04-02-06_n10000000"
    }
    """

    dir_end1 = (104, 115)
    dir_end2 = (5, 214)
    dir_force_axis=jnp.array([0, 0, 1])

    basedir = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/elastic-mod-sims-2-6/")
    force_to_dir = {
        0.05: "langevin_2023-02-06_22-46-27_n10000000",
        0.10: "langevin_2023-02-06_22-46-50_n10000000",
        0.15: "langevin_2023-02-06_22-47-15_n10000000",
        0.20: "langevin_2023-02-06_22-47-44_n10000000",
        0.25: "langevin_2023-02-06_22-48-34_n10000000",
        0.30: "langevin_2023-02-06_22-48-47_n10000000",
        0.35: "langevin_2023-02-06_22-48-49_n10000000",
        0.40: "langevin_2023-02-06_22-50-07_n10000000"
    }


    def compute_dist(state, end1, end2, force_axis):
        end1_com = (state.center[end1[0]] + state.center[end1[1]]) / 2
        end2_com = (state.center[end2[0]] + state.center[end2[1]]) / 2

        midp_disp = end1_com - end2_com
        projected_dist = jnp.dot(midp_disp, force_axis)
        return jnp.linalg.norm(projected_dist) # Note: incase it's negative



    force_to_pdist = dict()
    force_to_pdists = dict()
    for k, v in tqdm(force_to_dir.items()):
        bpath = basedir / v

        top_path = bpath / "generated.top"
        traj_path = bpath / "output.dat"

        top_info = TopologyInfo(top_path, reverse_direction=True)
        traj_info = TrajectoryInfo(top_info, traj_path=traj_path, reverse_direction=True)

        all_projected_dists = list()
        for s in tqdm(traj_info.states[3:]): # note: can vmap over this. Also, note we ignore the first two for equilibrium
            p_dist = compute_dist(s, end1=dir_end1, end2=dir_end2, force_axis=dir_force_axis)
            all_projected_dists.append(p_dist)
        force_to_pdists[k] = all_projected_dists
        force_to_pdist[k] = jnp.mean(jnp.array(all_projected_dists))

    xs = list(force_to_pdist.keys())
    ys = list(force_to_pdist.values())

    # plt.plot(xs, ys)
    # plt.show()

    pdb.set_trace()

    print("done")
