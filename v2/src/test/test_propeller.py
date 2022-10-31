import matplotlib.pyplot as plt
from math import radians, degrees
import numpy as np
import pdb

import jax.numpy as jnp

from loss.propeller import get_avg_propeller_twist
from utils import bcolors
from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo
from loader.get_params import get_default_params

TARGET_TWIST = 21.7 # degrees
tol = 3 # degrees

def run():
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    top_path = "v2/data/test-data/simple-helix-296.15-K/generated.top"
    traj_path = "v2/data/test-data/simple-helix-296.15-K/output.dat"

    print(f"----Checking propeller twist for trajectory at location: {traj_path}----")

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=traj_path, reverse_direction=True)

    # body = config_info.states[-1]
    all_twists = list()
    for body in tqdm(config_info.states[-40:]):
        # base_pairs = jnp.array([[0, 15], [1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9], [7, 8]])
        base_pairs = jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
        twist_deg = 180 - degrees(get_avg_propeller_twist(body, base_pairs))
        all_twists.append(twist_deg)
    mean_twist = np.mean(all_twists)

    print(f"Target twist: {TARGET_TWIST}")
    print(f"Mean twist: {mean_twist}")
    print(f"Tolerance: {tol}")

    if not np.abs(mean_twist - TARGET_TWIST) < tol:
        pdb.set_trace()
        print(bcolors.FAIL + "Fail!\n" + bcolors.ENDC)
    print(bcolors.OKGREEN + "Success!\n" + bcolors.ENDC)
    # plt.hist(all_twists)
    # plt.show()
    return
