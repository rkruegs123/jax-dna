import pdb
from tqdm import tqdm
from pathlib import Path

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

Array = util.Array
FLAGS = jax_config.FLAGS
DYNAMICS_STEPS = 100

f32 = util.f32
f64 = util.f64

DTYPE = [f32]
if FLAGS.jax_enable_x64:
    DTYPE += [f64]


def coth(x):
    # return 1 / jnp.tanh(x)
    return (jnp.exp(2*x) + 1) / (jnp.exp(2*x) - 1)

def compute_dist(state, end1, end2, force_axis):
    end1_com = (state.center[end1[0]] + state.center[end1[1]]) / 2
    end2_com = (state.center[end2[0]] + state.center[end2[1]]) / 2

    midp_disp = end1_com - end2_com
    projected_dist = jnp.dot(midp_disp, force_axis)
    return jnp.linalg.norm(projected_dist) # Note: incase it's negative

def calculate_x(force, l0, lps, k, kT):
    y = ((force * l0**2)/(lps*kT))**(1/2)
    x = l0 * (1 + force/k - kT/(2*force*l0) * (1 + y*coth(y)))
    return x

# Used for fitting via non-linear lsq
def WLC(coeffs, x_data, force_data, kT):
    # coeffs = [L0, Lp, K]
    l0 = coeffs[0]
    lps = coeffs[1]
    k = coeffs[2]

    x_calc = calculate_x(force_data, l0, lps, k, kT)
    residual = x_data - x_calc
    # y = ((force_data * coeffs[0]**2)/(coeffs[1]*kT))**(1/2)
    # residual = x_data - coeffs[0] * (1 + force_data/coeffs[2] - kT/(2*force_data*coeffs[0]) * (1 + y*coth(y)))
    return residual



def read_oxdna_long_data():
    import pickle

    # basedir = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/ext-mod-oxdna")
    # basedir = Path("/n/brenner_lab/Lab/JAXDNA/box_200/1e9steps")
    # basedir = Path("/n/brenner_lab/Lab/JAXDNA/box_200/dt_5e-3/samp_rate_5000")
    basedir = Path("/n/brenner_lab/Lab/JAXDNA/box_200/dt_5e-3/jax-dna-samp_rate_1k_2e7_steps")
    # basedir = Path("/n/brenner_lab/User/rkrueger/jaxmd-oxdna/v2/data/output/langevin_2023-02-21_17-45-32_n50000")
    top_path = basedir / "generated.top"
    dir_end2 = (104, 115)
    dir_end1 = (5, 214)
    dir_force_axis = jnp.array([0, 0, 1])

    force_fnames = {
        # 0: "traj_F0_1e9.dat",

        # 0.05: "output.dat"

        # For jax-dna-samp_rate_1k_2e7_steps
        # 0.05: "traj_0.05.dat",
        # 0.055: "traj_0.055.dat",
        # 0.06: "traj_0.06.dat",
        # 0.065: "traj_0.065.dat",
        # 0.07: "traj_0.07.dat",
        # 0.075: "traj_0.075.dat",
        # 0.08: "traj_0.08.dat",
        # 0.09: "traj_0.09.dat",
        # 0.095: "traj_0.095.dat",
        # 0.1: "traj_0.1.dat",
        # 0.15: "traj_0.15.dat",
        # 0.2: "traj_0.2.dat",
        0.3: "traj_0.3.dat",
        0.4: "traj_0.4.dat",
        0.5: "traj_0.5.dat",
        0.6: "traj_0.6.dat",
        0.7: "traj_0.7.dat",
        0.75: "traj_0.75.dat"

        # 0.1: "traj_F0.1.dat",
        # 0.15: "traj_F0.15.dat",
        # 0.2: "traj_F0.2.dat",
        # 0.25: "traj_F0.25.dat",
        # 0.3: "traj_F0.3.dat",
        # 0.35: "traj_F0.35.dat",
        # 0.4: "traj_F0.4.dat",
        # 0.45: "traj_F0.45.dat",
        # 0.5: "traj_F0.5.dat",
        # 0.55: "traj_F0.55.dat",
        # 0.6: "traj_F0.6.dat",
        # 0.65: "traj_F0.65.dat",
        # 0.7: "traj_F0.7.dat",
        # 0.75: "traj_F0.75.dat",
        # 0.8: "traj_F0.8.dat",
        # 0.85: "traj_F0.85.dat"
    }

    force_to_pdists = dict()
    for k, v in tqdm(force_fnames.items()):
        traj_path = basedir / v

        top_info = TopologyInfo(top_path, reverse_direction=True)
        traj_info = TrajectoryInfo(top_info, traj_path=traj_path, reverse_direction=True)

        all_projected_dists = list()
        for s in tqdm(traj_info.states): # note: can vmap over this. Also, note we ignore the first two for equilibrium
            p_dist = compute_dist(s, end1=dir_end1, end2=dir_end2, force_axis=dir_force_axis)
            all_projected_dists.append(p_dist)
        force_to_pdists[k] = all_projected_dists
        pickle.dump(all_projected_dists, open(f"pdists_{k}.pkl", "wb"))
        # pickle.dump(all_projected_dists, open(f"pdists_autocorrelate_{k}.pkl", "wb"))
    
    # pickle.dump(force_to_pdists, open(f"pdists_{k}.pkl", "wb"))
    return force_to_pdists


def analyze_oxdna():
    import pickle

    # basedir = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/ext-mod-oxdna/box97")
    # basedir = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/ext-mod-oxdna/box200")
    # basedir = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/ext-mod-2-21/dt_8e-3")
    basedir = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/jax-dna-samp_rate_1k_2e7_steps")
    # basedir = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/ext-mod-oxdna/box200/samp_rate_5000")

    top_path = basedir / "generated.top"
    dir_end2 = (104, 115)
    dir_end1 = (5, 214)
    dir_force_axis=jnp.array([0, 0, 1])

    """
    forces = [
        "0.1",
        "0.2", "0.3", "0.4",
        "0.5",
        # "0.6",
        "0.75", "0.8"]
    """
    forces = [
        "0.05",
        # "0.055",
        # "0.06",
        # "0.065",
        # "0.07",
        # "0.075",
        # "0.08",
        # "0.09",
        # "0.095",
        "0.1",
        # "0.15",
        "0.2",
        # "0.25",
        "0.3",
        # "0.35",
        "0.4",
        # "0.45",
        "0.5",
        # "0.55",
        "0.6",
        # "0.65",
        # "0.7",
        "0.75",
        # "0.8",
        # "0.85"
    ]


    all_pdists = dict()
    for f in tqdm(forces, desc="Loading"):
        bpath = basedir / f"pdists_{f}.pkl"
        pdists = pickle.load(open(bpath, "rb"))
        # all_pdists[f] = pdists[eval(f)]
        # all_pdists[f] = pdists
        all_pdists[f] = pdists[::40]

    pdb.set_trace()
    start_idx = 100

    kT = get_kt(t=DEFAULT_TEMP)
    force_vals = jnp.array([eval(f) for f in forces])
    all_l0s = list()
    all_lps = list()
    all_ks = list()
    n = len(all_pdists["0.2"])
    all_f_lens = {f: list() for f in forces}
    stride = 50
    for up_to in tqdm(range(start_idx+stride, n, stride)):
        x_init = jnp.array([39.87, 50.60, 44.54]) # initialize to the true values

        lens = list()
        for f in forces:
            f_pdists = all_pdists[f]
            f_len = onp.mean(f_pdists[start_idx:up_to]) # running average
            # f_len = onp.mean(f_pdists[up_to-stride:up_to]) # sliding window
            # if f == "0.05":
                # f_len = 34.96
                # f_len = 34.86
            all_f_lens[f].append(f_len)
            lens.append(f_len)
        lens = jnp.array(lens)

        gn = GaussNewton(residual_fun=WLC)
        gn_sol = gn.run(x_init, x_data=lens, force_data=force_vals, kT=kT).params

        all_l0s.append(gn_sol[0])
        all_lps.append(gn_sol[1])
        all_ks.append(gn_sol[2])

    for f, f_lens in all_f_lens.items():
        plt.plot(f_lens, label=f)
    plt.legend()
    plt.show()

    pdb.set_trace()

    plt.plot(all_l0s, label="l0")
    plt.plot(all_lps, label="lp")
    plt.plot(all_ks, label="k")
    plt.legend()
    plt.show()

    pdb.set_trace()
    return

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from jaxopt import GaussNewton, LevenbergMarquardt
    from utils import get_kt, DEFAULT_TEMP

    analyze_oxdna()
    pdb.set_trace()

    # read_oxdna_long_data()

    # pdb.set_trace()

    kT = get_kt(t=DEFAULT_TEMP)

    forces = jnp.array([
        0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
        # 0.35,
        0.375,
        # 0.4
    ]) * 2
    # lens = jnp.array([33.47837202, 34.63343703, 35.13880103, 35.48323747,
    #                   35.73831831, 35.95160141, 36.12325301, 36.27638711])
    lens = jnp.array([35.25930400991341, 36.71562944132667,  37.79990723786074,
                      38.34555511086673,  38.68266174743726,
                      38.941749288990714,  39.1546694320399,
                      # 39.33738895751,
                      39.40541037118505,
                      # 39.492305574343014
    ])

    # forces = jnp.array([0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.375]) * 2
    # lens = jnp.array([35.2236991090926, 36.63971120822037, 37.76098514351735, 38.28013817558361, 38.685128174964454, 38.93867142307843, 39.154639682794254, 39.42241780082958])

    # plt.plot(forces, lens)
    # plt.show()


    pdb.set_trace()


    # plot the WLC

    """
    kT_si = 4.088 # in pN*nm
    forces = onp.linspace(2.5, 40, 100) # in pN
    computed_extensions = [calculate_x(force, 339.6 / 10, 431.0 / 10, 2166, kT_si) for force in forces] # in nm
    plt.plot(computed_extensions, forces)
    plt.xlabel("Extension (nm)")
    plt.ylabel("Force (pN)")
    plt.show()

    pdb.set_trace()
    """



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
    # gn = LevenbergMarquardt(residual_fun=WLC)
    gn_sol = gn.run(x_init, x_data=lens, force_data=forces, kT=kT).params

    x_init_si = jnp.array([339.6 / 10.0, 431.0 / 10.0, 2166.0])
    forces_si = forces * 48.63 # pN
    lens_si = lens * 0.8518 # nm
    kT_si = 4.08846006711 # in pN*nm
    gn_si = GaussNewton(residual_fun=WLC)
    gn_sol_si = gn_si.run(x_init_si, x_data=lens_si, force_data=forces_si, kT=kT_si).params


    pdb.set_trace()


    # Check the fit

    test_forces = onp.linspace(0.05, 0.8, 10) # in simulation units
    computed_extensions = [calculate_x(force, gn_sol[0], gn_sol[1], gn_sol[2], kT) for force in test_forces] # in nm
    plt.plot(computed_extensions, test_forces, label="fit")
    plt.plot(lens, forces, label="ours")
    plt.xlabel("Extension")
    plt.ylabel("Force")
    plt.legend()
    plt.show()

    test_forces = onp.linspace(2.5, 40, 100) # in pN
    computed_extensions = [calculate_x(force, gn_sol_si[0], gn_sol_si[1], gn_sol_si[2], kT_si) for force in test_forces] # in nm
    tom_extensions = [calculate_x(force, x_init_si[0], x_init_si[1], x_init_si[2], kT_si) for force in test_forces] # in nm
    plt.plot(computed_extensions, test_forces, label="fit")
    plt.plot(tom_extensions, test_forces, label="tom fit")
    plt.scatter(lens_si, forces_si, label="ours")
    plt.xlabel("Extension (nm)")
    plt.ylabel("Force (pN)")
    plt.legend()
    plt.show()



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

    """
    dir_end1 = (104, 115)
    dir_end2 = (5, 214)
    dir_force_axis=jnp.array([0, 0, 1])

    basedir = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/elastic-mod-sims-2-6/")
    force_to_dir = {
        0.025: "langevin_2023-02-08_12-47-49_n5000000", # FIXME: only 5e6 steps
        0.05: "langevin_2023-02-06_22-46-27_n10000000",
        0.10: "langevin_2023-02-06_22-46-50_n10000000",
        0.15: "langevin_2023-02-06_22-47-15_n10000000",
        0.20: "langevin_2023-02-06_22-47-44_n10000000",
        0.25: "langevin_2023-02-06_22-48-34_n10000000",
        0.30: "langevin_2023-02-06_22-48-47_n10000000",
        0.35: "langevin_2023-02-06_22-48-49_n10000000",
        0.375: "langevin_2023-02-08_12-46-35_n5000000", # FIXME: only 5e6 steps
        0.40: "langevin_2023-02-06_22-50-07_n10000000"
    }
    """

    dir_end2 = (104, 115)
    dir_end1 = (5, 214)
    dir_force_axis=jnp.array([0, 0, 1])

    basedir = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/ext-mod-sims-2-10/")
    force_to_dir = {
        0.025: "langevin_2023-02-10_00-58-38_n3000000",
        0.05: "langevin_2023-02-10_00-58-52_n3000000",
        0.10: "langevin_2023-02-10_00-59-28_n3000000",
        0.15: "langevin_2023-02-10_00-59-22_n3000000",
        0.20: "langevin_2023-02-10_00-59-02_n3000000",
        0.25: "langevin_2023-02-10_00-59-32_n3000000",
        0.30: "langevin_2023-02-10_00-59-57_n3000000",
        0.375: "langevin_2023-02-10_01-00-05_n3000000"
    }




    force_to_pdist = dict()
    force_to_pdists = dict()
    mean_start_idx = 20
    for k, v in tqdm(force_to_dir.items()):
        bpath = basedir / v

        top_path = bpath / "generated.top"
        traj_path = bpath / "output.dat"

        top_info = TopologyInfo(top_path, reverse_direction=True)
        traj_info = TrajectoryInfo(top_info, traj_path=traj_path, reverse_direction=True)

        all_projected_dists = list()
        for s in tqdm(traj_info.states): # note: can vmap over this. Also, note we ignore the first two for equilibrium
            p_dist = compute_dist(s, end1=dir_end1, end2=dir_end2, force_axis=dir_force_axis)
            all_projected_dists.append(p_dist)
        force_to_pdists[k] = all_projected_dists
        force_to_pdist[k] = jnp.mean(jnp.array(all_projected_dists[mean_start_idx:]))

    xs = list(force_to_pdist.keys())
    ys = list(force_to_pdist.values())

    # plt.plot(xs, ys)
    # plt.show()

    pdb.set_trace()

    # Sample code
    # start_idx = 25
    # means = [onp.mean(force_to_pdists[0.025][start_idx:end_idx]) * 0.8518 for end_idx in jnp.arange(len(force_to_pdists[0.025]))]

    print("done")
