from tqdm import tqdm
import pdb
from pathlib import Path
import shutil
import datetime
import matplotlib.pyplot as plt
import numpy as onp
import pickle
import time
import argparse

from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax import jit
from jax.tree_util import Partial
import jax

from jax_md.rigid_body import RigidBody
from jax_md import space, util, simulate


from energy import factory # FIXME: will want to replace with a different energy function
# import langevin
from loader import get_params
from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo
from metadynamics import cv
import metadynamics.utils as md_utils
import metadynamics.energy as md_energy
from utils import bcolors
from utils import nucleotide_mass, get_kt, moment_of_inertia, get_one_hot, DEFAULT_TEMP
from utils import base_site, stack_site, back_site




def plot_1d(heights, centers, widths,
            show_fig=True, save_fig=False, fpath=None):
    test_cvs = onp.linspace(-2, 10, 200)
    # test_cvs = onp.linspace(0, 10, 40)
    biases = [md_utils.sum_of_gaussians(heights, centers, widths, tmp_cv) for tmp_cv in test_cvs]
    plt.plot(test_cvs, biases)
    if show_fig:
        plt.show()
    if save_fig:
        if not fpath:
            raise RuntimeError(f"Must provide fname to save 1D plot")
        plt.savefig(fpath)
    plt.clf()


def plot_2d(repulsive_wall_fn, heights, centers, widths, d_critical,
            show_fig=True, save_fig=False, fpath=None):
    sample_n_bps = onp.linspace(-1, 8, 100)
    # sample_thetas = onp.linspace(0, 3.14, 100)
    # sample_distances = onp.linspace(0, 3, 30)
    sample_distances = onp.linspace(0, d_critical - 2, 200)
    b, a = onp.meshgrid(sample_n_bps, sample_distances)
    # b, a = onp.meshgrid(sample_thetas, sample_distances)
    vals = onp.empty((b.shape))
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            vals[i, j] = repulsive_wall_fn(heights, centers, widths, b[i, j], a[i, j]) # FIXME: maybe swap a and b?
    l_a = a.min()
    r_a = a.max()
    l_b = b.min()
    r_b = b.max()
    # l_val, r_val = -onp.abs(vals).max(), onp.abs(vals).max()
    l_val, r_val = onp.abs(vals).min(), onp.abs(vals).max()

    figure, axes = plt.subplots()
    c = axes.pcolormesh(a, b, vals, cmap='cool', vmin=l_val, vmax=r_val)
    axes.axis([l_a, r_a, l_b, r_b])
    figure.colorbar(c)
    plt.xlabel("Interstrand Distance")
    # plt.ylabel("# Base Pairs")
    plt.ylabel("Theta")
    if show_fig:
        plt.show()
    if save_fig:
        if not fpath:
            raise RuntimeError(f"Must provide fname to save 2D plot")
        plt.savefig(fpath)
    plt.clf()



f64 = util.f64

def run_single_metad(args, cv1_bps, cv2_bps, key,
                     output_basedir="v2/data/output/"):

    top_path = args['top_path']
    conf_path = args['conf_path']

    n_steps = args['n_steps']
    stride = args['stride']
    n_gaussians = n_steps // stride

    T = args['temp']
    dt = args['dt']

    height_0 = args['init_height']
    width_cv1 = args['width_cv1']
    width_cv2 = args['width_cv2']
    cv1_method = args['cv1_method']

    d_critical = args['d_critical']
    wall_strength = args['wall_strength']

    save_every = args['save_every']
    save_output = args['save_output']
    plot_every = args['plot_every']
    if plot_every < 0:
        plot_every = None

    well_tempered = args['well_tempered']
    delta_T = args['delta_T']

    if not Path(top_path).exists():
        raise RuntimeError(f"Topology file does not exist at location: {top_path}")

    if not Path(conf_path).exists():
        raise RuntimeError(f"Configuration file does not exist at location: {conf_path}")

    if save_output:
        output_basedir = Path(output_basedir)
        if not output_basedir.exists():
            raise RuntimeError(f"Output base directory does not exist at location: {output_basedir}")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f"metad_{timestamp}"
        run_dir = output_basedir / run_name
        run_dir.mkdir(parents=False, exist_ok=False)
        shutil.copy(top_path, run_dir)
        shutil.copy(conf_path, run_dir)

        params_str = ""
        for k, v in args.items():
            params_str += f"{k}: {v}\n"

        with open(run_dir / "params.txt", "w+") as f:
            f.write(params_str)
        print(bcolors.WARNING + f"Created directory and copied simulation information at location: {run_dir}" + bcolors.ENDC)


    print(bcolors.OKBLUE + f"Setting up simulation..." + bcolors.ENDC)
    # Typical information
    mass = RigidBody(center=jnp.array([nucleotide_mass]),
                     orientation=jnp.array([moment_of_inertia]))

    gamma = RigidBody(center=jnp.array([DEFAULT_TEMP/2.5]),
                      orientation=jnp.array([DEFAULT_TEMP/7.5]))

    params = [2.0, 0.25, 0.7525]

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    displacement_fn, shift_fn = space.periodic(config_info.box_size)

    body = config_info.states[0]
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT


    # Metapotential information

    heights = jnp.zeros(n_gaussians, dtype=f64)

    centers = jnp.zeros((n_gaussians, 2), dtype=f64)
    # widths = jnp.full((n_gaussians, 2), width_0, dtype=f64)
    widths = jnp.full((n_gaussians, 2), jnp.array([width_cv1, width_cv2]), dtype=f64)

    """
    # 1D case
    centers = jnp.zeros(n_gaussians, dtype=f64)
    widths = jnp.full(n_gaussians, width_cv1, dtype=f64)
    """


    height_fn = md_utils.get_height_fn(height_0, well_tempered=well_tempered,
                                       kt=kT, delta_T=delta_T)
    # height_fn = md_utils.get_height_fn(height_0, well_tempered=False)
    # height_fn = md_utils.get_height_fn(height_0, well_tempered=True, kt=kT, delta_T=20.0)
    # n_bp_fn = cv.get_n_bp_fn_original(cv1_bps, displacement_fn) # cv1
    n_bp_fn = cv.get_n_bp_fn_custom(cv1_bps, displacement_fn, cv1_method)

    str0_3p_idx = cv1_bps[:, 0][0] # may have mixed up 3p and 5p but it doesn't matter
    str0_5p_idx = cv1_bps[:, 0][-1]
    str1_3p_idx = cv1_bps[:, 1][-1]
    str1_5p_idx = cv1_bps[:, 1][0]
    theta_fn = cv.get_theta_fn(str0_3p_idx, str0_5p_idx, str1_3p_idx, str1_5p_idx)
    theta_fn = jit(theta_fn)

    # interstrand_dist_fn = cv.get_interstrand_dist_fn(cv2_bps, displacement_fn)
    # interstrand_dist_fn = jit(interstrand_dist_fn)
    interstrand_dist_fn = cv.get_min_dist_fn(cv2_bps, displacement_fn)
    interstrand_dist_fn = jit(interstrand_dist_fn)
    repulsive_wall_fn = md_utils.get_repulsive_wall_fn(d_critical, wall_strength)
    repulsive_wall_fn = jit(repulsive_wall_fn)

    height_fn = jit(height_fn)
    n_bp_fn = jit(n_bp_fn)


    # Wrap the energy function
    base_energy_fn, compute_subterms = factory.energy_fn_factory(
        displacement_fn,
        back_site, stack_site, base_site,
        top_info.bonded_nbrs, top_info.unbonded_nbrs)
    compute_subterms = jit(Partial(compute_subterms, seq=seq, params=params))
    base_energy_fn = Partial(base_energy_fn, seq=seq, params=params)
    # energy_fn = jit(Partial(energy_fn, seq=seq, params=params))

    # md_energy_fn = md_energy.factory(base_energy_fn, n_bp_fn)
    md_energy_fn = md_energy.factory_2d(base_energy_fn,
                                        n_bp_fn, interstrand_dist_fn,
                                        # theta_fn, interstrand_dist_fn,
                                        repulsive_wall_fn)
    md_energy_fn = jit(md_energy_fn)

    # init_fn, step_fn = langevin.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
    init_fn, step_fn = simulate.nvt_langevin(md_energy_fn, shift_fn, dt, kT, gamma)
    step_fn = jit(step_fn)

    # state = init_fn(key, body, mass=mass, seq=seq, params=params)
    state = init_fn(key, body, mass=mass,
                    heights=heights, centers=centers, widths=widths)


    trajectory = [state.position]
    energies = [md_energy_fn(state.position,
                             heights=heights, centers=centers, widths=widths)]
    subterms = [compute_subterms(state.position)]
    print(bcolors.OKBLUE + f"Starting simulation..." + bcolors.ENDC)


    start = time.time()
    for i in tqdm(range(n_steps), colour="blue"):
        state = step_fn(state, heights=heights, centers=centers, widths=widths)

        if i % stride == 0:

            # 1D case
            # iter_cv = n_bp_fn(state.position)
            # iter_bias = md_utils.sum_of_gaussians(heights, centers, widths, iter_cv)
            # num_gauss = i // stride
            # # widths = widths.at[num_guass].set(width_0)
            # heights = heights.at[num_gauss].set(height_fn(iter_bias))
            # centers = centers.at[num_gauss].set(iter_cv)

            iter_cv1 = n_bp_fn(state.position)
            # iter_cv1 = theta_fn(state.position)
            iter_cv2 = interstrand_dist_fn(state.position)
            iter_bias = repulsive_wall_fn(heights, centers, widths, iter_cv1, iter_cv2)
            num_gauss = i // stride
            heights = heights.at[num_gauss].set(height_fn(iter_bias))
            centers = centers.at[num_gauss, 0].set(iter_cv1)
            centers = centers.at[num_gauss, 1].set(iter_cv2)

        if i % save_every == 0:
            energies.append(md_energy_fn(state.position,
                                         heights=heights, centers=centers, widths=widths))
            trajectory.append(state.position)
            subterms.append(compute_subterms(state.position))

        if plot_every and i % plot_every == 0:
            pdb.set_trace()
            # Plot the metapotential
            # plot_1d(heights, centers, widths)

            plot_2d(repulsive_wall_fn, heights, centers, widths, d_critical,
                    show_fig=True, save_fig=False, fpath=None)

    end = time.time()
    sim_time = onp.round(end - start, 2)
    print(f"Simulation took: {sim_time} seconds")

    start = time.time()
    final_traj = TrajectoryInfo(top_info, states=trajectory, box_size=config_info.box_size)
    if save_output:
        print(bcolors.OKBLUE + f"Writing trajectory to file..." + bcolors.ENDC)
        final_traj.write(run_dir / "output.dat", reverse=True, write_topology=False)

        with open(run_dir / "centers.pkl", "wb") as cf:
            pickle.dump(centers, cf)
        with open(run_dir / "widths.pkl", "wb") as wf:
            pickle.dump(widths, wf)
        with open(run_dir / "heights.pkl", "wb") as hf:
            pickle.dump(heights, hf)

        plt.plot(list(range(len(centers))), centers)
        plt.title("Centers")
        plt.xlabel("# Gaussian")
        plt.ylabel("Center (i.e. CV)")
        plt.savefig(run_dir / "centers.png")
        plt.clf()

        plot_2d(repulsive_wall_fn, heights, centers, widths,
                show_fig=False, save_fig=True, fpath=run_dir / "heatmap.png")
        # plot_1d(heights, centers, widths,
        #         show_fig=False, save_fig=True, fpath=run_dir / "metapotential.png")

    end = time.time()
    analysis_time = onp.round(end - start, 2)
    print(f"Analysis took: {analysis_time} seconds")

    times_str = f"Simulation: {sim_time} seconds\nAnalysis: {analysis_time} seconds\n"
    with open(run_dir / "times.txt", "w+") as f:
        f.write(times_str)

    return final_traj, energies


def build_argparse():
    parser = argparse.ArgumentParser(description="Metadynamics simulation for an 8bp helix")
    parser.add_argument('--s', dest="stride", type=int, default=500, help="Stride for MetaD")
    parser.add_argument('--n-steps', type=int, default=1000000, help="Num. of steps for MetaD")
    parser.add_argument('--init-height', type=float, default=0.25, help="Height of gaussian deposited")
    parser.add_argument('--width-cv1', type=float, default=0.20, help="Width of Gaussian @ # of bps")
    parser.add_argument('--width-cv2', type=float, default=0.05, help="Width of Gaussian @ interstrand distance")
    parser.add_argument('--d-critical', type=float, default=15.0, help="Where to put the wall for interstrand distance")
    parser.add_argument('--wall-strength', type=float, default=1000.0, help="Strength of the wall for interstrand distance")
    parser.add_argument('--dt', type=float, default=5e-3, help="Time step for integration")
    parser.add_argument('--temp', type=float, default=DEFAULT_TEMP, help="Temperature for simulation (K)")

    parser.add_argument('-t', '--top-path', type=str,
                        default="data/simple-helix/generated.top",
                        help='Path to topology file')
    parser.add_argument('-c', '--conf-path', type=str,
                        default="data/simple-helix/start.conf", # could also be unbound.conf
                        help='Path to input configuration')

    parser.add_argument('--save-every', type=int, default=10000, help="Interval for saving trajectory")
    parser.add_argument('--plot-every', type=int, default=-1, help="Interval for plotting. -1 means no plotting.")
    parser.add_argument("--save-output", default=True, action="store_false",
                        help="Whether or not to save output")

    parser.add_argument("--well-tempered", default=False, action="store_true",
                        help="Whether or not to use well-tempered MetaD")
    parser.add_argument('--delta-T', type=float, default=20.0,
                        help="Hyperparameter for well-tempered MetaD")

    parser.add_argument('--cv1-method', type=str,
                        default="sigmoid", # could also be unbound.conf
                        choices=["sigmoid", "review"],
                        help='The method for computing the distance between base sites')

    return parser

if __name__ == "__main__":
    parser = build_argparse()
    args = vars(parser.parse_args())

    # top_path = "data/simple-helix/generated.top"
    # conf_path = "data/simple-helix/start.conf"
    # conf_path = "data/simple-helix/unbound.conf"
    key = random.PRNGKey(0)

    # n_steps = int(1e3)
    # stride = 100
    # stride = 250
    # n_gaussians = n_steps // stride

    cv1_bps = jnp.array([
        [0, 15],
        [1, 14],
        [2, 13],
        [3, 12],
        [4, 11],
        [5, 10],
        [6, 9],
        [7, 8]
    ])

    cv2_bps = jnp.array([
        # [0, 15],
        [1, 14],
        [2, 13],
        [3, 12],
        [4, 11],
        [5, 10],
        [6, 9],
        # [7, 8]
    ])

    # height_0 = 1.0
    # height_0 = 0.05 # from chicago group
    # height_0 = 0.25
    # width_0 = 0.25
    # width_cv1 = 0.035 # radians
    # width_cv1 = 0.2
    # width_cv2 = 0.03 # interstrand distance
    # width_cv2 = 0.05 # interstrand distance

    # d_critical = 15.0
    # wall_strength = 1000

    # run_single_metad(args, cv1_bps, cv2_bps, key)


    bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/metad_2023-01-10_02-40-11")

    centers = pickle.load(open(bpath / "centers.pkl", "rb"))
    heights = jnp.full(centers.shape[0], 0.05, dtype=f64)
    # heights = pickle.load(open(bpath / "heights.pkl", "rb"))
    # widths = pickle.load(open(bpath / "widths.pkl", "rb"))
    widths = jnp.full((centers.shape[0], 2), jnp.array([0.035, 0.05]), dtype=f64)


    pdb.set_trace()


    repulsive_wall_fn = md_utils.get_repulsive_wall_fn(d_critical=15.0, wall_strength=1000.0)
    repulsive_wall_fn = jit(repulsive_wall_fn)

    plot_2d(repulsive_wall_fn, heights, centers, widths)
