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

from jax.config import config; 
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

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


f64 = util.f64

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
            cv1_method, cv2_method,
            show_fig=True, save_fig=False, fpath=None):

    if show_fig and save_fig:
        raise RuntimeError(f"Can only save or show fig.")

    n_cv1_samples = 100
    if cv1_method in ["nbp-sigmoid", "nbp-review"]:
        sample_cv1s = onp.linspace(-1, 8, n_cv1_samples)
        ylabel = "# Base Pairs"
    elif cv1_method == "theta":
      sample_cv1s = onp.linspace(0, 3.14, n_cv1_samples)
      ylabel = "Theta"
    elif cv1_method == "ratio-contacts":
        sample_cv1s = onp.linspace(0, 1, n_cv1_samples)
        ylabel = "Ratio of Native Contacts"
    else:
        raise RuntimeError(f"Invalid CV1 method: {cv1_method}")

    # sample_distances = onp.linspace(0, 3, 30)
    sample_distances = onp.linspace(0, d_critical - 2, 200)
    b, a = onp.meshgrid(sample_cv1s, sample_distances)
    vals = onp.empty((b.shape))
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            vals[i, j] = repulsive_wall_fn(heights, centers, widths, b[i, j], a[i, j])
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
    plt.ylabel(ylabel)
    if show_fig:
        plt.show()
    if save_fig:
        if not fpath:
            raise RuntimeError(f"Must provide fname to save 2D plot")
        plt.savefig(fpath)
    plt.clf()



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
    cv2_method = args['cv2_method']

    d_critical = args['d_critical']
    wall_strength = args['wall_strength']
    box_type = args['box_type']

    save_every = args['save_every']
    save_output = args['save_output']
    plot_every = args['plot_every']
    if plot_every < 0:
        plot_every = None

    well_tempered = args['well_tempered']
    delta_T = args['delta_T']

    smooth_min_k = args['smooth_min_k']

    if not Path(top_path).exists():
        raise RuntimeError(f"Topology file does not exist at location: {top_path}")

    if not Path(conf_path).exists():
        raise RuntimeError(f"Configuration file does not exist at location: {conf_path}")

    if save_output:
        output_basedir = Path(output_basedir)
        if not output_basedir.exists():
            raise RuntimeError(f"Output base directory does not exist at location: {output_basedir}")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f"metad_{timestamp}_{cv1_method}_{cv2_method}_{box_type}"
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



    init_fene_params = [2.0, 0.25, 0.7525]
    init_stacking_params = [
        1.3448, 2.6568, 6.0, 0.4, 0.9, 0.32, 0.75, # f1(dr_stack)
        1.30, 0.0, 0.8, # f4(theta_4)
        0.90, 0.0, 0.95, # f4(theta_5p)
        0.90, 0.0, 0.95, # f4(theta_6p)
        2.0, -0.65, # f5(-cos(phi1))
        2.0, -0.65 # f5(-cos(phi2))
    ]
    params = init_fene_params + init_stacking_params

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)

    if box_type == "periodic":
        displacement_fn, shift_fn = space.periodic(config_info.box_size)
    elif box_type == "free":
        displacement_fn, shift_fn = space.free()
    else:
        raise RuntimeError(f"Invalid box type: {box_type}")

    body = config_info.states[0]
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT

    gamma = RigidBody(center=jnp.array([kT/2.5]),
                      orientation=jnp.array([kT/7.5]))


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
    height_fn = jit(height_fn)

    if cv1_method in ['nbp-review', 'nbp-sigmoid']:
        cv1_fn = cv.get_n_bp_fn_custom(cv1_bps, displacement_fn, cv1_method)
    elif cv1_method == "theta":
        str0_3p_idx = cv1_bps[:, 0][0] # may have mixed up 3p and 5p but it doesn't matter
        str0_5p_idx = cv1_bps[:, 0][-1]
        str1_3p_idx = cv1_bps[:, 1][-1]
        str1_5p_idx = cv1_bps[:, 1][0]
        cv1_fn = cv.get_theta_fn(str0_3p_idx, str0_5p_idx, str1_3p_idx, str1_5p_idx)
    elif cv1_method == "ratio-contacts":
        ref_top_path = args['ref_top_path']
        ref_conf_path = args['ref_conf_path']
        q_lambda = args['q_lambda']
        q_gamma = args['q_gamma']
        q_threshold = args['q_threshold']

        ref_top_info = TopologyInfo(ref_top_path, reverse_direction=True)
        ref_config_info = TrajectoryInfo(ref_top_info, traj_path=ref_conf_path, reverse_direction=True)
        reference_body = ref_config_info.states[0]

        cv1_fn = cv.get_q_fn(reference_body, cv1_bps, displacement_fn,
                             q_lambda, q_gamma, q_threshold)
    else:
        raise RuntimeError(f"Invalid CV1 method: {cv1_method}")
    cv1_fn = jit(cv1_fn)



    if cv2_method == "com-dist":
        cv2_fn = cv.get_interstrand_dist_fn(cv2_bps, displacement_fn)
    elif cv2_method == "min-dist":
        cv2_fn = cv.get_min_dist_fn(cv2_bps, displacement_fn, k=smooth_min_k)
    else:
        raise RuntimeError(f"Invalid CV2 method: {cv2_method}")
    cv2_fn = jit(cv2_fn)

    repulsive_wall_fn = md_utils.get_repulsive_wall_fn(d_critical, wall_strength)
    repulsive_wall_fn = jit(repulsive_wall_fn)


    # Wrap the energy function
    base_energy_fn, compute_subterms = factory.energy_fn_factory(
        displacement_fn,
        back_site, stack_site, base_site,
        top_info.bonded_nbrs, top_info.unbonded_nbrs)
    compute_subterms = jit(Partial(compute_subterms, seq=seq, params=params))
    base_energy_fn = Partial(base_energy_fn, seq=seq, params=params)
    base_energy_fn = jit(base_energy_fn)

    # md_energy_fn = md_energy.factory(base_energy_fn, cv1_fn)
    md_energy_fn = md_energy.factory_2d(base_energy_fn,
                                        cv1_fn, cv2_fn,
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
            # iter_cv = cv1_fn(state.position)
            # iter_bias = md_utils.sum_of_gaussians(heights, centers, widths, iter_cv)
            # num_gauss = i // stride
            # # widths = widths.at[num_guass].set(width_0)
            # heights = heights.at[num_gauss].set(height_fn(iter_bias))
            # centers = centers.at[num_gauss].set(iter_cv)

            iter_cv1 = cv1_fn(state.position)
            iter_cv2 = cv2_fn(state.position)
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
                    cv1_method, cv2_method,
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

        plot_2d(repulsive_wall_fn, heights, centers, widths, d_critical,
                cv1_method, cv2_method,
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
    parser.add_argument('-s', '--stride', dest="stride", type=int, default=500, help="Stride for MetaD")
    parser.add_argument('--n-steps', type=int, default=1000000, help="Num. of steps for MetaD")
    parser.add_argument('--init-height', type=float, default=0.25, help="Height of gaussian deposited") # note: de Pablo group used 0.05
    parser.add_argument('--width-cv1', type=float, default=0.20, help="Width of Gaussian @ # of bps") # note: de Pablo used `width_cv1 = 0.035` for theta
    parser.add_argument('--width-cv2', type=float, default=0.05, help="Width of Gaussian @ interstrand distance")
    parser.add_argument('--d-critical', type=float, default=15.0, help="Where to put the wall for interstrand distance")
    parser.add_argument('--wall-strength', type=float, default=1000.0, help="Strength of the wall for interstrand distance")
    parser.add_argument('--dt', type=float, default=5e-3, help="Time step for integration")
    parser.add_argument('--temp', type=float, default=DEFAULT_TEMP, help="Temperature for simulation (K)")

    parser.add_argument('-t', '--top-path', type=str,
                        default="data/simple-helix/generated.top",
                        help='Path to topology file')
    parser.add_argument('-c', '--conf-path', type=str,
                        default="data/simple-helix/start.conf",
                        # default="data/simple-helix/unbound.conf", # unbound
                        help='Path to input configuration')

    parser.add_argument('--save-every', type=int, default=10000, help="Interval for saving trajectory")
    parser.add_argument('--plot-every', type=int, default=-1, help="Interval for plotting. -1 means no plotting.")
    parser.add_argument("--save-output", default=True, action="store_false",
                        help="Whether or not to save output")

    parser.add_argument("--well-tempered", default=False, action="store_true",
                        help="Whether or not to use well-tempered MetaD")
    parser.add_argument('--delta-T', type=float, default=20.0,
                        help="Hyperparameter for well-tempered MetaD")
    parser.add_argument('--smooth-min-k', type=float, default=100.0,
                        help="K-value for our smooth minimum function")

    parser.add_argument('--cv1-method', type=str,
                        default="nbp-sigmoid",
                        choices=["nbp-sigmoid", "nbp-review",
                                 "theta", "ratio-contacts"],
                        help='The first collective variable')
    parser.add_argument('--cv2-method', type=str,
                        default="com-dist",
                        choices=["com-dist", "min-dist"],
                        help="The second collective variable (a distance)")

    parser.add_argument('--box-type', type=str,
                        default="periodic",
                        choices=["periodic", "free"],
                        help="The type of boxed used for simulation")

    # Arguments for "ratio-contacts" option for cv1
    parser.add_argument('--ref-top-path', type=str,
                        default="data/simple-helix/generated.top",
                        help='Path to *reference* topology file')
    parser.add_argument('--ref-conf-path', type=str,
                        default="data/simple-helix/start.conf",
                        help='Path to *reference* input configuration')
    parser.add_argument('--q-lambda', type=float, default=1.5,
                        help="Lambda value for computing the ratio of native contacts")
    parser.add_argument('--q-gamma', type=float, default=60,
                        help="Gamma value for computing the ratio of native contacts")
    parser.add_argument('--q-threshold', type=float, default=0.45,
                        help="Distance threshold to determine which pairs to include when computing the ratio of native contacts")

    return parser

if __name__ == "__main__":
    parser = build_argparse()
    args = vars(parser.parse_args())

    key = random.PRNGKey(0)

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
        # [1, 14],
        [2, 13],
        [3, 12],
        [4, 11],
        [5, 10],
        # [6, 9],
        # [7, 8]
    ])


    run_single_metad(args, cv1_bps, cv2_bps, key)


    """
    # bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/metad_2023-01-11_01-02-24")
    # bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/metad_2023-01-11_23-40-40")
    # bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/metad_2023-01-16_17-11-40")
    # bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/metad_2023-01-16_17-11-30")
    bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/metad_2023-01-19_13-04-07")


    d_critical = 10.0
    width_cv1 = 0.018
    width_cv2 = 0.013
    height0 = 0.17

    centers = pickle.load(open(bpath / "centers.pkl", "rb"))
    widths = pickle.load(open(bpath / "widths.pkl", "rb"))
    heights = pickle.load(open(bpath / "heights.pkl", "rb"))

    pdb.set_trace()

    plt.plot(list(range(len(centers))), centers[:, 0])
    plt.title("Centers")
    plt.xlabel("# Gaussian")
    plt.ylabel("Center (i.e. CV)")
    # plt.savefig(bpath / "centers.png")
    # plt.clf()
    plt.show()
    plt.clf()

    pdb.set_trace()


    repulsive_wall_fn = md_utils.get_repulsive_wall_fn(d_critical=d_critical, wall_strength=1000.0)
    repulsive_wall_fn = jit(repulsive_wall_fn)
    """



    """
    widths = jnp.full((centers.shape[0], 2), jnp.array([width_cv1, width_cv2]), dtype=f64)
    heights = jnp.full(centers.shape[0], height0, dtype=f64)

    with open(bpath / "widths.pkl", "wb") as wf:
        pickle.dump(widths, wf)
    with open(bpath / "heights.pkl", "wb") as hf:
        pickle.dump(heights, hf)
    """

    """
    plot_2d(repulsive_wall_fn, heights, centers, widths, d_critical,
            cv1_method="ratio-contacts", cv2_method="com-dist",
            show_fig=True, save_fig=False, fpath=None)

    pdb.set_trace()
    """
