import pdb
from pathlib import Path
from tqdm import tqdm
import time
import numpy as onp
from copy import deepcopy
import functools
import matplotlib.pyplot as plt
import shutil
import argparse
from jaxopt import GaussNewton

import jax
import jax.numpy as jnp
from jax import jit, vmap, random, lax, tree_util, pmap
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, ext_force
from jax_dna.dna1 import model

from jax.config import config
config.update("jax_enable_x64", True)



displacement_fn, shift_fn = space.free()
dt = 5e-3
t_kelvin = utils.DEFAULT_TEMP
kT = utils.get_kt(t_kelvin)
gamma = rigid_body.RigidBody(
    center=jnp.array([kT/2.5], dtype=jnp.float64),
    orientation=jnp.array([kT/7.5], dtype=jnp.float64))
mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                            orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))
params = deepcopy(model.EMPTY_BASE_PARAMS)
em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)

ext_force_bps1 = [5, 214]
ext_force_bps2 = [104, 115]
dir_force_axis = jnp.array([0, 0, 1])

x_init = jnp.array([39.87, 50.60, 44.54]) # initialize to the true values
x_init_si = jnp.array([x_init[0] * utils.nm_per_oxdna_length,
                       x_init[1] * utils.nm_per_oxdna_length,
                       x_init[2] * utils.oxdna_force_to_pn])

# Load the system
sys_basedir = Path("data/sys-defs/wlc-fit")
top_path = sys_basedir / "generated.top"
top_info = topology.TopologyInfo(top_path, reverse_direction=True)

conf_path = sys_basedir / "generated.dat"
conf_info = trajectory.TrajectoryInfo(
    top_info,
    read_from_file=True, traj_path=conf_path, reverse_direction=True
)
init_body = conf_info.get_states()[0]
seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)


def compute_dist(state):
    end1_com = (state.center[ext_force_bps1[0]] + state.center[ext_force_bps1[1]]) / 2
    end2_com = (state.center[ext_force_bps2[0]] + state.center[ext_force_bps2[1]]) / 2

    midp_disp = end1_com - end2_com
    projected_dist = jnp.dot(midp_disp, dir_force_axis)
    return jnp.linalg.norm(projected_dist) # Note: incase it's negative

def coth(x):
    # return 1 / jnp.tanh(x)
    return (jnp.exp(2*x) + 1) / (jnp.exp(2*x) - 1)

def calculate_x(force, l0, lps, k, kT):
    y = ((force * l0**2)/(lps*kT))**(1/2)
    x = l0 * (1 + force/k - kT/(2*force*l0) * (1 + y*coth(y)))
    return x

def WLC(coeffs, x_data, force_data, kT):
    # coefficients ordering: [L0, Lp, K]
    l0 = coeffs[0]
    lps = coeffs[1]
    k = coeffs[2]

    x_calc = calculate_x(force_data, l0, lps, k, kT)
    residual = x_data - x_calc
    return residual

def sim_force(total_force_magnitude, key, n_eq_steps, sample_every, n_steps_per_batch, batch_size):

    # Setup our force fn
    force_magnitude_per_end = total_force_magnitude / 2.0
    energy_fn = lambda body: em.energy_fn(body,
                                          seq=seq_oh,
                                          bonded_nbrs=top_info.bonded_nbrs,
                                          unbonded_nbrs=top_info.unbonded_nbrs.T)

    _, force_fn = ext_force.get_force_fn(energy_fn, top_info.n, displacement_fn,
                                         ext_force_bps1,
                                         [0, 0, force_magnitude_per_end], [0, 0, 0, 0])

    _, force_fn = ext_force.get_force_fn(force_fn, top_info.n, displacement_fn,
                                         ext_force_bps2,
                                         [0, 0, -force_magnitude_per_end], [0, 0, 0, 0])
    force_fn = jit(force_fn)

    # Setup equilibration
    assert(n_steps_per_batch % sample_every == 0)
    num_points_per_batch = n_steps_per_batch // sample_every
    def eq_fn(eq_key):
        init_fn, step_fn = simulate.nvt_langevin(force_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(eq_key, init_body, mass=mass)
        def fori_step_fn(t, state):
            return step_fn(state)
        fori_step_fn = jit(fori_step_fn)

        eq_state = lax.fori_loop(0, n_eq_steps, fori_step_fn, init_state)
        return eq_state.position

    # Setup sampling simulation
    def batch_fn(batch_key, eq_body):
        init_fn, step_fn = simulate.nvt_langevin(force_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(batch_key, eq_body, mass=mass)

        def fori_step_fn(t, state):
            return step_fn(state)
        fori_step_fn = jit(fori_step_fn)

        @jit
        def scan_fn(state, step):
            state = lax.fori_loop(0, sample_every, fori_step_fn, state)
            return state, state.position

        _, traj = lax.scan(scan_fn, init_state, jnp.arange(num_points_per_batch))
        return traj

    # Equilibrate and sample
    key, eq_key = random.split(key)
    eq_keys = random.split(eq_key, batch_size)
    eq_bodies = pmap(eq_fn)(eq_keys)

    batch_keys = random.split(key, batch_size)
    batch_trajs = pmap(batch_fn)(batch_keys, eq_bodies)

    num_bases = batch_trajs.center.shape[2]
    assert(batch_trajs.center.shape[3] == 3)

    combined_center = batch_trajs.center.reshape(-1, num_bases, 3)
    combined_quat_vec = batch_trajs.orientation.vec.reshape(-1, num_bases, 4)

    combined_traj = rigid_body.RigidBody(
        center=combined_center,
        orientation=rigid_body.Quaternion(combined_quat_vec))

    return combined_traj


def run(args):
    forces = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
    tom_extensions = {f: calculate_x(f, x_init[0], x_init[1], x_init[2], kT) for f in forces}

    # Load arguments
    key_seed = args['key_seed']
    key = random.PRNGKey(key_seed)
    n_steps_per_batch = args['n_steps_per_batch']
    n_eq_steps = args['n_eq_steps']
    n_devices = jax.local_device_count()
    n_expected_devices = args['n_expected_devices']
    batch_size = n_devices
    assert(n_devices == n_expected_devices)
    sample_every = args['sample_every']
    run_name = args['run_name']
    num_points_per_batch = n_steps_per_batch // sample_every
    num_points = num_points_per_batch * batch_size
    running_avg_interval = args['running_avg_interval']
    min_running_avg_idx = args['min_running_avg_idx']
    force_to_sim = args['force_to_sim']
    assert(force_to_sim in forces)


    # Setup the logging directoroy
    if run_name is None:
        raise RuntimeError(f"Must set a run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    log_path = run_dir / "log.txt"


    with open(log_path, "a") as f:
        f.write(f"Generate trajectory...\n")


    f_key, key = random.split(key)
    start = time.time()
    traj = sim_force(force_to_sim, f_key, n_eq_steps, sample_every,
                     n_steps_per_batch, batch_size=n_devices)
    end = time.time()
    with open(log_path, "a") as f:
        f.write(f"- Simulated force {force_to_sim}: {end - start} seconds\n")


    # Compute projected distances for all forces
    with open(log_path, "a") as f:
        f.write(f"\nComputing projected distances...\n")
    pdists = {}
    start = time.time()
    pdists = vmap(compute_dist)(traj)
    end = time.time()
    with open(log_path, "a") as f:
        f.write(f"- took {end - start} seconds\n")


    # Compute the running average of forces
    with open(log_path, "a") as f:
        f.write(f"\nComputing distance running averages...\n")
    num_running_avg_points = num_points // running_avg_interval
    start = time.time()


    pdists_sampled = pdists[::running_avg_interval]
    running_averages = jnp.cumsum(pdists_sampled) / jnp.arange(1, pdists_sampled.shape[0]+1)
    end = time.time()
    with open(log_path, "a") as f:
        f.write(f"- took {end - start} seconds\n")

    # Plot the running averages
    plt.plot(running_averages, label=f"{force_to_sim}")
    plt.xlabel("Sample")
    plt.ylabel("Avg. Distance (oxDNA units)")
    plt.title(f"Cumulative average")
    plt.legend()
    plt.savefig(run_dir / "len_running_avg.png")
    plt.clf()

    # Plot the running averages (truncated)
    plt.plot(running_averages[min_running_avg_idx:], label=f"{force_to_sim}")
    plt.xlabel("Sample")
    plt.ylabel("Avg. Distance (oxDNA units)")
    plt.title(f"Cumulative average")
    plt.legend()
    plt.savefig(run_dir / "len_running_avg_truncated.png")
    plt.clf()

    # Calculate the running WLC fits
    with open(log_path, "a") as f:
        f.write(f"\nComputing WLC fit running averages...\n")
    all_l0s = list()
    all_lps = list()
    all_ks = list()
    start = time.time()
    for i in range(num_running_avg_points):
        f_lens = list()
        for f in forces:
            if f == force_to_sim:
                f_lens.append(running_averages[i])
            else:
                f_lens.append(tom_extensions[f])
        f_lens = jnp.array(f_lens)

        gn = GaussNewton(residual_fun=WLC)
        gn_sol = gn.run(x_init, x_data=f_lens, force_data=jnp.array(forces), kT=kT).params

        all_l0s.append(gn_sol[0])
        all_lps.append(gn_sol[1])
        all_ks.append(gn_sol[2])
    end = time.time()
    with open(log_path, "a") as f:
        f.write(f"- took {end - start} seconds\n")

    # Plot running WLC fits
    plt.plot(all_l0s, label="l0")
    plt.plot(all_lps, label="lp")
    plt.plot(all_ks, label="k")
    plt.legend()
    plt.title(f"WLC Fit Running Avg.")
    plt.xlabel("Time")
    plt.savefig(run_dir / "wlc_fit_running_avg.png")
    plt.clf()

    # Plot running WLC fits (truncated)
    plt.plot(all_l0s[min_running_avg_idx:], label="l0")
    plt.plot(all_lps[min_running_avg_idx:], label="lp")
    plt.plot(all_ks[min_running_avg_idx:], label="k")
    plt.legend()
    plt.title(f"WLC Fit Running Avg. (Truncated)")
    plt.xlabel("Time")
    plt.savefig(run_dir / "wlc_fit_running_avg_truncanted.png")
    plt.clf()

    # Plot running WLC fits (truncated, SI units)
    plt.plot(jnp.array(all_l0s[min_running_avg_idx:])*utils.nm_per_oxdna_length, label="l0")
    plt.plot(jnp.array(all_lps[min_running_avg_idx:])*utils.nm_per_oxdna_length, label="lp")
    plt.legend()
    plt.title(f"WLC Fit Running Avg. (Truncated)")
    plt.xlabel("Time")
    plt.ylabel("Length (nm)")
    plt.savefig(run_dir / "wlc_fit_lens_running_avg_truncanted_si.png")
    plt.clf()

    plt.plot(jnp.array(all_ks[min_running_avg_idx:])*utils.oxdna_force_to_pn, label="k")
    plt.legend()
    plt.title(f"WLC Fit Running Avg. (Truncated)")
    plt.xlabel("Time")
    plt.ylabel("Extensional Modulus (pN)")
    plt.savefig(run_dir / "wlc_fit_ext_mod_running_avg_truncanted_si.png")
    plt.clf()

    # Test the fit against true values

    ## oxDNA units
    final_f_lens = list()
    for f in forces:
        if f == force_to_sim:
            final_f_lens.append(jnp.mean(pdists))
        else:
            final_f_lens.append(tom_extensions[f])
    final_f_lens = jnp.array(final_f_lens)

    gn = GaussNewton(residual_fun=WLC)
    gn_sol = gn.run(x_init, x_data=final_f_lens, force_data=jnp.array(forces), kT=kT).params

    test_forces = onp.linspace(0.05, 0.8, 20) # in simulation units
    computed_extensions = [calculate_x(force, gn_sol[0], gn_sol[1], gn_sol[2], kT) for force in test_forces]
    tom_extensions = [calculate_x(force, x_init[0], x_init[1], x_init[2], kT) for force in test_forces]

    plt.plot(computed_extensions, test_forces, label="fit")
    plt.scatter(final_f_lens, forces, label="samples")
    plt.plot(tom_extensions, test_forces, label="tom fit")
    plt.xlabel("Extension (oxDNA units)")
    plt.ylabel("Force (oxDNA units)")
    plt.title("Fit Evaluation, oxDNA Units")
    plt.legend()
    plt.savefig(run_dir / "fit_evaluation_oxdna.png")
    plt.clf()


    ## SI units
    test_forces_si = test_forces * utils.oxdna_force_to_pn # in pN
    final_f_lens_si = final_f_lens * utils.nm_per_oxdna_length # in nm
    kT_si = 4.08846006711 # in pN*nm
    forces_si = jnp.array(forces) * utils.oxdna_force_to_pn # pN

    gn_si = GaussNewton(residual_fun=WLC)
    gn_sol_si = gn_si.run(x_init_si, x_data=final_f_lens_si,
                          force_data=forces_si, kT=kT_si).params

    computed_extensions_si = [calculate_x(force, gn_sol_si[0], gn_sol_si[1], gn_sol_si[2], kT_si) for force in test_forces_si] # in nm
    tom_extensions_si = [calculate_x(force, x_init_si[0], x_init_si[1], x_init_si[2], kT_si) for force in test_forces_si] # in nm

    plt.plot(computed_extensions_si, test_forces_si, label="fit")
    plt.scatter(final_f_lens_si, forces_si, label="samples")
    plt.plot(tom_extensions_si, test_forces_si, label="tom fit")
    plt.xlabel("Extension (nm)")
    plt.ylabel("Force (pN)")
    plt.title("Fit Evaluation, SI Units")
    plt.legend()
    plt.savefig(run_dir / "fit_evaluation_si.png")
    plt.clf()


    # Write all the trajectories (after we've done relevant analysis)
    with open(log_path, "a") as f:
        f.write(f"\nWriting trajectory file...\n")

    traj_info = trajectory.TrajectoryInfo(
        top_info, read_from_states=True, states=traj, box_size=conf_info.box_size)
    traj_info.write(run_dir / f"traj.dat", reverse=True)
    end = time.time()
    with open(log_path, "a") as f:
        f.write(f"- {end - start} seconds\n")


def get_parser():
    parser = argparse.ArgumentParser(description="Simulate force extension curve with 110 bp duplex")

    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--n-steps-per-batch', type=int, default=int(1e7),
                        help="Number of total steps (post-equilibration) for sampling per batch. Currently, same for every force, but we should fix this") # FIXME: fix this
    parser.add_argument('--n-eq-steps', type=int, help="Number of equilibration steps")

    parser.add_argument('--n-expected-devices', type=int,
                        help="Expected number of devices. Present as a sanity check. This also serves as the batch size.")
    parser.add_argument('--sample-every', type=int, default=int(1e4),
                        help="Frequency of sampling reference states.")
    parser.add_argument('--key-seed', type=int, default=0, help="Integer seed for key")
    parser.add_argument('--running-avg-interval', type=int, default=10, help="Interval (w.r.t. # of points) for computing running averages")
    parser.add_argument('--min-running-avg-idx', type=int, default=50, help="Min. index for plotting truncated running average")
    parser.add_argument('--force-to-sim', type=float, help="The force to simulate")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
