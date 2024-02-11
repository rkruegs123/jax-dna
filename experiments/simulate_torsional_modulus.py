import pdb
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
import functools
import time
import matplotlib.pyplot as plt
import numpy as onp
import subprocess
import shutil
import argparse
import pandas as pd
import random
import seaborn as sns

import optax
import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.dna1 import model, oxdna_utils
from jax_dna.loss import pitch

from jax.config import config
config.update("jax_enable_x64", True)


checkpoint_every = 50
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def get_all_quartets(n_nucs_per_strand):
    s1_nucs = list(range(n_nucs_per_strand))
    s2_nucs = list(range(n_nucs_per_strand, n_nucs_per_strand*2))
    s2_nucs.reverse()

    bps = list(zip(s1_nucs, s2_nucs))
    n_bps = len(s1_nucs)
    all_quartets = list()
    for i in range(n_bps-1):
        bp1 = bps[i]
        bp2 = bps[i+1]
        all_quartets.append(bp1 + bp2)
    return jnp.array(all_quartets, dtype=jnp.int32)

def run(args):
    # Load parameters
    device = args['device']
    if device == "cpu":
        backend = "CPU"
    elif device == "gpu":
        backend = "CUDA"
    else:
        raise RuntimeError(f"Invalid device: {device}")
    n_threads = args['n_threads']
    key = args['key']
    n_sims = args['n_sims']
    if device == "gpu":
        raise NotImplementedError(f"Still need to implement GPU version...")
    n_steps_per_sim = args['n_steps_per_sim']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    assert(n_steps_per_sim % sample_every == 0)
    n_ref_states_per_sim = n_steps_per_sim // sample_every
    n_ref_states = n_ref_states_per_sim * n_sims
    run_name = args['run_name']
    oxdna_path = Path(args['oxdna_path'])
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"
    oxdna_cuda_device = args['oxdna_cuda_device']
    oxdna_cuda_list = args['oxdna_cuda_list']

    # Setup the logging directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    params_str += f"n_ref_states: {n_ref_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    # Load the system
    sys_basedir = Path("data/templates/torsional-modulus")
    input_template_path = sys_basedir / "input"

    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
    assert(seq_oh.shape[0] == 60) # 30 bp duplex
    n = seq_oh.shape[0]

    quartets = get_all_quartets(n_nucs_per_strand=n // 2)
    quartets = quartets[5:24] # Restrict to central 20 bp

    rise_per_bp = 3.4 / utils.ang_per_oxdna_length # oxDNA length units
    contour_length = quartets.shape[0] * rise_per_bp # oxDNA length units

    # FIXME: use real oxDNA theta0. Maybe from a calculation of the pitch?
    # theta0 is 35 degrees per bp -- http://nanobionano.unibo.it/StrutturisticaAcNucl/BryantCozzarelliBustamanteDNATorqueMeasurement.pdf
    exp_theta0_per_bp = 35 * jnp.pi/180.0 # radians
    exp_theta0 = exp_theta0_per_bp * quartets.shape[0]

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        reverse_direction=True
    )

    box_size = conf_info.box_size # Only for writing the trajectory
    displacement_fn, shift_fn = space.free()


    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT

    # Do the simulation
    random.seed(key)

    params = deepcopy(model.EMPTY_BASE_PARAMS)
    iter_dir = run_dir / f"simulation"
    iter_dir.mkdir(parents=False, exist_ok=False)

    oxdna_utils.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)

    sim_start = time.time()
    if device == "cpu":
        procs = list()

    ## Setup the simulation information
    for r in range(n_sims):
        repeat_dir = iter_dir / f"r{r}"
        repeat_dir.mkdir(parents=False, exist_ok=False)

        shutil.copy(top_path, repeat_dir / "sys.top")

        init_conf_info = deepcopy(conf_info)

        init_conf_info.traj_df.t = onp.full(seq_oh.shape[0], r*n_steps_per_sim)
        init_conf_info.write(repeat_dir / "init.conf", reverse=False, write_topology=False)

        oxdna_utils.rewrite_input_file(
            input_template_path, repeat_dir,
            temp=f"{t_kelvin}K", steps=n_steps_per_sim,
            init_conf_path=str(repeat_dir / "init.conf"),
            top_path=str(repeat_dir / "sys.top"),
            save_interval=sample_every, seed=random.randrange(100),
            equilibration_steps=n_eq_steps, dt=dt,
            no_stdout_energy=0, backend=backend,
            cuda_device=oxdna_cuda_device, cuda_list=oxdna_cuda_list,
            log_file=str(repeat_dir / "sim.log"),
        )

        if device == "cpu":
            procs.append(subprocess.Popen([oxdna_exec_path, repeat_dir / "input"]))

    if device == "cpu":
        for p in procs:
            p.wait()

        for p in procs:
            rc = p.returncode
            if rc != 0:
                raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")
    elif device == "gpu":
        mps_script_path = "./scripts/mps.sh"
        mps_cmd = f"{mps_script_path} {oxdna_path} {iter_dir} {n_sims}"
        mps_proc = subprocess.run(mps_cmd, shell=True)
        if mps_proc.returncode != 0:
            raise RuntimeError(f"Generating states via MPS failed with error code: {mps_proc.returncode}")

    else:
        raise RuntimeError(f"Invalid device: {device}")

    sim_end = time.time()

    # Analyze

    ## Combine trajectories
    combine_cmd = "cat "
    for r in range(n_sims):
        repeat_dir = iter_dir / f"r{r}"
        combine_cmd += f"{repeat_dir}/output.dat "
    combine_cmd += f"> {iter_dir}/output.dat"
    combine_proc = subprocess.run(combine_cmd, shell=True)
    if combine_proc.returncode != 0:
        raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")

    ## Load states from oxDNA simulation
    load_start = time.time()
    traj_info = trajectory.TrajectoryInfo(
        top_info, read_from_file=True,
        traj_path=iter_dir / "output.dat",
        reverse_direction=True)
    traj_states = traj_info.get_states()
    n_traj_states = len(traj_states)
    assert(n_traj_states == n_ref_states)
    traj_states = utils.tree_stack(traj_states)
    load_end = time.time()

    ## Load the oxDNA energies
    energy_df_columns = ["time", "potential_energy", "kinetic_energy", "total_energy"]
    energy_dfs = [pd.read_csv(iter_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
                              delim_whitespace=True)[1:] for r in range(n_sims)]
    energy_df = pd.concat(energy_dfs, ignore_index=True)

    ## Generate an energy function
    em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
    energy_fn = lambda body: em.energy_fn(
        body,
        seq=seq_oh,
        bonded_nbrs=top_info.bonded_nbrs,
        unbonded_nbrs=top_info.unbonded_nbrs.T)
    energy_fn = jit(energy_fn)

    ## Check energies
    calc_start = time.time()
    energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
    _, calc_energies = scan(energy_scan_fn, None, traj_states)
    calc_end = time.time()

    gt_energies = energy_df.potential_energy.to_numpy() * seq_oh.shape[0]

    atol_places = 3
    tol = 10**(-atol_places)
    energy_diffs = list()
    for i, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
        diff = onp.abs(calc - gt)
        energy_diffs.append(diff)

    sns.distplot(calc_energies, label="Calculated", color="red")
    sns.distplot(gt_energies, label="Reference", color="green")
    plt.legend()
    plt.savefig(run_dir / f"energies.png")
    plt.clf()

    sns.histplot(energy_diffs)
    plt.savefig(run_dir / f"energy_diffs.png")
    plt.clf()

    # Compute the torsional modulus
    fjoules_per_oxdna_energy = utils.joules_per_oxdna_energy * 1e15 # 1e15 fJ per J
    fm_per_oxdna_length = utils.ang_per_oxdna_length * 1e5 # 1e5 fm per Angstrom
    all_theta = list()
    all_theta0 = list()
    running_avg_freq = 10
    min_rs_idx = 50 # minimum idx for running average
    all_c_fjfm = list()
    for rs_idx in tqdm(range(n_ref_states)):
        ref_state = traj_states[rs_idx]
        pitches = pitch.get_all_pitches(ref_state, quartets, displacement_fn, model.com_to_hb)
        theta = pitches.sum()
        all_theta.append(theta)
        theta0 = onp.mean(all_theta)
        all_theta0.append(theta0)

        if rs_idx % running_avg_freq == 0 and rs_idx > min_rs_idx:

            curr_theta0 = onp.mean(all_theta)
            all_dtheta = onp.array(all_theta) - curr_theta0
            all_dtheta_sqr = all_dtheta**2
            
            avg_dtheta_sqr = onp.mean(all_dtheta_sqr)

            C_oxdna = (kT * contour_length) / avg_dtheta_sqr # oxDNA units
            C_fjfm = C_oxdna * fjoules_per_oxdna_energy * fm_per_oxdna_length
            all_c_fjfm.append(C_fjfm)

    all_theta = onp.array(all_theta)
    all_theta0 = onp.array(all_theta0)
    theta0 = onp.mean(all_theta0)

    all_dtheta = all_theta - theta0
    all_dtheta_sqr = all_dtheta**2


    ## Plot running average of theta0
    plt.plot(all_theta0, label="Running Avg.")
    plt.ylabel("Theta0")
    plt.axhline(y=exp_theta0, linestyle="--", label="Expected Theta0")
    plt.legend()
    plt.savefig(run_dir / "theta0_running_avg.png")
    plt.clf()
                         

    ## Plot running average of C
    plt.plot(all_c_fjfm)
    plt.ylabel("C (fJfm)")
    plt.title("C running average")
    plt.savefig(run_dir / "c_running_avg.png")
    plt.clf()

    ## Plot the distribution of thetas
    sns.histplot(all_dtheta)
    plt.savefig(run_dir / f"dtheta_dist.png")
    plt.clf()

    sns.histplot(all_dtheta_sqr)
    plt.savefig(run_dir / f"dtheta_sqr_dist.png")
    plt.clf()

    ## Compute the final C
    avg_dtheta_sqr = onp.mean(all_dtheta_sqr)
    C_oxdna = (kT * contour_length) / avg_dtheta_sqr # oxDNA units
    C_fjfm = C_oxdna * fjoules_per_oxdna_energy * fm_per_oxdna_length


    summary_str = f"Avg. dtheta sqr: {avg_dtheta_sqr}\n"
    summary_str += f"Avg. dtheta: {onp.mean(all_dtheta)}\n"
    summary_str += f"C (oxDNA units): {C_oxdna}\n"
    summary_str += f"C (fJfm): {C_fjfm}\n"
    summary_str += f"Theta_0: {theta0}\n"
    with open(run_dir / "summary.txt", "w+") as f:
        f.write(summary_str)



def get_parser():

    parser = argparse.ArgumentParser(description="Calculate persistence length via standalone oxDNA package")

    parser.add_argument('--device', type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--n-sims', type=int, default=2,
                        help="Number of individual simulations")
    parser.add_argument('--n-steps-per-sim', type=int, default=100000,
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=0,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='Run name')
    parser.add_argument('--oxdna-cuda-device', type=int, default=0,
                        help="CUDA device for running oxDNA simulations")
    parser.add_argument('--oxdna-cuda-list', type=str, default="verlet",
                        choices=["no", "verlet"],
                        help="CUDA neighbor lists")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())
    assert(args['n_eq_steps'] == 0)

    run(args)
