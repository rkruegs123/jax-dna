import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp
import subprocess
import shutil
import shlex
import argparse
import pandas as pd
import random

import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax

from jax_dna.common import utils, topology, trajectory, center_configuration, checkpoint
from jax_dna.loss import persistence_length
from jax_dna.dna1 import model, oxdna_utils

from jax.config import config
config.update("jax_enable_x64", True)


ALL_FORCES = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.375]
num_forces = len(ALL_FORCES)
ext_force_bps1 = [5, 214] # should each experience force_per_nuc
ext_force_bps2 = [104, 115] # should each experience -force_per_nuc
dir_force_axis = jnp.array([0, 0, 1])



def simulate(args):
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
    n_sims_per_force = args['n_sims_per_force']
    n_steps_per_sim = args['n_steps_per_sim']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    assert(n_steps_per_sim % sample_every == 0)
    n_ref_states_per_sim = n_steps_per_sim // sample_every
    n_ref_states = n_ref_states_per_sim * n_sims
    run_name = args['run_name']
    oxdna_path = Path(args['oxdna_path'])
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"
    t_kelvin = args['temp']
    oxdna_cuda_device = args['oxdna_cuda_device']
    oxdna_cuda_list = args['oxdna_cuda_list']

    low_forces_input = args['low_forces']
    low_forces = list()
    for lf in low_forces_input:
        lf_float = float(lf)
        assert(lf_float in ALL_FORCES)
        low_forces.append(lf_float)

    low_force_multiplier = args['low_force_multiplier']
    assert(low_force_multiplier >= 1)

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
    sys_basedir = Path("data/templates/force-ext")
    externals_basedir = sys_basedir / "externals"
    ## note: no external_path set
    # external_path = externals_basedir / f"external_{force_per_nuc}.conf"
    if not external_path.exists():
        raise RuntimeError(f"No external forces file at location: {external_path}")
    input_template_path = sys_basedir / "input"


    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
    assert(seq_oh.shape[0] == 220) # 110 bp duplex
    n = seq_oh.shape[0]

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        reverse_direction=True
        # reverse_direction=False
    )

    box_size = conf_info.box_size # Only for writing the trajectory
    displacement_fn, shift_fn = space.free()

    dt = 5e-3
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT

    # Do the simulation
    random.seed(key)

    params = deepcopy(model.EMPTY_BASE_PARAMS)
    iter_dir = ref_traj_dir / f"simulation"
    iter_dir.mkdir(parents=False, exist_ok=False)

    oxdna_utils.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)

    sim_start = time.time()
    if device == "cpu":
        procs = list()


    ## Setup the simulation information
    sim_lengths = list()
    sim_forces = list()
    sim_idxs = list()
    sim_start_steps = list()

    sim_idx = 0
    for force in ALL_FORCES:
        sim_start_step = 0
        for f_sim_idx in range(n_sims_per_force):
            sim_length = n_steps_per_sim
            if force in low_forces:
                sim_length *= low_force_multiplier

            sim_lengths.append(sim_length)
            sim_forces.append(force)
            sim_idxs.append(sim_idx)
            sim_start_steps.append(sim_start_step)

            sim_idx += 1
            sim_start_step += sim_length

    # TODO: run all the simulation
    # note: keeping naming convention as r{idx} for comptability with mps.sh
    for sim_length, sim_force, sim_idx, start_step in zip(sim_lengths, sim_forces, sim_idxs, sim_start_steps):
        repeat_dir = iter_dir / f"r{sim_idx}"
        repeat_dir.mkdir(parents=False, exist_ok=False)

        external_path = externals_basedir / f"external_{sim_force}.conf"

        shutil.copy(top_path, repeat_dir / "sys.top")
        shutil.copy(external_path, repeat_dir / "external.conf")

        init_conf_info = deepcopy(conf_info)

        init_conf_info.traj_df.t = onp.full(seq_oh.shape[0], start_step)
        init_conf_info.write(repeat_dir / "init.conf", reverse=False, write_topology=False)

        oxdna_utils.rewrite_input_file(
            input_template_path, repeat_dir,
            temp=f"{t_kelvin}K", steps=sim_length,
            init_conf_path=str(repeat_dir / "init.conf"),
            top_path=str(repeat_dir / "sys.top"),
            save_interval=sample_every, seed=random.randrange(100),
            equilibration_steps=n_eq_steps, dt=dt,
            no_stdout_energy=0, backend=backend,
            cuda_device=oxdna_cuda_device, cuda_list=oxdna_cuda_list,
            log_file=str(repeat_dir / "sim.log"),
            external_forces_file=str(repeat_dir / "external.conf")
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
        mps_cmd = f"./scripts/mps.sh {oxdna_path} {iter_dir} {n_sims}"
        mps_proc = subprocess.run(mps_cmd, shell=True)
        if mps_proc.returncode != 0:
            raise RuntimeError(f"Generating states via MPS failed with error code: {mps_proc.returncode}")
    else:
        raise RuntimeError(f"Invalid device: {device}")


    return sim_lengths, sim_forces, sim_idxs, sim_start_steps


def analyze(args):

    # FIXME: combine force input files into master force output files
    # FIXME: compute running averages with and without known persistence length (could add the Lp as an argument)
    # FIXME: should also add a simulate-only and analyze-only flag

    raise NotImplementedError


def run(args):
    simulate(args)
    analyze(args)


def get_parser():

    # Simulation arguments
    parser = argparse.ArgumentParser(description="Optimize a single force extension")

    parser.add_argument('--device', type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--n-sims-per-force', type=int, default=2,
                        help="Number of individual simulations per force")
    parser.add_argument('--n-steps-per-sim', type=int, default=100000,
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int,
                        # default=10000,
                        default=0,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='Run name')
    parser.add_argument('--temp', type=float, default=utils.DEFAULT_TEMP,
                        help="Simulation temperature in Kelvin")
    parser.add_argument('--oxdna-cuda-device', type=int, default=0,
                        help="CUDA device for running oxDNA simulations")
    parser.add_argument('--oxdna-cuda-list', type=str, default="verlet",
                        choices=["no", "verlet"],
                        help="CUDA neighbor lists")


    parser.add_argument('--low-forces', nargs='*',
                        help='Forces for which we simulate for longer')
    parser.add_argument('--low-force-multiplier', type=int, default=1,
                        help="Multiplicative factor for low force simulation length")


    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    assert(args['n_eq_steps'] == 0)

    run(args)
