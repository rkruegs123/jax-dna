import argparse
from pathlib import Path
import pdb
from copy import deepcopy
import shutil
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as onp
from tabulate import tabulate
import subprocess

import jax.numpy as jnp
from jax import vmap

from jax_dna.common import utils
from jax_dna.dna1 import model as model1
from jax_dna.dna1 import oxdna_utils as oxdna_utils1
from jax_dna.dna2 import model as model2
from jax_dna.dna2 import oxdna_utils as oxdna_utils2




def run(args):
    random.seed(0)

    device = args['device']
    assert(device == "cpu")

    n_threads = args['n_threads']
    n_sims = args['n_sims']
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
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    dt = 5e-3

    extrapolate_temps = jnp.array([float(et) for et in args['extrapolate_temps']]) # in Kelvin
    assert(jnp.all(extrapolate_temps[:-1] <= extrapolate_temps[1:])) # check that temps. are sorted
    n_extrap_temps = len(extrapolate_temps)
    extrapolate_kts = vmap(utils.get_kt)(extrapolate_temps)
    extrapolate_temp_str = ', '.join([f"{tc}K" for tc in extrapolate_temps])


    tm_dir = Path(args['tm_dir'])
    assert(tm_dir.exists())

    seq_dep = args['seq_dep']
    assert(not seq_dep)

    interaction = args['interaction']
    if interaction == "DNA_nomesh" or interaction == "DNA2_nomesh":
        salt = 0.5
    elif interaction == "RNA2":
        salt = 1.0
    else:
        raise RuntimeError(f"Invalid interaction type: {interaction}")

    conf_path_unbound = tm_dir / "init_unbound.conf"
    conf_path_bound = tm_dir / "init_bound.conf"

    top_path = tm_dir / "sys.top"
    input_template_path = tm_dir / "input"
    op_path = tm_dir / "op.txt"
    wfile_path = tm_dir / "wfile.txt"

    # Process the weights information
    weights_df = pd.read_fwf(wfile_path, names=["op", "weight"])
    num_ops = len(weights_df)
    bins = onp.arange(num_ops + 1) - 0.5

    op2idx = dict()
    idx2op = dict()
    idx2weight = dict()
    for row_idx, row in weights_df.iterrows():
        op = int(row.op)
        op2idx[op] = row_idx
        idx2op[row_idx] = op
        idx2weight[row_idx] = row.weight

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

    # Recompile once at the beginning with default parameters
    if interaction == "DNA_nomesh":
        params = deepcopy(model1.EMPTY_BASE_PARAMS)
        oxdna_utils1.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)
    elif interaction == "DNA2_nomesh":
        params = deepcopy(model2.default_base_params_seq_avg)
        oxdna_utils2.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)
    elif interaction == "RNA2":
        # technically we don't have to recompile because we never do, but might as well
        params = deepcopy(model2.default_base_params_seq_avg)
        oxdna_utils2.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)
    else:
        raise RuntimeError(f"Invalid interaction type: {interaction}")

    # Setup a run with bad weights
    initial_weights_dir = run_dir / "initial_weights"
    initial_weights_dir.mkdir(parents=False, exist_ok=False)

    procs = list()
    for i in range(n_sims):
        repeat_dir = initial_weights_dir / f"r{i}"
        repeat_dir.mkdir(parents=False, exist_ok=False)

        shutil.copy(top_path, repeat_dir / "sys.top")
        shutil.copy(wfile_path, repeat_dir / "wfile.txt")
        shutil.copy(op_path, repeat_dir / "op.txt")

        if i % 2 == 0:
            shutil.copy(conf_path_bound, repeat_dir / "init.conf")
        else:
            shutil.copy(conf_path_unbound, repeat_dir / "init.conf")

        oxdna_utils1.rewrite_input_file(
            input_template_path, repeat_dir,
            temp=f"{t_kelvin}K", steps=n_steps_per_sim,
            init_conf_path=str(repeat_dir / "init.conf"),
            top_path=str(repeat_dir / "sys.top"),
            save_interval=sample_every, seed=random.randrange(100),
            equilibration_steps=n_eq_steps,
            no_stdout_energy=0, weights_file=str(repeat_dir / "wfile.txt"),
            op_file=str(repeat_dir / "op.txt"),
            log_file=str(repeat_dir / "sim.log"),
            restart_step_counter=1, # Because we will not be concatenating the outputs, so we can equilibrate
            interaction_type=interaction,
            salt_concentration=salt,
            extrapolate_hist=extrapolate_temp_str
        )

        procs.append(subprocess.Popen([oxdna_exec_path, repeat_dir / "input"]))

    for p in procs:
        p.wait()

    for p in procs:
        rc = p.returncode
        if rc != 0:
            raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")


    # Read in energy files (which include OP metadata)
    all_op_idxs = list()
    all_weights = list()
    for i in range(n_sims):
        repeat_dir = initial_weights_dir / f"r{i}"
        data = onp.array(pd.read_fwf(repeat_dir / "energy.dat", header=None)[[5, 6]])

        for j in range(data.shape[0]):
            op = data[j][0]
            op_idx = op2idx[op]
            weight = data[j][1]
            assert(onp.isclose(idx2weight[op_idx], weight, atol=1e-3))
            all_op_idxs.append(op_idx)
            all_weights.append(weight)

    all_op_idxs = onp.array(all_op_idxs)
    all_weights = onp.array(all_weights)

    # Plot trajectory of order parameters
    plt.plot(all_op_idxs)
    for i in range(n_sims):
        plt.axvline(x=i*n_ref_states_per_sim, linestyle="--", color="red")
    plt.savefig(initial_weights_dir / "op_trajectory.png")
    plt.clf()

    # Plot the biased counts
    plt.hist(all_op_idxs, bins=bins)
    plt.xlabel("O.P. Index")
    plt.ylabel("Count")
    plt.savefig(initial_weights_dir / "biased_counts.png")
    plt.clf()


    # Unbias the counts
    f, ax = plt.subplots(1, 1, figsize=(6, 6))
    probs_hist = ax.hist(all_op_idxs, bins=bins,
                         weights=1 / all_weights)[0]
    plt.savefig(initial_weights_dir / "unbiased_counts.png")
    plt.clf()

    # Compute the optimal weights
    probs = onp.zeros(num_ops)
    for op_idx, op_weight in zip(all_op_idxs, all_weights):
        probs[op_idx] += 1/op_weight
    assert(onp.allclose(probs, probs_hist))

    normed = probs / sum(probs)
    optimal_weights = 1 / normed

    ## Save to file
    updated_weights_df = weights_df.copy(deep=True)
    updated_weights_df.weight = optimal_weights

    optimal_wfile_path = run_dir / "optimal_weights.txt"
    with open(optimal_wfile_path, "w") as of:
        content = tabulate(updated_weights_df.values.tolist(),
                           tablefmt="plain", numalign="left")
        of.write(content + "\n")


    # Check the optimal weights via a new round of simulations
    check_weights_dir = run_dir / "check_weights"
    check_weights_dir.mkdir(parents=False, exist_ok=False)


    op2idx = dict()
    idx2op = dict()
    idx2weight = dict()
    for row_idx, row in updated_weights_df.iterrows():
        op = int(row.op)
        op2idx[op] = row_idx
        idx2op[row_idx] = op
        idx2weight[row_idx] = row.weight

    procs = list()
    for i in range(n_sims):
        repeat_dir = check_weights_dir / f"r{i}"
        repeat_dir.mkdir(parents=False, exist_ok=False)

        shutil.copy(top_path, repeat_dir / "sys.top")
        shutil.copy(optimal_wfile_path, repeat_dir / "wfile.txt")
        shutil.copy(op_path, repeat_dir / "op.txt")

        if i % 2 == 0:
            shutil.copy(conf_path_bound, repeat_dir / "init.conf")
        else:
            shutil.copy(conf_path_unbound, repeat_dir / "init.conf")


        oxdna_utils1.rewrite_input_file(
            input_template_path, repeat_dir,
            temp=f"{t_kelvin}K", steps=n_steps_per_sim,
            init_conf_path=str(repeat_dir / "init.conf"),
            top_path=str(repeat_dir / "sys.top"),
            save_interval=sample_every, seed=random.randrange(100),
            equilibration_steps=n_eq_steps,
            no_stdout_energy=0, weights_file=str(repeat_dir / "wfile.txt"),
            op_file=str(repeat_dir / "op.txt"),
            log_file=str(repeat_dir / "sim.log"),
            restart_step_counter=1, # Because we will not be concatenating the outputs, so we can equilibrate
            interaction_type=interaction,
            salt_concentration=salt,
            extrapolate_hist=extrapolate_temp_str
        )

        procs.append(subprocess.Popen([oxdna_exec_path, repeat_dir / "input"]))

    for p in procs:
        p.wait()

    for p in procs:
        rc = p.returncode
        if rc != 0:
            raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")

    # Read in energy files (which include OP metadata)
    all_op_idxs = list()
    all_weights = list()
    for i in range(n_sims):
        repeat_dir = check_weights_dir / f"r{i}"
        data = onp.array(pd.read_fwf(repeat_dir / "energy.dat", header=None)[[5, 6]])

        for j in range(data.shape[0]):
            op = data[j][0]
            op_idx = op2idx[op]
            weight = data[j][1]
            assert(onp.isclose(idx2weight[op_idx], weight, atol=1e-3))
            all_op_idxs.append(op_idx)
            all_weights.append(weight)

    all_op_idxs = onp.array(all_op_idxs)
    all_weights = onp.array(all_weights)

    # Plot trajectory of order parameters
    plt.plot(all_op_idxs)
    for i in range(n_sims):
        plt.axvline(x=i*n_ref_states_per_sim, linestyle="--", color="red")
    plt.savefig(check_weights_dir / "op_trajectory.png")
    plt.clf()


    # Plot the biased counts
    plt.hist(all_op_idxs, bins=bins)
    plt.xlabel("O.P. Index")
    plt.ylabel("Count")
    plt.savefig(check_weights_dir / "biased_counts.png")
    plt.clf()


    # Unbias the counts
    f, ax = plt.subplots(1, 1, figsize=(6, 6))
    probs_hist = ax.hist(all_op_idxs, bins=bins,
                         weights=1 / all_weights)[0]
    plt.savefig(check_weights_dir / "unbiased_counts.png")
    plt.clf()




def get_parser():

    parser = argparse.ArgumentParser(description="Find the weights for a given hairpin")

    parser.add_argument('--device', type=str, default="cpu", choices=["cpu"])
    parser.add_argument('--n-threads', type=int, default=2,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--n-sims', type=int, default=2,
                        help="Number of individual simulations")
    parser.add_argument('--n-steps-per-sim', type=int, default=int(5e6),
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=int(1e5),
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=int(1e3),
                        help="Frequency of sampling reference states.")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='oxDNA base directory')
    parser.add_argument('--temp', type=float, default=300.15,
                        help="Temperature in kelvin")
    parser.add_argument('--extrapolate-temps', nargs='+',
                        help='Temperatures for extrapolation in Kelvin in ascending order',
                        default=[282.15, 285.15, 288.15, 291.15, 294.15, 297.15, 303.15, 306.15,
                                 309.15, 312.15], # corresponding to 9C, 12C, 15C, 18C, 21C, 24C, 30C, 33C, 36C, 39C
                        required=True)

    parser.add_argument('--tm-dir', type=str,
                        default="data/sys-defs/tm-1op/8bp",
                        help='Directory for duplex system')

    parser.add_argument('--interaction', type=str,
                        default="DNA_nomesh", choices=["DNA_nomesh", "DNA2_nomesh", "RNA2"],
                        help='Interaction type')

    parser.add_argument('--seq-dep', action='store_true')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
