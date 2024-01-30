import argparse
from pathlib import Path
import pdb
from copy import deepcopy
import shutil
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import subprocess

from jax_dna.common import utils
from jax_dna.dna1 import model, oxdna_utils



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

    stem_bp = args['stem_bp']
    loop_nt = args['loop_nt']
    hairpin_basedir = Path("data/sys-defs/hairpins")
    hairpin_dir = hairpin_basedir / f"{stem_bp}bp_stem_{loop_nt}nt_loop"
    assert(hairpin_dir.exists())
    init_conf_path = hairpin_dir / "init.conf"
    top_path = hairpin_dir / "sys.top"
    input_template_path = hairpin_dir / "input"
    op_path = hairpin_dir / "op.txt"
    wfile_path = hairpin_dir / "wfile.txt"

    # Process the weights information
    weights_df = pd.read_fwf(wfile_path, names=["op1", "op2", "weight"])
    num_ops = len(weights_df)
    bins = np.arange(num_ops + 1) - 0.5
    pair2idx = dict()
    idx2pair = dict()
    idx2weight = dict()
    for row_idx, row in weights_df.iterrows():
        op1 = int(row.op1)
        op2 = int(row.op2)
        pair2idx[(op1, op2)] = row_idx
        idx2pair[row_idx] = (op1, op2)
        idx2weight[row_idx] = row.weight


    # Setup the logging directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    # Recompile once at the beginning with default parameters
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    oxdna_utils.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)

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
        shutil.copy(init_conf_path, repeat_dir / "init.conf")


        oxdna_utils.rewrite_input_file(
            input_template_path, repeat_dir,
            temp=f"{t_kelvin}K", steps=n_steps_per_sim,
            init_conf_path=str(repeat_dir / "init.conf"),
            top_path=str(repeat_dir / "sys.top"),
            save_interval=sample_every, seed=random.randrange(100),
            equilibration_steps=n_eq_steps,
            no_stdout_energy=0, weights_file=str(repeat_dir / "wfile.txt"),
            op_file=str(repeat_dir / "op.txt"),
            log_file=str(repeat_dir / "sim.log"),
            restart_step_counter=1 # Because we will not be concatenating the outputs, so we can equilibrate
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
        data = np.array(pd.read_fwf(repeat_dir / "energy.dat", header=None)[[5, 6, 7]])

        for j in range(data.shape[0]):
            op1 = data[j][0]
            op2 = data[j][1]
            pair_idx = pair2idx[(op1, op2)]
            weight = data[j][2]
            assert(np.isclose(idx2weight[pair_idx], weight, atol=1e-3))
            all_op_idxs.append(pair_idx)
            all_weights.append(weight)

    all_op_idxs = np.array(all_op_idxs)
    all_weights = np.array(all_weights)

    # Plot trajectory of order parameters
    plt.plot(all_op_idxs)
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
    probs = np.zeros(num_ops)
    for op_idx, op_weight in zip(all_op_idxs, all_weights):
        probs[op_idx] += 1/op_weight
    assert(np.allclose(probs, probs_hist))

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


    pair2idx = dict()
    idx2pair = dict()
    idx2weight = dict()
    for row_idx, row in updated_weights_df.iterrows():
        op1 = int(row.op1)
        op2 = int(row.op2)
        pair2idx[(op1, op2)] = row_idx
        idx2pair[row_idx] = (op1, op2)
        idx2weight[row_idx] = row.weight

    procs = list()
    for i in range(n_sims):
        repeat_dir = check_weights_dir / f"r{i}"
        repeat_dir.mkdir(parents=False, exist_ok=False)

        shutil.copy(top_path, repeat_dir / "sys.top")
        shutil.copy(optimal_wfile_path, repeat_dir / "wfile.txt")
        shutil.copy(op_path, repeat_dir / "op.txt")
        shutil.copy(init_conf_path, repeat_dir / "init.conf")


        oxdna_utils.rewrite_input_file(
            input_template_path, repeat_dir,
            temp=f"{t_kelvin}K", steps=n_steps_per_sim,
            init_conf_path=str(repeat_dir / "init.conf"),
            top_path=str(repeat_dir / "sys.top"),
            save_interval=sample_every, seed=random.randrange(100),
            equilibration_steps=n_eq_steps,
            no_stdout_energy=0, weights_file=str(repeat_dir / "wfile.txt"),
            op_file=str(repeat_dir / "op.txt"),
            log_file=str(repeat_dir / "sim.log"),
            restart_step_counter=1 # Because we will not be concatenating the outputs, so we can equilibrate
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
        data = np.array(pd.read_fwf(repeat_dir / "energy.dat", header=None)[[5, 6, 7]])

        for j in range(data.shape[0]):
            op1 = data[j][0]
            op2 = data[j][1]
            pair_idx = pair2idx[(op1, op2)]
            weight = data[j][2]
            assert(np.isclose(idx2weight[pair_idx], weight, atol=1e-3))
            all_op_idxs.append(pair_idx)
            all_weights.append(weight)

    all_op_idxs = np.array(all_op_idxs)
    all_weights = np.array(all_weights)

    # Plot trajectory of order parameters
    plt.plot(all_op_idxs)
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
    parser.add_argument('--temp', type=float, default=utils.DEFAULT_TEMP,
                        help="Temperature in kelvin")

    parser.add_argument('--stem-bp', type=int, default=4,
                        help="Number of base pairs comprising the stem")
    parser.add_argument('--loop-nt', type=int, default=8,
                        help="Number of nucleotides comprising the loop")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
