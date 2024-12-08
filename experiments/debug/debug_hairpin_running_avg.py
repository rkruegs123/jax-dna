import pdb
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
import random
from copy import deepcopy
import functools

import jax
import jax.numpy as jnp
from jax import vmap, jit

from jax_dna.loss import tm

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)



def hairpin_tm_running_avg(traj_hist_files, n_stem_bp, n_dist_thresholds):
    n_files = len(traj_hist_files)
    num_ops = 2
    n_skip_lines = 2 + num_ops

    # Open the first file to read relevant statistics
    with open(traj_hist_files[0], 'r') as f:
        repr_lines = f.readlines()

    assert(repr_lines[0][0] == "#")
    lines_per_hist = 1
    for l in repr_lines[1:]:
        if l[0] != '#':
            lines_per_hist += 1
        else:
            break

    n_lines = len(repr_lines)
    assert(n_lines % lines_per_hist == 0)
    n_hists = n_lines // lines_per_hist

    nvalues = lines_per_hist - 1
    ntemps = len(repr_lines[1].split()) - n_skip_lines # number of *extrapolated* temps

    ## extrapolated temperatures in celsius
    extrapolated_temps = [float(x) * 3000. - 273.15 for x in repr_lines[0].split()[-ntemps:]]
    extrapolated_temps = onp.array(extrapolated_temps)

    # Load contents of all files
    all_flines = list()
    for fname in traj_hist_files:
        with open(fname, 'r') as f:
            all_flines.append(f.readlines())

    # Compute running averages
    all_tms = list()
    all_widths = list()
    start_hist_idx = 50
    # start_hist_idx = 0
    # start_hist_idx = 5
    assert(n_hists > start_hist_idx)
    for hist_idx in tqdm(range(start_hist_idx, n_hists), desc="Traj. histogram running avg."):
        start_line = hist_idx * lines_per_hist
        end_line = start_line + lines_per_hist

        # Construct a matrix of unbiased counts for each temperature and order parameter
        unbiased_counts = onp.zeros((ntemps, nvalues))
        for f_idx in range(n_files):
            f_hist_lines = all_flines[f_idx][start_line:end_line]
            for op_idx, op_line in enumerate(f_hist_lines[1:]): # ignore the header
                tokens = op_line.split()
                op_unbiased_temp_counts = onp.array([float(t) for t in tokens[n_skip_lines:]])
                unbiased_counts[:, op_idx] += op_unbiased_temp_counts

        unbound_op_idxs_extended = onp.array([n_stem_bp*d_idx for d_idx in range(n_dist_thresholds)])
        # bound_op_idxs_extended = onp.array(list(range(1, 1+n_stem_bp)))
        bound_op_idxs_extended = onp.array(list(range(1, n_stem_bp)))

        unbound_unbiased_counts = unbiased_counts[:, unbound_op_idxs_extended]
        bound_unbiased_counts = unbiased_counts[:, bound_op_idxs_extended]

        ratios = list()
        # temps_to_check = extrapolated_temps
        temps_to_check = extrapolated_temps[:-5]
        # for t_idx in range(len(extrapolated_temps)):
        for t_idx in range(len(temps_to_check)):
            unbound_count = unbound_unbiased_counts[t_idx].sum()
            bound_count = bound_unbiased_counts[t_idx].sum()

            ratio = bound_count / unbound_count
            ratios.append(ratio)
        ratios = onp.array(ratios)


        # tm_ = tm.compute_tm(extrapolated_temps, ratios)
        # width_ = tm.compute_width(extrapolated_temps, ratios)

        tm_ = tm.compute_tm(temps_to_check, ratios)
        width_ = tm.compute_width(temps_to_check, ratios)

        all_tms.append(tm_)
        all_widths.append(width_)

    pdb.set_trace()
    return all_tms, all_widths

iter_dir = Path("/home/ryan/Downloads/iter0/")
wfile_path = "/home/ryan/Downloads/iter0/r0/wfile.txt"
weights_df = pd.read_fwf(wfile_path, names=["op1", "op2", "weight"])
num_ops = len(weights_df)
n_stem_bp = len(weights_df.op1.unique())
n_dist_thresholds = len(weights_df.op2.unique())

## Compute running avg. from `traj_hist.dat`
all_traj_hist_fpaths = list()
for r in range(32):
# for r in range(1):
# for r in range(n_sims):

    repeat_dir = iter_dir / f"r{r}"
    all_traj_hist_fpaths.append(repeat_dir / "traj_hist.dat")


### Note: the below should really go in its own loss file
all_running_tms, all_running_widths = hairpin_tm_running_avg(
    all_traj_hist_fpaths, n_stem_bp, n_dist_thresholds)
