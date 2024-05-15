# ruff: noqa
# fmt: off
import math
import numpy as onp
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import argparse
from pathlib import Path

import jax.numpy as jnp


def compute_tm(temps, finfs):
    x = jnp.flip(finfs)
    y = jnp.flip(temps)
    xin = jnp.arange(0.1, 1., 0.1)
    f = jnp.interp(xin, x, y)
    return f[4] # evaluate at 0.5

def compute_width(temps, finfs):
    x = jnp.flip(finfs)
    y = jnp.flip(temps)
    xin = jnp.arange(0.1, 1., 0.1)
    f = jnp.interp(xin, x, y)
    return f[1] - f[7] # 0.2 to 0.8

def compute_finf(counts):
    unbound_count = counts[0]
    bound_count = jnp.sum(counts[1:])
    ratio = bound_count / unbound_count
    finf = 1 + 1/(2*ratio) - jnp.sqrt((1 + 1/(2*ratio))**2 - 1)
    return finf

# note: assumes 1 order parameter
def traj_hist_running_avg_1d(traj_hist_files):
    n_files = len(traj_hist_files)
    n_skip_lines = 3

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

        finfs = []
        for m in unbiased_counts:
            finf = compute_finf(m)
            finfs.append(finf)
        finfs = onp.array(finfs)

        tm = compute_tm(extrapolated_temps, finfs)
        width = compute_width(extrapolated_temps, finfs)

        all_tms.append(tm)
        all_widths.append(width)

    return all_tms, all_widths


if __name__ == "__main__":
    files = [
        "output/tm-sim-test/ref_traj/iter0/r0/traj_hist.dat",
        "output/tm-sim-test/ref_traj/iter0/r1/traj_hist.dat"
    ]
    all_tms, all_widths = traj_hist_running_avg_1d(files)

    plt.plot(all_tms)
    plt.xlabel("Iteration")
    plt.ylabel("Tm")
    plt.show()

    plt.plot(all_widths)
    plt.xlabel("Iteration")
    plt.ylabel("Width")
    plt.show()
