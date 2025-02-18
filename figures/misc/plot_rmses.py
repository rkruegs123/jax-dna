import numpy as onp
from pathlib import Path
import pdb
import pickle

from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from figures.utils import colors, labels


font = {'size': 48}
rc('font', **font)
rc('text', usetex=True)


data_dir = Path("figures/misc/data")
opt_dir = data_dir / "test-joint-zf-full-diff-plotting-longer-logging"
output_dir = Path("figures/misc/output")

load_iter = 170
max_iter = 146
assert(max_iter < load_iter)

ref_rmses_fpath = opt_dir / "obj" / f"pdb_ref_rmses_i{load_iter}.pkl"
with open(ref_rmses_fpath, 'rb') as f:
    ref_rmses = pickle.load(f)

ref_times_fpath = opt_dir / "obj" / f"pdb_ref_times_i{load_iter}.pkl"
with open(ref_times_fpath, 'rb') as f:
    ref_times = pickle.load(f)

mean_rmse_path = opt_dir / "log" / "rmse.txt"
mean_rmses = onp.loadtxt(mean_rmse_path)[:max_iter]

# pdb_ids = ['1ZAA', '1A1L', '1AAY']
# colors = ["#1E88E5", "#FFC107", "#004D40"]

pdb_ids = ['1A1L', '1AAY', '1ZAA']
colors = ["#FFC107", "#004D40", "#1E88E5"]

for width, height in [(20, 14), (20, 16), (20, 18)]:
# for width, height in [(20, 14)]:

    fig, ax = plt.subplots(figsize=(width, height))

    for pdb_idx, pdb_id in enumerate(pdb_ids):
        pdb_ref_rmses = onp.array(ref_rmses[pdb_id])
        pdb_ref_times = onp.array(ref_times[pdb_id])

        pdb_ref_rmses = pdb_ref_rmses[pdb_ref_times < max_iter] # Note: order matters here
        pdb_ref_times = pdb_ref_times[pdb_ref_times < max_iter]

        pdb_all_rmses_fpath = opt_dir / "log" / f"{pdb_id}_rmse.txt"
        pdb_all_rmses = onp.loadtxt(pdb_all_rmses_fpath)[:max_iter]

        ax.plot(pdb_all_rmses, color=colors[pdb_idx], label=pdb_id, linewidth=3)
        ax.scatter(pdb_ref_times, pdb_ref_rmses, color=colors[pdb_idx], s=100)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSD (oxDNA units)")

    # y_vals = [10.5, 11, 11.5, 12]
    # ax.set_yticks(y_vals)
    # ax.set_yticklabels([r'$10.5$', r'$11$', r'$11.5$', r'$12$'])

    # x_vals = [0, 50, 100, 150, 200]
    # ax.set_xticks(x_vals)
    # ax.set_xticklabels([r'$0$', r'$50$', r'$100$', r'$150$', r'$200$'])

    leg = ax.legend(title="PDB ID", prop={'size': 35}, loc="upper right")
    # change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(5.0)

    # plt.show()
    # plt.savefig(output_dir / f"individual_rmses_{width}x{height}.pdf")
    plt.savefig(output_dir / f"individual_rmses_{width}x{height}.svg")
