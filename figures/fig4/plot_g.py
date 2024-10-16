import numpy as onp
from pathlib import Path
import pdb

from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from figures.utils import colors, labels


font = {'size': 48}
rc('font', **font)
rc('text', usetex=True)


data_dir = Path("figures/fig4/data")
g_dir = data_dir / "moduli/g-all"
output_dir = Path("figures/fig4/output")

max_iter = 300

for width, height in [(20, 14), (24, 14), (28, 14)]:

    fig, ax = plt.subplots(figsize=(width, height))


    fin_gs_path = g_dir / "obj/fin_gs.npy"
    fin_gs = onp.load(fin_gs_path)[:max_iter]

    fin_ref_iters_path = g_dir / "obj/fin_ref_iters.npy"
    fin_ref_iters = onp.load(fin_ref_iters_path)

    fin_ref_gs_path = g_dir / "obj/fin_ref_gs.npy"
    fin_ref_gs = onp.load(fin_ref_gs_path)

    keep_ref_iters = fin_ref_iters < max_iter
    fin_ref_iters = fin_ref_iters[keep_ref_iters]
    fin_ref_gs = fin_ref_gs[keep_ref_iters]

    ax.plot(fin_gs, color="black", linewidth=3)
    ax.scatter(fin_ref_iters, fin_ref_gs, color="black", s=100)

    ax.fill_between(onp.arange(len(fin_gs)), -100, -80, color='green', alpha=0.3, label="Experimental Uncertainty", transform=ax.get_yaxis_transform())

    ax.set_xlabel("Iteration")
    ax.set_ylabel('g ($pN·nm$)', usetex=False)

    # y_vals = [10.5, 11, 11.5, 12]
    # ax.set_yticks(y_vals)
    # ax.set_yticklabels([r'$10.5$', r'$11$', r'$11.5$', r'$12$'])

    # x_vals = [0, 25, 50]
    # ax.set_xticks(x_vals)
    # ax.set_xticklabels([r'$0$', r'$25$', r'$50$'])
    # ax.set_ylim(top=45)

    leg = ax.legend(prop={'size': 48})

    # plt.show()
    # plt.savefig(output_dir / f"gs_{width}x{height}.pdf")
    plt.savefig(output_dir / f"gs_{width}x{height}.svg")
