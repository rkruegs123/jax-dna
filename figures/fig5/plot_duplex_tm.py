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


data_dir = Path("figures/fig5/data")
target = 317
tm_dir = data_dir / f"duplex/t{target}"
output_dir = Path("figures/fig5/output")

max_iter = 300

for width, height in [(20, 14), (24, 14), (28, 14)]:

    fig, ax = plt.subplots(figsize=(width, height))

    fin_tms_path = tm_dir / "obj/tms_i85.npy"
    fin_tms = onp.load(fin_tms_path)[:max_iter]

    fin_ref_iters_path = tm_dir / "obj/ref_iters_i85.npy"
    fin_ref_iters = onp.load(fin_ref_iters_path)

    fin_ref_tms_path = tm_dir / "obj/ref_tms_i85.npy"
    fin_ref_tms = onp.load(fin_ref_tms_path)

    keep_ref_iters = fin_ref_iters < max_iter
    fin_ref_iters = fin_ref_iters[keep_ref_iters]
    fin_ref_tms = fin_ref_tms[keep_ref_iters]

    ax.plot(fin_tms, color="black", linewidth=3, label="Simulation")
    ax.scatter(fin_ref_iters, fin_ref_tms, color="black", s=100)

    ax.axhline(y=target, linewidth=3, linestyle="--", color="red", label="Target")

    # ax.fill_between(onp.arange(len(fin_gs)), -100, -80, color='green', alpha=0.3, label="Experimental Uncertainty", transform=ax.get_yaxis_transform())

    ax.set_xlabel("Iteration")
    ax.set_ylabel('Tm (K)', usetex=False)

    # y_vals = [10.5, 11, 11.5, 12]
    # ax.set_yticks(y_vals)
    # ax.set_yticklabels([r'$10.5$', r'$11$', r'$11.5$', r'$12$'])

    # x_vals = [0, 25, 50]
    # ax.set_xticks(x_vals)
    # ax.set_xticklabels([r'$0$', r'$25$', r'$50$'])
    # ax.set_ylim(top=45)

    leg = ax.legend(prop={'size': 48})
    for line in leg.get_lines():
        line.set_linewidth(5.0)

    # plt.show()
    plt.savefig(output_dir / f"tm_duplex_{width}x{height}.pdf")
