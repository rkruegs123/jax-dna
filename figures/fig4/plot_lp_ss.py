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
lp_ss_dir = data_dir / "lp-ss/"
output_dir = Path("figures/fig4/output")
target = 40

max_iter = 90


target_dir = lp_ss_dir / f"t{target}"

fin_lps_path = target_dir / "obj/fin_lps.npy"
fin_lps = onp.load(fin_lps_path)[:max_iter]

fin_ref_iters_path = target_dir / "obj/fin_ref_iters.npy"
fin_ref_iters = onp.load(fin_ref_iters_path)

fin_ref_lps_path = target_dir / "obj/fin_ref_lps.npy"
fin_ref_lps = onp.load(fin_ref_lps_path)

keep_ref_iters = fin_ref_iters < max_iter
fin_ref_iters = fin_ref_iters[keep_ref_iters]
fin_ref_lps = fin_ref_lps[keep_ref_iters]

# for width, height in [(20, 14), (24, 14), (28, 14)]:
for width, height in [(16, 14)]:


    fig, ax = plt.subplots(figsize=(width, height))

    ax.plot(fin_lps, color="black", linewidth=3)
    ax.scatter(fin_ref_iters, fin_ref_lps, color="black", s=100)
    ax.axhline(y=target, linewidth=3, linestyle="--", color="red", label="Target")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Lp (nm)")

    # y_vals = [10.5, 11, 11.5, 12]
    # ax.set_yticks(y_vals)
    # ax.set_yticklabels([r'$10.5$', r'$11$', r'$11.5$', r'$12$'])

    x_vals = [0, 30, 60, 90]
    ax.set_xticks(x_vals)
    ax.set_xticklabels([r'$0$', r'$30$', r'$60$', r'$90$'])
    # ax.set_ylim(top=45)

    leg = ax.legend(prop={'size': 35})
    # change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(5.0)

    plt.show()
    # plt.savefig(output_dir / f"lps_ss_{width}x{height}.pdf")
    plt.close()
