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
output_dir = Path("figures/fig4/output")

joint = False

if joint:
    joint_dir = data_dir / "moduli/c-seff-joint"
    max_iter = 300

    fin_cs_path = joint_dir / "obj/fin_cs.npy"
    fin_seffs_path = joint_dir / "obj/fin_seffs.npy"

    fin_ref_iters_cs_path = joint_dir / "obj/fin_ref_iters.npy"
    fin_ref_cs_path = joint_dir / "obj/fin_ref_cs.npy"

    fin_ref_iters_seffs_path = joint_dir / "obj/fin_ref_iters.npy"
    fin_ref_seffs_path = joint_dir / "obj/fin_ref_seffs.npy"

else:

    max_iter = 60

    seff_dir = data_dir / "moduli/seff-all"
    c_dir = data_dir / "moduli/c-all"

    fin_cs_path = c_dir / "obj/cs_i90.npy"
    fin_seffs_path = seff_dir / "obj/seffs_i120.npy"

    fin_ref_iters_cs_path = c_dir / "obj/ref_iters_i90.npy"
    fin_ref_cs_path = c_dir / "obj/ref_cs_i90.npy"

    fin_ref_iters_seffs_path = seff_dir / "obj/ref_iters_i120.npy"
    fin_ref_seffs_path = seff_dir / "obj/ref_seffs_i120.npy"


fin_cs = onp.load(fin_cs_path)[:max_iter]
fin_seffs = onp.load(fin_seffs_path)[:max_iter]

fin_ref_iters_cs = onp.load(fin_ref_iters_cs_path)
fin_ref_cs = onp.load(fin_ref_cs_path)

fin_ref_iters_seffs = onp.load(fin_ref_iters_seffs_path)
fin_ref_seffs = onp.load(fin_ref_seffs_path)


keep_ref_iters_cs = fin_ref_iters_cs < max_iter
keep_ref_iters_seffs = fin_ref_iters_seffs < max_iter

fin_ref_iters_cs = fin_ref_iters_cs[keep_ref_iters_cs]
fin_ref_cs = fin_ref_cs[keep_ref_iters_cs]

fin_ref_iters_seffs = fin_ref_iters_seffs[keep_ref_iters_seffs]
fin_ref_seffs = fin_ref_seffs[keep_ref_iters_seffs]


for width, height in [(20, 14), (24, 14), (28, 14)]:
# for width, height in [(20, 14)]:

    # fig, ax = plt.subplots(figsize=(width, height))
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[1, 1], figsize=(width, height))

    ax1.plot(fin_cs, color="black", linewidth=3)
    ax1.scatter(fin_ref_iters_cs, fin_ref_cs, color="black", s=100)

    ax1.fill_between(onp.arange(len(fin_cs)), 436-16, 436+16, color='green', alpha=0.3, label="Experimental Uncertainty", transform=ax1.get_yaxis_transform())

    ax2.plot(fin_seffs, color="black", linewidth=3)
    ax2.scatter(fin_ref_iters_seffs, fin_ref_seffs, color="black", s=100)
    ax2.fill_between(onp.arange(len(fin_seffs)), 1045-92, 1045+92, color='green', alpha=0.3, label="Experimental Uncertainty", transform=ax2.get_yaxis_transform())

    ax2.set_xlabel("Iteration")
    ax1.set_ylabel('c ($pN·nm^2$)', usetex=False)
    ax2.set_ylabel('$\mathrm{S_{eff}}$ ($pN$)', usetex=False)
    # ax.set_ylabel('c (pN·nm2)')

    # y_vals = [10.5, 11, 11.5, 12]
    # ax.set_yticks(y_vals)
    # ax.set_yticklabels([r'$10.5$', r'$11$', r'$11.5$', r'$12$'])

    # x_vals = [0, 25, 50]
    # ax.set_xticks(x_vals)
    # ax.set_xticklabels([r'$0$', r'$25$', r'$50$'])
    # ax.set_ylim(top=45)

    leg = ax1.legend(prop={'size': 48})

    # plt.show()
    # plt.savefig(output_dir / f"c_seff_{width}x{height}.pdf")
    plt.savefig(output_dir / f"c_seff_{width}x{height}.svg")
