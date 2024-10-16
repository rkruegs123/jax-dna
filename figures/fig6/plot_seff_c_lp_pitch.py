import numpy as onp
from pathlib import Path
import pdb

from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from figures.utils import colors, labels


font = {'size': 24}
rc('font', **font)
rc('text', usetex=True)


data_dir = Path("figures/fig6/data")
joint_dir = data_dir / "seff-c-lp-pitch"
output_dir = Path("figures/fig6/output")

max_iter = 60

# for width, height in [(20, 14), (24, 14), (28, 14)]:
for width, height in [(20, 14)]:

    # fig, ax = plt.subplots(figsize=(width, height))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, height_ratios=[1, 1, 1, 1], figsize=(width, height), sharex=True)


    fin_cs_path = joint_dir / "obj/cs_i100.npy"
    fin_cs = onp.load(fin_cs_path)[:max_iter]

    fin_seffs_path = joint_dir / "obj/seffs_i100.npy"
    fin_seffs = onp.load(fin_seffs_path)[:max_iter]

    fin_lps_path = joint_dir / "obj/lps_i100.npy"
    fin_lps = onp.load(fin_lps_path)[:max_iter]

    fin_pitches_path = joint_dir / "obj/pitches_i100.npy"
    fin_pitches = onp.load(fin_pitches_path)[:max_iter]

    fin_ref_iters_path = joint_dir / "obj/ref_iters_i100.npy"
    fin_ref_iters = onp.load(fin_ref_iters_path)

    fin_ref_cs_path = joint_dir / "obj/ref_cs_i100.npy"
    fin_ref_cs = onp.load(fin_ref_cs_path)

    fin_ref_seffs_path = joint_dir / "obj/ref_seffs_i100.npy"
    fin_ref_seffs = onp.load(fin_ref_seffs_path)

    fin_ref_lps_path = joint_dir / "obj/ref_lps_i100.npy"
    fin_ref_lps = onp.load(fin_ref_lps_path)

    fin_ref_pitches_path = joint_dir / "obj/ref_pitches_i100.npy"
    fin_ref_pitches = onp.load(fin_ref_pitches_path)

    keep_ref_iters = fin_ref_iters < max_iter
    fin_ref_iters = fin_ref_iters[keep_ref_iters]
    fin_ref_cs = fin_ref_cs[keep_ref_iters]
    fin_ref_seffs = fin_ref_seffs[keep_ref_iters]
    fin_ref_lps = fin_ref_lps[keep_ref_iters]
    fin_ref_pitches = fin_ref_pitches[keep_ref_iters]

    ax1.plot(fin_cs, color="black", linewidth=3)
    ax1.scatter(fin_ref_iters, fin_ref_cs, color="black", s=100)

    ax1.fill_between(onp.arange(len(fin_cs)), 436-16, 436+16, color='green', alpha=0.3, label="Uncertainty", transform=ax1.get_yaxis_transform())

    ax2.plot(fin_seffs, color="black", linewidth=3)
    ax2.scatter(fin_ref_iters, fin_ref_seffs, color="black", s=100)
    ax2.fill_between(onp.arange(len(fin_seffs)), 1045-92, 1045+92, color='green', alpha=0.3, label="Uncertainty", transform=ax2.get_yaxis_transform())

    ax3.plot(fin_lps, color="black", linewidth=3)
    ax3.scatter(fin_ref_iters, fin_ref_lps, color="black", s=100)
    ax3.fill_between(onp.arange(len(fin_lps)), 50-3, 50+3, color='green', alpha=0.3, label="Uncertainty", transform=ax3.get_yaxis_transform())

    ax4.plot(fin_pitches, color="black", linewidth=3)
    ax4.scatter(fin_ref_iters, fin_ref_pitches, color="black", s=100)
    ax4.fill_between(onp.arange(len(fin_pitches)), 10.0, 10.5, color='green', alpha=0.3, label="Uncertainty", transform=ax4.get_yaxis_transform())





    ax4.set_xlabel("Iteration")
    ax1.set_ylabel('c ($pN·nm^2$)', usetex=False)
    ax2.set_ylabel('$\mathrm{S_{eff}}$ ($pN$)', usetex=False)
    ax3.set_ylabel('$\mathrm{L_{ps}}$ ($nm$)', usetex=False)
    ax4.set_ylabel('Pitch (bp/turn)', usetex=False)
    # ax.set_ylabel('c (pN·nm2)')

    # y_vals = [10.5, 11, 11.5, 12]
    # ax.set_yticks(y_vals)
    # ax.set_yticklabels([r'$10.5$', r'$11$', r'$11.5$', r'$12$'])

    # x_vals = [0, 25, 50]
    # ax.set_xticks(x_vals)
    # ax.set_xticklabels([r'$0$', r'$25$', r'$50$'])
    # ax.set_ylim(top=45)

    leg = ax1.legend(prop={'size': 24})

    # plt.show()
    plt.savefig(output_dir / f"joint_seff_c_lp_pitch_{width}x{height}.pdf")
    plt.clf()
