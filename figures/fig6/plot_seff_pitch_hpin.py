import numpy as onp
from pathlib import Path
import pdb

from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from figures.utils import colors, labels


font = {'size': 36}
rc('font', **font)
rc('text', usetex=True)


data_dir = Path("figures/fig6/data")
joint_dir = data_dir / "seff-pitch-hpin"
output_dir = Path("figures/fig6/output")

max_iter = 60

# for width, height in [(20, 14), (24, 14), (28, 14)]:
for width, height in [(20, 14)]:

    # fig, ax = plt.subplots(figsize=(width, height))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, height_ratios=[1, 1, 1], figsize=(width, height), sharex=True)


    fin_seffs_path = joint_dir / "obj/seffs_i90.npy"
    fin_seffs = onp.load(fin_seffs_path)[:max_iter]

    fin_tms_path = joint_dir / "obj/tms_i90.npy"
    fin_tms = onp.load(fin_tms_path)[:max_iter]

    fin_pitches_path = joint_dir / "obj/pitches_i90.npy"
    fin_pitches = onp.load(fin_pitches_path)[:max_iter]

    fin_ref_iters_path = joint_dir / "obj/ref_iters_i90.npy"
    fin_ref_iters = onp.load(fin_ref_iters_path)

    fin_ref_seffs_path = joint_dir / "obj/ref_seffs_i90.npy"
    fin_ref_seffs = onp.load(fin_ref_seffs_path)

    fin_ref_tms_path = joint_dir / "obj/ref_tms_i90.npy"
    fin_ref_tms = onp.load(fin_ref_tms_path)

    fin_ref_pitches_path = joint_dir / "obj/ref_pitches_i90.npy"
    fin_ref_pitches = onp.load(fin_ref_pitches_path)

    keep_ref_iters = fin_ref_iters < max_iter
    fin_ref_iters = fin_ref_iters[keep_ref_iters]
    fin_ref_seffs = fin_ref_seffs[keep_ref_iters]
    fin_ref_tms = fin_ref_tms[keep_ref_iters]
    fin_ref_pitches = fin_ref_pitches[keep_ref_iters]

    ax1.plot(fin_seffs, color="black", linewidth=3)
    ax1.scatter(fin_ref_iters, fin_ref_seffs, color="black", s=100)
    ax1.fill_between(onp.arange(len(fin_seffs)), 1045-92, 1045+92, color='green', alpha=0.3, label="Uncertainty", transform=ax1.get_yaxis_transform())

    ax2.plot(fin_tms, color="black", linewidth=3)
    ax2.scatter(fin_ref_iters, fin_ref_tms, color="black", s=100)
    ax2.fill_between(onp.arange(len(fin_tms)), 334.5-0.5, 334.5+0.5, color='green', alpha=0.3, label="Uncertainty", transform=ax2.get_yaxis_transform())

    ax3.plot(fin_pitches, color="black", linewidth=3)
    ax3.scatter(fin_ref_iters, fin_ref_pitches, color="black", s=100)
    ax3.fill_between(onp.arange(len(fin_pitches)), 10.0, 10.5, color='green', alpha=0.3, label="Uncertainty", transform=ax3.get_yaxis_transform())





    ax3.set_xlabel("Iteration")
    ax1.set_ylabel('$\mathrm{S_{eff}}$ ($pN$)', usetex=False)
    ax2.set_ylabel('$\mathrm{T_m}$ (K)', usetex=False)
    ax3.set_ylabel('Pitch (bp/turn)', usetex=False)
    # ax.set_ylabel('c (pN·nm2)')

    y_vals = [10.0, 10.5, 11]
    ax3.set_yticks(y_vals)
    ax3.set_yticklabels([r'$10$', r'$10.5$', r'$11$'])

    # x_vals = [0, 25, 50]
    # ax.set_xticks(x_vals)
    # ax.set_xticklabels([r'$0$', r'$25$', r'$50$'])
    # ax.set_ylim(top=45)

    leg = ax1.legend(prop={'size': 36})

    # plt.show()
    plt.savefig(output_dir / f"joint_seff_pitch_hpin_{width}x{height}.pdf")

    plt.close()
