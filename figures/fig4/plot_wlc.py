import numpy as onp
from pathlib import Path
import pdb

from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from figures.utils import colors, labels
from jax_dna.common.utils import nm_per_oxdna_length, oxdna_force_to_pn


font = {'size': 48}
rc('font', **font)
rc('text', usetex=True)


data_dir = Path("figures/fig4/data")
t25_dir = data_dir / "wlc/ext-mod-25"
output_dir = Path("figures/fig4/output")

max_iter = 100

# for width, height in [(20, 14), (24, 14), (28, 14)]:
for width, height in [(20, 14)]:

    # fig, ax = plt.subplots(figsize=(width, height))
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[1, 1], figsize=(width, height))


    lps_path = t25_dir / "obj/lps_i90.npy"
    lps = onp.load(lps_path)[:max_iter] * nm_per_oxdna_length

    ext_mods_path = t25_dir / "obj/ext_mods_i90.npy"
    ext_mods = onp.load(ext_mods_path)[:max_iter] * oxdna_force_to_pn

    ref_iters_path = t25_dir / "obj/ref_iters_i90.npy"
    ref_iters = onp.load(ref_iters_path)


    ref_lps_path = t25_dir / "obj/ref_lps_i90.npy"
    ref_lps = onp.load(ref_lps_path) * nm_per_oxdna_length

    ref_ext_mods_path = t25_dir / "obj/ref_ext_mods_i90.npy"
    ref_ext_mods = onp.load(ref_ext_mods_path) * oxdna_force_to_pn


    keep_ref_iters = ref_iters < max_iter
    ref_iters = ref_iters[keep_ref_iters]
    ref_lps = ref_lps[keep_ref_iters]
    ref_ext_mods = ref_ext_mods[keep_ref_iters]

    ext_mods_to_plot = list()
    xs_to_plot = list()
    start_bad_region_idx = None
    for i in range(len(ext_mods)):
        ext_mod = ext_mods[i]
        if ext_mod < 0:

            if start_bad_region_idx is None:
                start_bad_region_idx = i

        else:


            if start_bad_region_idx is not None:

                ax1.plot(xs_to_plot, ext_mods_to_plot, color="black", linewidth=3)

                ax1.plot([start_bad_region_idx-1, i], [ext_mods_to_plot[-1], ext_mod], color="black", linewidth=2, linestyle="--")
                start_bad_region_idx = None

                ext_mods_to_plot = list()
                xs_to_plot = list()

            ext_mods_to_plot.append(ext_mod)
            xs_to_plot.append(i)
    ax1.plot(xs_to_plot, ext_mods_to_plot, color="black", linewidth=3)



    ref_xs_to_plot = list()
    ref_ext_mods_to_plot = list()
    for i in range(len(ref_ext_mods)):
        ext_mod = ref_ext_mods[i]
        if ext_mod < 0:
            pass
        else:
            ref_ext_mods_to_plot.append(ext_mod)
            ref_xs_to_plot.append(ref_iters[i])


    ax1.scatter(ref_xs_to_plot, ref_ext_mods_to_plot, color="black", s=100)
    ax1.axhline(y=25*oxdna_force_to_pn, label="Target", linestyle="--", color="red", linewidth=3)

    ax2.plot(lps, color="black", linewidth=3)
    ax2.scatter(ref_iters, ref_lps, color="black", s=100)

    ax2.set_xlabel("Iteration")
    ax1.set_ylabel('$\mathrm{S}$ ($\mathrm{pN}$)', usetex=False)
    ax2.set_ylabel('$\mathrm{L_{ps}}$ ($\mathrm{nm}$)', usetex=False)

    # y_vals = [10.5, 11, 11.5, 12]
    # ax.set_yticks(y_vals)
    # ax.set_yticklabels([r'$10.5$', r'$11$', r'$11.5$', r'$12$'])

    # x_vals = [0, 25, 50]
    # ax.set_xticks(x_vals)
    # ax.set_xticklabels([r'$0$', r'$25$', r'$50$'])
    # ax.set_ylim(top=45)

    leg = ax1.legend(prop={'size': 48})

    # plt.show()
    plt.savefig(output_dir / f"wlc_{width}x{height}.pdf")
    plt.savefig(output_dir / f"wlc_{width}x{height}.svg")
    plt.close()


# Plot change in WLC
iter0_extensions = onp.array([34.68624333, 36.5369262, 37.79959167, 38.34516158, 38.7016258, 38.9616109, 39.1611338, 39.42964732]) * nm_per_oxdna_length # nm
iter92_extensions = onp.array([32.27539561, 34.13732982, 35.5014128, 36.11890343, 36.59914181, 36.82990479, 37.14445323, 37.46313616]) * nm_per_oxdna_length # nm

PER_NUC_FORCES = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.375]
TOTAL_FORCES = onp.array(PER_NUC_FORCES) * 2.0
TOTAL_FORCES_SI = TOTAL_FORCES * oxdna_force_to_pn # pN

for width, height in [(14, 12)]:
    fig, ax = plt.subplots(figsize=(width, height))

    ax.plot(iter0_extensions, TOTAL_FORCES_SI, linewidth=4, color="black", label="Initial")
    ax.scatter(iter0_extensions, TOTAL_FORCES_SI, s=200, color="black")

    ax.plot(iter92_extensions, TOTAL_FORCES_SI, linewidth=4, color="green", label="Optimized")
    ax.scatter(iter92_extensions, TOTAL_FORCES_SI, s=200, color="green")


    ax.set_xlabel("Extension (nm)")
    ax.set_ylabel("Tension / pN")
    leg = ax.legend(prop={'size': 48})
    plt.tight_layout()

    # plt.show()
    plt.savefig(output_dir / f"wlc_force_ext_change_{width}x{height}.pdf")
    plt.close()
