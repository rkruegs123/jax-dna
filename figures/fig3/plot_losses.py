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


data_dir = Path("figures/fig3/data")
t11_dir = data_dir / "pitch-opts/target11/"
output_dir = Path("figures/fig3/output")

for optimizer in ["rmsprop", "adam"]:

    for width, height in [(20, 14), (24, 14), (28, 14)]:

        fig, ax = plt.subplots(figsize=(width, height))
        for dir_name, name in [("all", "all"), ("fene", "fene"), ("stacking", "stacking"), ("hb", "hydrogen_bonding"), ("cr-stacking", "cross_stacking"), ("cx-stacking", "coaxial_stacking")]:

            fin_pitches_path = t11_dir / optimizer / dir_name / "obj/fin_pitches.npy"
            fin_pitches = onp.load(fin_pitches_path)

            fin_ref_iters_path = t11_dir / optimizer / dir_name / "obj/fin_ref_iters.npy"
            fin_ref_iters = onp.load(fin_ref_iters_path)

            fin_ref_pitches_path = t11_dir / optimizer / dir_name / "obj/fin_ref_pitches.npy"
            fin_ref_pitches = onp.load(fin_ref_pitches_path)

            ax.plot(fin_pitches, color=colors[name], label=labels[name], linewidth=3)
            ax.scatter(fin_ref_iters, fin_ref_pitches, color=colors[name], s=100)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Pitch")

        y_vals = [10.5, 11, 11.5, 12]
        ax.set_yticks(y_vals)
        ax.set_yticklabels([r'$10.5$', r'$11$', r'$11.5$', r'$12$'])

        x_vals = [0, 50, 100, 150, 200]
        ax.set_xticks(x_vals)
        ax.set_xticklabels([r'$0$', r'$50$', r'$100$', r'$150$', r'$200$'])

        leg = ax.legend(title="Free Parameters", prop={'size': 35}, loc="upper left")
        # change the line width for the legend
        for line in leg.get_lines():
            line.set_linewidth(5.0)

        # plt.show()
        plt.savefig(output_dir / f"losses_{optimizer}_{width}x{height}.pdf")
