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

fig, ax = plt.subplots(figsize=(20, 14))
for dir_name, name in [("fene", "fene"), ("stacking", "stacking"), ("hb", "hydrogen_bonding"), ("cr-stacking", "cross_stacking"), ("cx-stacking", "coaxial_stacking"), ("all", "all")]:

    fin_pitches_path = t11_dir / dir_name / "obj/fin_pitches.npy"
    fin_pitches = onp.load(fin_pitches_path)

    fin_ref_iters_path = t11_dir / dir_name / "obj/fin_ref_iters.npy"
    fin_ref_iters = onp.load(fin_ref_iters_path)

    fin_ref_pitches_path = t11_dir / dir_name / "obj/fin_ref_pitches.npy"
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
plt.show()



"""
the bars should be color coded based on the type of thing they come from
- can put the color map in the utilities
- should use the same colors in the loss plot
- all should be black

plot should be organized per the following:
- overview of grad estimator at top
- second row, 1/3 width: time per grad
- second row, 2/3 width: loss curve with a line per type of parameter
- third row: gradients/sensitivity analysis
"""
