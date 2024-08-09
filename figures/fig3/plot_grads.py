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


data_dir = Path("figures/fig3/data")
all_opt_dir = data_dir / "pitch-opts/target11/all/"
grads_path = all_opt_dir / "log/grads.txt"

with open(grads_path) as gf:
    grads_lines = [line.rstrip() for line in gf]

iter_lines = list()
curr_iter_lines = list()
for gline in grads_lines:
    if not gline:
        continue
    if gline[:9] == "Iteration" and curr_iter_lines:
        iter_lines.append(curr_iter_lines)
        curr_iter_lines = list()
    else:
        curr_iter_lines.append(gline)
iter_lines.append(curr_iter_lines)

print(f"Found gradients for {len(iter_lines)} iterations.")

iter0_glines = iter_lines[0]

# Read into dictionary
iter0_grads = dict()
curr_key = None
for gline in iter0_glines[1:]:
    if gline[0] == "-":
        key = gline[2:]
        assert(key not in iter0_grads)
        iter0_grads[key] = dict()
        curr_key = key
    elif gline[0] == "\t":
        pkey, pval = gline.strip()[2:].split(':')
        iter0_grads[curr_key][pkey] = abs(float(pval.strip()))
    else:
        raise RuntimeError(f"Unrecognized line format. Stopping in case of bug.")


# Sort and extract
flattened_data = [(key1, key2, val) for key1, sub_dict in iter0_grads.items() for key2, val in sub_dict.items()]
sorted_data = sorted(flattened_data, key=lambda x: x[2], reverse=True)
m = 10
top_m_values = sorted_data[:m]

# Plot
label_dict = {
    "fene": {
        "r0_backbone": r'$\delta r^0_{backbone}$',
        "delta_backbone": r'$\Delta_{backbone}$',

    },
    "cross_stacking": {
        "r0_cross": r'$\delta r^0_{cross}$',
        "dr_c_cross": r'$\delta r^c_{cross}$',
        "theta0_cross_1": r'$\theta^0_{cross,1}$',
        "theta0_cross_4": r'$\theta^0_{cross,4}$'
    },
    "stacking": {
        "dr0_stack": r'$\delta r^0_{stack}$',
        "theta0_stack_5": r'$\theta^0_{stack,5}$'
    },
    "hydrogen_bonding": {
        "dr0_hb": r'$\delta r^0_{hb}$',
        "theta0_hb_7": r'$\theta^0_{hb,7}$'
    }
}
plot_xlabels = list()
bar_heights = list()
legend_colors = list()
legend_labels = list()
for key1, key2, val in top_m_values:
    bar_heights.append(val)
    plot_xlabels.append(label_dict[key1][key2])
    # legend_colors.append(f"tab:{colors[key1]}")
    legend_colors.append(f"{colors[key1]}")

    if labels[key1] in legend_labels:
        legend_labels.append(f"_{labels[key1]}")
    else:
        legend_labels.append(labels[key1])

fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylabel(r'$|\nabla_{\theta}\mathcal{L}|$')
y_vals = [0, 5, 10, 15, 20]
ax.set_yticks(y_vals)
ax.set_yticklabels([r'$0$', r'$5$', r'$10$', r'$15$', r'$20$'])
ax.set_xlabel("Parameter")
ax.bar(plot_xlabels, bar_heights, label=legend_labels, color=legend_colors)
ax.legend(title="Interaction")
plt.tight_layout()
plt.show()
