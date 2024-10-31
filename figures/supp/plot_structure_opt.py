import numpy as onp
from pathlib import Path
import pdb

from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from figures.utils import colors, labels


font = {'size': 76}
rc('font', **font)
rc('text', usetex=True)


data_dir = Path("figures/supp/data")
max_update_iters = 3
structure_dir = data_dir / "structure/burns_natnano_2015" / f"m{max_update_iters}"
output_dir = Path("figures/supp/output")


# for width, height in [(20, 14), (24, 14), (28, 14)]:
for width, height in [(20, 12), (20, 14), (20, 16)] :

    fig, ax = plt.subplots(figsize=(width, height))

    fin_rmsds_path = structure_dir / "obj/rmses_i400.npy"
    fin_rmsds = onp.load(fin_rmsds_path)

    fin_ref_iters_path = structure_dir / "obj/ref_iters_i400.npy"
    fin_ref_iters = onp.load(fin_ref_iters_path)

    fin_ref_rmsds_path = structure_dir / "obj/ref_rmses_i400.npy"
    fin_ref_rmsds = onp.load(fin_ref_rmsds_path)

    ax.plot(fin_rmsds, color="black", linewidth=6)
    ax.scatter(fin_ref_iters, fin_ref_rmsds, color="black", s=300)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSD (oxDNA units)")

    y_vals = [0.8, 1.0, 1.2]
    ax.set_yticks(y_vals)
    ax.set_yticklabels([r'$0.8$', r'$1.0$', r'$1.2$'])

    x_vals = [0, 200, 400]
    ax.set_xticks(x_vals)
    ax.set_xticklabels([r'$0$', r'$200$', r'$400$'])

    plt.tight_layout()

    # plt.show()
    plt.savefig(output_dir / f"rmsds_losses_{width}x{height}.pdf")

    plt.close()


# Load the gradients from the first iteration
grads_path = structure_dir / "log"/ "grads.txt"

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
# m = 10
m = 5
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
        "a_stack_5": r'$a_{stack,5}$',
        "theta0_stack_5": r'$\theta^0_{stack,5}$',
        "theta0_stack_4": r'$\theta^0_{stack,4}$'
    },
    "hydrogen_bonding": {
        "dr0_hb": r'$\delta r^0_{hb}$',
        "theta0_hb_4": r'$\theta^0_{hb,4}$',
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
    legend_colors.append(f"{colors[key1]}")

    if labels[key1] in legend_labels:
        legend_labels.append(f"_{labels[key1]}")
    else:
        legend_labels.append(labels[key1])



font = {'size': 48}
rc('font', **font)
rc('text', usetex=True)

# for width, height in [(20, 8), (28, 8), (32, 8)]:
# for width, height in [(20, 14), (15, 8), (20, 8)]:
for width, height in [(20, 14), (20, 10)]:
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_ylabel(r'$|\nabla_{\theta}\mathcal{L}|$')

    # y_vals = [0, 5, 10, 15, 20]
    # ax.set_yticks(y_vals)
    # ax.set_yticklabels([r'$0$', r'$5$', r'$10$', r'$15$', r'$20$'])

    ax.set_xlabel("Parameter")
    ax.bar(plot_xlabels, bar_heights, label=legend_labels, color=legend_colors)
    ax.legend(title="Interaction")
    plt.tight_layout()

    # plt.show()
    plt.savefig(output_dir / f"rmsd_top{m}_grads_{width}x{height}.pdf")

    plt.close()
