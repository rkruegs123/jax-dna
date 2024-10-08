import numpy as onp
from pathlib import Path
import pdb

from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib

from figures.utils import colors, labels, label_dict


font = {'size': 36}
rc('font', **font)
rc('text', usetex=True)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


data_dir = Path("figures/fig4/data")
moduli_dir = data_dir / "moduli"
c_dir = moduli_dir / "c-all/"
seff_dir = moduli_dir / "seff-all/"
g_dir = moduli_dir / "g-all/"
output_dir = Path("figures/fig4/output")


m = 5
sizes = [(8, 8), (12, 8), (16, 8)]
# sizes = [(16, 8)]

for m_name, m_dir in [("seff", seff_dir), ("c", c_dir), ("g", g_dir)]:

    grads_path = m_dir / "log/grads.txt"
    assert(grads_path.exists())

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
    top_m_values = sorted_data[:m]

    # Plot
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

    for width, height in sizes:
        fig, ax = plt.subplots(figsize=(width, height))
        ax.set_ylabel(r'$|\nabla_{\theta}\mathcal{L}|$')

        # y_vals = [0, 50, 100, 150]
        # ax.set_yticks(y_vals)
        # ax.set_yticklabels([r'$0$', r'$50$', r'$100$', r'$150$'])

        ax.set_xlabel("Parameter")
        ax.bar(plot_xlabels, bar_heights, label=legend_labels, color=legend_colors)
        ax.legend(title="Interaction")
        plt.tight_layout()

        # plt.show()
        # plt.savefig(output_dir / f"{m_name}_top{m}_grads_{width}x{height}.pdf")
        plt.savefig(output_dir / f"{m_name}_top{m}_grads_{width}x{height}.svg")

        plt.close()
