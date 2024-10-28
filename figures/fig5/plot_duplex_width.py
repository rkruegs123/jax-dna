import numpy as onp
from pathlib import Path
import pdb

from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import jax.numpy as jnp

from figures.utils import colors, labels


font = {'size': 48}
rc('font', **font)
rc('text', usetex=True)


data_dir = Path("figures/fig5/data")


target = 8
width_dir = data_dir / "duplex" / f"oxdna1" / f"w{target}"
max_iter = 250


assert(width_dir.exists())
output_dir = Path("figures/fig5/output")


fin_tms_path = width_dir / f"obj/fin_tms.npy"
fin_tms = onp.load(fin_tms_path)[:max_iter]

fin_widths_path = width_dir / f"obj/fin_widths.npy"
fin_widths = onp.load(fin_widths_path)[:max_iter]

fin_ref_iters_path = width_dir / f"obj/fin_ref_iters.npy"
fin_ref_iters = onp.load(fin_ref_iters_path)

fin_ref_tms_path = width_dir / f"obj/fin_ref_tms.npy"
fin_ref_tms = onp.load(fin_ref_tms_path)

fin_ref_widths_path = width_dir / f"obj/fin_ref_widths.npy"
fin_ref_widths = onp.load(fin_ref_widths_path)

keep_ref_iters = fin_ref_iters < max_iter
fin_ref_iters = fin_ref_iters[keep_ref_iters]
fin_ref_tms = fin_ref_tms[keep_ref_iters]
fin_ref_widths = fin_ref_widths[keep_ref_iters]


for width, height in [(20, 14), (24, 14), (28, 14)]:
# for width, height in [(28, 14)]:
# for width, height in []:

    fig, ax = plt.subplots(figsize=(width, height))

    ax.plot(fin_widths, color="black", linewidth=3, label="Simulation")
    ax.scatter(fin_ref_iters, fin_ref_widths, color="black", s=100)

    ax.axhline(y=target, linewidth=3, linestyle="--", color="red", label="Target")

    # ax.fill_between(onp.arange(len(fin_gs)), -100, -80, color='green', alpha=0.3, label="Experimental Uncertainty", transform=ax.get_yaxis_transform())

    ax.set_xlabel("Iteration")
    ax.set_ylabel('Width (K)', usetex=False)

    # y_vals = [10.5, 11, 11.5, 12]
    # ax.set_yticks(y_vals)
    # ax.set_yticklabels([r'$10.5$', r'$11$', r'$11.5$', r'$12$'])

    # x_vals = [0, 25, 50]
    # ax.set_xticks(x_vals)
    # ax.set_xticklabels([r'$0$', r'$25$', r'$50$'])
    # ax.set_ylim(top=45)

    leg = ax.legend(prop={'size': 48})
    for line in leg.get_lines():
        line.set_linewidth(5.0)

    # plt.show()
    plt.savefig(output_dir / f"width_duplex_w{target}_{width}x{height}.pdf")

    plt.close()



# Plot inset
first_ref_iter = 0
first_ref_iter_dir = width_dir / "ref_traj" / f"iter{first_ref_iter}"
init_finfs = onp.load(first_ref_iter_dir / "melting_finfs.npy")
init_temps = onp.load(first_ref_iter_dir / "melting_temps_discrete.npy")

init_extrap_finfs = onp.linspace(0.1, 0.95, 100)
init_extrap_temps = jnp.interp(init_extrap_finfs, init_finfs, init_temps)

last_ref_iter = fin_ref_iters[-1]
last_ref_iter_dir = width_dir / "ref_traj" / f"iter{last_ref_iter}"
last_finfs = onp.load(last_ref_iter_dir / "melting_finfs.npy")
last_temps = onp.load(last_ref_iter_dir / "melting_temps_discrete.npy")

last_extrap_finfs = onp.linspace(0.1, 0.95, 100)
last_extrap_temps = jnp.interp(last_extrap_finfs, last_finfs, last_temps)


fig, ax = plt.subplots(figsize=(12, 10))
# ax.plot(init_temps, init_finfs, color="black", linewidth=3, label="Initial")
ax.plot(init_extrap_temps, init_extrap_finfs, color="black", linewidth=3, label="Initial")
# ax.plot(last_temps, last_finfs, color="green", linewidth=3, label="Optimized")

pdb.set_trace()
ax.plot(last_extrap_temps, last_extrap_finfs, color="green", linewidth=3, label="Optimized")

ax.set_ylabel("Duplex Yield")
ax.set_xlabel("T / K")

ax.set_yticks([0.0, 0.5, 1.0])
ax.set_yticklabels([r'$0.0$', r'$0.5$', r'$1.0$'])

# ax.set_xticks([320, 330, 340, 350])
# ax.set_xticklabels([r'$320$', r'$330$', r'$340$', r'$350$'])

ax.tick_params(width=3, size=10)

leg = ax.legend(prop={'size': 48})
for line in leg.get_lines():
    line.set_linewidth(5.0)

plt.tight_layout()
plt.show()
