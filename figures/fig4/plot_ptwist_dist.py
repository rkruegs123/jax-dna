import numpy as onp
from pathlib import Path
import pdb
from scipy.stats import norm
import seaborn as sns

from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from figures.utils import colors, labels


font = {'size': 48}
rc('font', **font)
rc('text', usetex=True)


data_dir = Path("figures/fig4/data")
ptwist_dist_dir = data_dir / "ptwist-dist"
output_dir = Path("figures/fig4/output")


# Panel A

target_mean = 21.7
target_var = 5.5
target_std = onp.sqrt(target_var)

width = 15
height = 10
fig, ax = plt.subplots(figsize=(width, height))

init_mean = 21.7
init_var = 1.5
init_std = onp.sqrt(init_var)

xs_init = onp.linspace(init_mean-7*init_std, init_mean+7*init_std, 1000)
ys_init = norm.pdf(xs_init, init_mean, init_std)
ax.plot(xs_init, ys_init, label=f"Init", color="black", linewidth=5)


xs_target = onp.linspace(target_mean-4*target_std, target_mean+4*target_std, 1000)
ys_target = norm.pdf(xs_target, target_mean, target_std)
ax.plot(xs_target, ys_target, label=f"Target", color="green", linewidth=5)


ax.set_ylabel("Probability Density")
ax.set_xlabel("Propeller Twist (deg)")

leg = ax.legend()
for line in leg.get_lines():
    line.set_linewidth(5.0)

plt.tight_layout()
# plt.show()
plt.savefig(output_dir / f"ptwist_dist_opt_overview.pdf")
plt.close()




# Panel B
font = {'size': 36}
rc('font', **font)

fin_means_path = ptwist_dist_dir / "obj/fin_means.npy"
fin_means = onp.load(fin_means_path)[1:]

fin_ref_means_path = ptwist_dist_dir / "obj/fin_ref_means.npy"
fin_ref_means = onp.load(fin_ref_means_path)[1:]

fin_vars_path = ptwist_dist_dir / "obj/fin_vars.npy"
fin_vars = onp.load(fin_vars_path)[1:]

fin_ref_vars_path = ptwist_dist_dir / "obj/fin_ref_vars.npy"
fin_ref_vars = onp.load(fin_ref_vars_path)[1:]

fin_ref_iters_path = ptwist_dist_dir / "obj/fin_ref_iters.npy"
fin_ref_iters = onp.load(fin_ref_iters_path)[1:]

kl_divergence_path = ptwist_dist_dir / "log/loss.txt"
kl_divergences = onp.loadtxt(kl_divergence_path)[1:]

width = 15
height = 10
# fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[1, 1], figsize=(width, height))
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, height_ratios=[1, 1, 1], figsize=(width, height))

ax1.plot(fin_means, color="black", linewidth=3)
ax1.scatter(fin_ref_iters, fin_ref_means, color="black", s=100)
ax1.axhline(y=target_mean, linewidth=3, linestyle="--", color="red")

y_vals = [21, 25]
ax1.set_yticks(y_vals)
ax1.set_yticklabels([r'$21$', r'$25$'])

ax2.plot(fin_vars, color="black", linewidth=3)
ax2.scatter(fin_ref_iters, fin_ref_vars, color="black", s=100)
ax2.axhline(y=target_var, linewidth=3, linestyle="--", color="red")

# ax2.set_xlabel("Iteration")
ax3.set_xlabel("Iteration")
ax1.set_ylabel('Mean', usetex=False)
ax2.set_ylabel('Variance', usetex=False)


ax3.plot(kl_divergences, color="black", linewidth=3)
ax3.scatter(fin_ref_iters, kl_divergences[fin_ref_iters], color="black", s=100)
ax3.set_ylabel('KL Div.', usetex=False)


# leg = ax1.legend(prop={'size': 48})
# leg = ax1.legend()

plt.tight_layout()
# plt.show()
plt.savefig(output_dir / f"ptwist_dist_opt_iters.pdf")
plt.close()


# Panel C
font = {'size': 54}
rc('font', **font)

iter1_ptwists = onp.load(ptwist_dist_dir / "ref_traj" / "iter1" / "ptwists.npy")
iter247_ptwists = onp.load(ptwist_dist_dir / "ref_traj" / "iter247" / "ptwists.npy")

width = 15
height = 10
fig, ax = plt.subplots(figsize=(width, height))

sns.histplot(iter1_ptwists, label="Initial", color="black", stat="density")
sns.histplot(iter247_ptwists, label="Final", color="green", stat="density")

ax.set_ylabel("Probability Density")
ax.set_xlabel("Propeller Twist (deg)")

leg = ax.legend()
for line in leg.get_lines():
    line.set_linewidth(5.0)

plt.tight_layout()

# plt.show()
plt.savefig(output_dir / f"ptwist_dist_opt_hist.pdf")
plt.close()
