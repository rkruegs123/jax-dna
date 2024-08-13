import numpy as onp
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc
import pdb

rc('text', usetex=True)
plt.rcParams.update({'font.size': 36})

output_dir = Path("figures/fig2/output")

target_pitch = 11.0

data_dir = Path("figures/fig2/data")
opt_dir = data_dir / "opt"

loss_path = opt_dir / "log" / "loss.txt"
losses = onp.loadtxt(loss_path)

pitch_path = opt_dir / "log" / "pitch.txt"
pitches = onp.loadtxt(pitch_path)

max_iters = 100
losses = losses[:max_iters]
pitches = pitches[:max_iters]
num_iters = len(losses)
iters = onp.arange(num_iters)

fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(iters, losses, color="black")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
plt.tight_layout()
# plt.show()
plt.savefig(output_dir / "loss.pdf")
plt.close()


fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(iters, pitches, color="black")
ax.axhline(y=target_pitch, linestyle="--", color="red", label="Target Pitch", linewidth=2)

ax.set_xlabel("Iteration")
ax.set_ylabel("Pitch (bp/turn)")

y_vals = [10.75, 11.0]
ax.set_yticks(y_vals)
ax.set_yticklabels([r'$10.75$', r'$11$'])
ax.set_ylim(ymin=10.7, ymax=11.1)

ax.legend(loc="upper right")
# ax.legend(bbox_to_anchor=[1.0, 0.95])
plt.tight_layout()


# plt.show()
plt.savefig(output_dir / "pitch.pdf")

plt.close()
