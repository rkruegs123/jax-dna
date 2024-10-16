import numpy as onp
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
plt.rcParams.update({'font.size': 36})

output_dir = Path("figures/fig2/output")

data_dir = Path("figures/fig2/data")
grad_mag_data_dir = data_dir / "grad-mag-scaling"

length_path = grad_mag_data_dir / "length.txt"
lengths = onp.loadtxt(length_path)
mean_grad_abs_path = grad_mag_data_dir / "mean_grad_abs.txt"
mean_grad_abs = onp.loadtxt(mean_grad_abs_path)

fig, ax = plt.subplots(figsize=(12, 10))

ax.plot(lengths, onp.log(mean_grad_abs), color="black", linewidth=2)
ax.set_xlabel("Simulation Steps")
# ax.set_ylabel(r"$\log \left( \langle |\nabla_{\theta}\mathcal{L}| \rangle \right)$")
# ax.set_ylabel(r"$\log \left( \langle |\nabla_{\theta}| \rangle \right)$")
ax.set_ylabel(r"$\log \langle |\nabla_{\theta}| \rangle $")
# ax.set_ylabel("Log (Mean Gradient Absolute Value)")
# ax.set_ylabel("Log (Mean |Gradient|)")
plt.tight_layout()

# plt.show()
plt.savefig(output_dir / "grad_mag_scaling.pdf")
