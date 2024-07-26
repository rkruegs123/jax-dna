import numpy as onp
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
plt.rcParams.update({'font.size': 24})

data_dir = Path("figures/fig2/data")
grad_mag_data_dir = data_dir / "grad-mag-scaling"

length_path = grad_mag_data_dir / "length.txt"
lengths = onp.loadtxt(length_path)
mean_grad_abs_path = grad_mag_data_dir / "mean_grad_abs.txt"
mean_grad_abs = onp.loadtxt(mean_grad_abs_path)

fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(lengths, onp.log(mean_grad_abs), color="black")
ax.set_xlabel("Simulation Length")
# ax.set_ylabel(r"$\log \left( \langle |\nabla_{\theta}\mathcal{L}| \rangle \right)$")
# ax.set_ylabel(r"$\log \left( \langle |\nabla_{\theta}| \rangle \right)$")
ax.set_ylabel(r"$\log \langle |\nabla_{\theta}| \rangle $")
# ax.set_ylabel("Log (Mean Gradient Absolute Value)")
# ax.set_ylabel("Log (Mean |Gradient|)")
plt.show()
