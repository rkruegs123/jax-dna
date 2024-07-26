import numpy as onp
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc
import pdb

rc('text', usetex=True)
plt.rcParams.update({'font.size': 24})

data_dir = Path("figures/fig2/data")
time_data_dir = data_dir / "time-scaling"

length_path = time_data_dir / "sim_length.txt"
lengths = onp.loadtxt(length_path)

sim_time_path = time_data_dir / "sim_time_ckpt.txt"
sim_times = onp.loadtxt(sim_time_path)

grad_time_path = time_data_dir / "grad_time_ckpt.txt"
grad_times = onp.loadtxt(grad_time_path)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(lengths, sim_times, label="Forward Simulation")
ax.plot(lengths, grad_times, label="Gradient")
ax.set_xlabel("Simulation Length")
ax.set_ylabel("Time (s)")
ax.legend()
plt.show()
