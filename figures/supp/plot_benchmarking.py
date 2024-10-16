import numpy as onp
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc
import pdb
import json


rc('text', usetex=True)
plt.rcParams.update({'font.size': 36})

output_dir = Path("figures/supp/output")

data_dir = Path("figures/supp/data")


# Plot just forward simulations
sim_only_fpath = data_dir / "benchmarking_results_no_grad_a100_50k.json"

with open(sim_only_fpath) as f:
    sim_data = json.load(f)



# Extract relevant data for plotting
categories = ['8 BP, CPU', '500 BP, GPU']
labels = ['JAX-MD', 'oxDNA']
# colors = ['#1f77b4', '#ff7f0e']
colors = ['red', 'blue']

# Data for "8 BP, CPU"
data_8bp_cpu = [
    sim_data["8bp-cpu-jaxdna-no-False-0.2"],
    sim_data["8bp-cpu-oxdna-no-False-0.2"]
]

# Data for "500 BP, GPU"
data_500bp_gpu = [
    sim_data["500bp-cuda-jaxdna-verlet-False-0.2"],
    sim_data["500bp-cuda-oxdna-verlet-False-0.2"]
]

# Combine all data for easier plotting
means = [data[0] for data in data_8bp_cpu] + [data[0] for data in data_500bp_gpu]
stds = [data[1] for data in data_8bp_cpu] + [data[1] for data in data_500bp_gpu]

# Grouping for each category
n_groups = 2  # Two categories: 8 BP, CPU and 500 BP, GPU
n_bars = 2  # Two bars per group: JAX-MD and oxDNA

fig, ax = plt.subplots(figsize=(12, 8))

index = onp.arange(n_groups)  # Positions for the two main categories
bar_width = 0.35  # Width of the bars

# Plot bars for JAX-MD and oxDNA
rects1 = ax.bar(index, means[0::2], bar_width, yerr=stds[0::2], label='JAX-MD', color=colors[0])
rects2 = ax.bar(index + bar_width, means[1::2], bar_width, yerr=stds[1::2], label='oxDNA', color=colors[1])

"""
error_bar_settings = {'capsize': 10, 'elinewidth': 3, 'capthick': 3}
rects1 = ax.bar(index, means[0::2], bar_width, yerr=stds[0::2], label='JAX-MD',
                color=colors[0], error_kw=error_bar_settings)
rects2 = ax.bar(index + bar_width, means[1::2], bar_width, yerr=stds[1::2], label='oxDNA',
                color=colors[1], error_kw=error_bar_settings)
"""


# Labels, title, and axes formatting
ax.set_xlabel('Benchmark')
ax.set_ylabel('Time (s)')
ax.set_title('Forward Simulation Benchmarking')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(categories)
ax.legend()

# Display the plot
plt.tight_layout()
# plt.show()
plt.savefig(output_dir / "forward_sim_benchmarking.pdf")
plt.close()




# Load data for Forward and Gradient
with open(data_dir / "benchmarking_results_ckpt0_2k.json") as f:
    sim_data_no_ckpt = json.load(f)

# Load data for Gradient + Checkpointing
with open(data_dir / "benchmarking_results_ckpt100_2k.json") as f:
    sim_data_ckpt = json.load(f)

# Extract relevant data for plotting
categories = ['8 BP, CPU', '8 BP, GPU']
labels = ['Forward', 'Gradient', 'Gradient + Checkpointing']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
colors = ['red', 'blue', 'green']

# Data for "8 BP, CPU"
data_8bp_cpu = [
    sim_data_no_ckpt["8bp-cpu-jaxdna-no-False-0.2"],  # Forward
    sim_data_no_ckpt["8bp-cpu-jaxdna-no-True-0.2"],   # Gradient
    sim_data_ckpt["8bp-cpu-jaxdna-no-True-0.2"]       # Gradient + Checkpointing
]

# Data for "8 BP, GPU"
data_8bp_gpu = [
    sim_data_no_ckpt["8bp-cuda-jaxdna-no-False-0.2"],  # Forward
    sim_data_no_ckpt["8bp-cuda-jaxdna-no-True-0.2"],   # Gradient
    sim_data_ckpt["8bp-cuda-jaxdna-no-True-0.2"]       # Gradient + Checkpointing
]

# Combine all data for easier plotting
means = [data[0] for data in data_8bp_cpu] + [data[0] for data in data_8bp_gpu]
stds = [data[1] for data in data_8bp_cpu] + [data[1] for data in data_8bp_gpu]

# Grouping for each category
n_groups = 2  # Two categories: 8 BP, CPU and 8 BP, GPU
n_bars = 3    # Three bars per group: Forward, Gradient, Gradient + Checkpointing

fig, ax = plt.subplots(figsize=(12, 8))

index = onp.arange(n_groups)  # Positions for the two main categories
bar_width = 0.25  # Width of the bars

# Plot bars for Forward, Gradient, and Gradient + Checkpointing
rects1 = ax.bar(index, means[0::3], bar_width, yerr=stds[0::3], label='Forward', color=colors[0])
rects2 = ax.bar(index + bar_width, means[1::3], bar_width, yerr=stds[1::3], label='Gradient', color=colors[1])
rects3 = ax.bar(index + 2 * bar_width, means[2::3], bar_width, yerr=stds[2::3], label='Gradient Remat.', color=colors[2])

# Labels, title, and axes formatting
ax.set_xlabel('Benchmark')
ax.set_ylabel('Time (s)')
ax.set_title('Gradient Calculation Benchmarking')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(categories)
ax.legend()

# Display the plot
plt.tight_layout()

# plt.show()
plt.savefig(output_dir / "gradient_calc_benchmarking.pdf")
plt.close()
