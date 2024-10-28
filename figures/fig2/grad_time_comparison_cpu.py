import numpy as onp
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc
import pdb
import json


rc('text', usetex=True)
plt.rcParams.update({'font.size': 36})

output_dir = Path("figures/fig2/output")

data_dir = Path("figures/supp/data") # Note: this data is stored as supplement data



# Load data for Forward and Gradient
with open(data_dir / "benchmarking_results_ckpt0_2k.json") as f:
    sim_data_no_ckpt = json.load(f)

# Load data for Gradient + Checkpointing
with open(data_dir / "benchmarking_results_ckpt100_2k.json") as f:
    sim_data_ckpt = json.load(f)

# Extract relevant data for plotting
categories = ['8 BP, CPU']
labels = ['Forward', 'Gradient', 'Gradient Remat.']
colors = ['red', 'blue', 'green']

# Data for "8 BP, CPU"
data_8bp_cpu = [
    sim_data_no_ckpt["8bp-cpu-jaxdna-no-False-0.2"],  # Forward
    sim_data_no_ckpt["8bp-cpu-jaxdna-no-True-0.2"],   # Gradient
    sim_data_ckpt["8bp-cpu-jaxdna-no-True-0.2"]       # Gradient + Checkpointing
]


# Combine all data for easier plotting
means = [data[0] for data in data_8bp_cpu]
stds = [data[1] for data in data_8bp_cpu]

# Grouping for each category


for width, height in [(10, 8), (12, 8), (14, 8)]:

    fig, ax = plt.subplots(figsize=(width, height))

    ax.bar(labels, means, yerr=stds, color=colors)

    # Labels, title, and axes formatting
    ax.set_ylabel('Time (s)')
    # ax.set_title('Gradient Calculation Benchmarking')
    # ax.legend()

    # Display the plot
    plt.tight_layout()

    # plt.show()
    plt.savefig(output_dir / f"gradient_calc_benchmarking_cpu_{width}x{height}.pdf")
    plt.close()
