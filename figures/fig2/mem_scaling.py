import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
plt.rcParams.update({'font.size': 24})


no_nbrs = [
    ("A100 MIG, 5GB", 5, 120),
    ("A100 MIG, 10GB", 10, 240),
    ("A100 MIG, 20GB", 20, 500),
    ("V100, 32GB", 32, 820),
    ("A100, 40GB", 40, 1010),
    ("A40, 48GB", 48, 1150),
    ("A100, 80GB", 80, 2040),
]

nbrs = [
    ("A100 MIG, 5GB", 5, 130),
    ("A100 MIG, 10GB", 10, 280),
    ("A100 MIG, 20GB", 20, 570),
    ("V100, 32GB", 32, 930),
    ("A100, 40GB", 40, 1160),
    ("A40, 48GB", 48, 1310),
    ("A100, 80GB", 80, 2350),
]


fig, ax = plt.subplots(figsize=(12, 8))
all_gbs_no_nbrs = list()
all_lengths_no_nbrs = list()
for name, gb, max_length in no_nbrs:
    all_gbs_no_nbrs.append(gb)
    all_lengths_no_nbrs.append(max_length)
    if "MIG" in name:
        ax.axhline(y=gb, linestyle="--", color="grey")
    else:
        ax.axhline(y=gb, linestyle="--", color="blue")
        ax.text(2040, gb+3, name, va='center', ha='left')

ax.plot(all_lengths_no_nbrs, all_gbs_no_nbrs, label="No neighbors", color="red")
all_lengths_nbrs = [entry[2] for entry in nbrs]
all_gbs_nbrs = [entry[1] for entry in nbrs]
ax.plot(all_lengths_nbrs, all_gbs_nbrs, label="Neighbors", color="green")
ax.set_xlabel("Simulation Length")
ax.set_ylabel("GPU Memory (GB)")
ax.set_xlim([0, 2500])
ax.set_ylim([0, 90])
ax.legend()
plt.show()
