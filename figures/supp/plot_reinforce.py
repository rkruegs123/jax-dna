import numpy as onp
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc
import pdb

rc('text', usetex=True)
plt.rcParams.update({'font.size': 36})

output_dir = Path("figures/supp/output")

target_pitch = 11

data_dir = Path("figures/supp/data")
reinforce_dir = data_dir / "reinforce"

target_dir = reinforce_dir / f"t{target_pitch}"

for width, height in [(16, 10), (14, 10), (12, 10)]:

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axhline(y=target_pitch, linestyle="--", color="red", label="Target Pitch", linewidth=3)

    all_runs = [
        # (n_steps, batch_size)
        # (1000, 10),
        # (1000, 25),
        (1000, 50),
        (100, 100),
        # (100, 25),
        # (100, 50),
        (500, 100),
        # (500, 25),
        # (500, 50)
    ]


    max_iters = 1000
    for n_steps, batch_size in all_runs:
        run_dir = target_dir / f"n{n_steps}-b{batch_size}"


        pitch_path = run_dir / "log" / "pitch.txt"
        pitches = onp.loadtxt(pitch_path)

        pitches = pitches[:max_iters]
        num_iters = len(pitches)
        iters = onp.arange(num_iters)

        # ax.plot(iters, pitches, color="black")
        ax.plot(iters, pitches, label=f"{n_steps} Steps, Batch Size = {batch_size}", linewidth=3)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Pitch (bp/turn)")



    # y_vals = [10.75, 11.0]
    # ax.set_yticks(y_vals)
    # ax.set_yticklabels([r'$10.75$', r'$11$'])
    # ax.set_ylim(ymin=10.7, ymax=11.1)

    # ax.legend(loc="upper right")
    ax.legend()
    # ax.legend(bbox_to_anchor=[1.0, 0.95])
    plt.tight_layout()


    # plt.show()
    plt.savefig(output_dir / f"reinforce_pitch_{width}_{height}.pdf")

    plt.close()
