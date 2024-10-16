import matplotlib.pyplot as plt
import numpy as onp
import pdb
import argparse
from pathlib import Path
import glob
from tqdm import tqdm
import matplotlib.animation as animation
from matplotlib import rc

rc('text', usetex=True)
plt.rcParams.update({'font.size': 36})


import jax.numpy as jnp

from jax_dna.common.utils import DNA_ALPHA



def animate_directory(basedir, show_plot, save_plot, save_basedir):

    if save_plot and not str(save_basedir):
        raise RuntimeError(f"Must provide a save basedir")

    save_basedir = Path(save_basedir)
    if save_plot and not save_basedir.exists():
        raise RuntimeError(f"Save basedir does not exist at location: {save_basedir}")

    basedir = Path(basedir)
    assert(basedir.exists())
    pseq_basedir = basedir / "pseq"
    assert(pseq_basedir.exists())

    # Load pseqs
    npy_files = [f for f in pseq_basedir.glob("*.npy")]
    num_pseqs = len(npy_files)

    log_dir = basedir / "log"
    assert(log_dir.exists())
    rgs = onp.loadtxt(log_dir / "obs.txt")

    all_pseq_idxs = list()
    for f in npy_files:
        stem = f.stem
        assert(stem[:6] == "pseq_i")
        tokens = stem.split("pseq_i")
        assert(len(tokens) == 2)
        all_pseq_idxs.append(int(tokens[-1]))

    assert(set(all_pseq_idxs) == set(range(num_pseqs)))

    ## Load in order
    pseqs = list()
    for pseq_idx in tqdm(range(num_pseqs), desc="Loading pseqs"):
        pseq_path = pseq_basedir / f"pseq_i{pseq_idx}.npy"
        assert(pseq_path.exists())
        pseq = jnp.load(pseq_path)
        pseqs.append(pseq)
    pseqs = onp.array(pseqs)


    # Process
    assert(len(pseqs.shape) == 3)
    assert(pseqs.shape[2] == len(DNA_ALPHA))
    num_res = pseqs.shape[2]
    num_frames = pseqs.shape[0]
    seq_len = pseqs.shape[1]

    # Make a dummy/reference plot
    pseq0 = pseqs[0]
    x_labels = onp.arange(seq_len)

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = list()
    bottom0 = onp.zeros(seq_len)
    # colors = ["red", "blue", "orange", "purple"]
    colors = ['#4dad49', '#377db8', '#000000', 'red']
    for res_idx in range(num_res):
        res_weights = pseq0[:, res_idx]
        tmp_bar = ax.bar(x_labels, res_weights, bottom=bottom0, label=DNA_ALPHA[res_idx], color=colors[res_idx])
        bars.append(tmp_bar)
        bottom0 += res_weights

    ax.set_title("Residue distribution")
    ax.set_xlabel("Residue")
    # ax.legend(loc="center right")

    def update(frame):
        pseq = pseqs[frame]

        ret_val = None
        bottom = onp.zeros(seq_len)

        for res_idx in range(num_res):
            res_bar = bars[res_idx]
            res_weights = pseq[:, res_idx]

            for bar, wi, b in zip(res_bar, res_weights, bottom):
                bar.set_height(wi)
                bar.set_y(b)

            if ret_val is None:
                ret_val = res_bar
            else:
                ret_val += res_bar

            bottom += res_weights

        return ret_val

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=3, blit=True)

    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], title='Residue', loc="center right")
    ax.legend(handles[::-1], labels[::-1], title='Residue', bbox_to_anchor=(1.0, 0.85))

    plt.tight_layout()

    if show_plot:
        plt.show()
    elif save_plot:
        animation_fname = "pseq_animation.gif"
        ani.save(filename=save_basedir / animation_fname, writer="pillow")

    plt.clf()






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get identity distributions from a pseq or directory of pseqs")

    parser.add_argument('--basedir', type=str)

    parser.add_argument('--show-plot', action='store_true')
    parser.add_argument('--save-plot', action='store_true')
    parser.add_argument('--save-basedir', type=str, default="")

    args = vars(parser.parse_args())

    basedir = args['basedir']

    show_plot = args['show_plot']
    save_plot = args['save_plot']
    save_basedir = args['save_basedir']
    assert(int(show_plot) + int(save_plot) == 1)

    animate_directory(basedir, show_plot, save_plot, save_basedir)
