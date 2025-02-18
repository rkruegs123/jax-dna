import functools
import pdb

from jax import jit, vmap
import jax.numpy as jnp

from jax_dna.loss.rna.utils import fit_plane


def get_widths(back_sites, n, first_base, last_base):

    def single_width(bp_idx):
        i = first_base + bp_idx
        dr = (back_sites[i] - back_sites[n-i-1])
        return jnp.linalg.norm(dr)
    n_bps = last_base - first_base - 1
    widths = vmap(single_width)(jnp.arange(n_bps))

    return widths.mean()
    # return widths


if __name__ == "__main__":
    from pathlib import Path
    from jax_dna.common import utils, topology, trajectory
    import numpy as onp
    import matplotlib.pyplot as plt

    basedir = Path("data/templates/rna2-13bp-md")

    top_path = basedir / "sys.top"
    top_info = topology.TopologyInfo(
        top_path,
        reverse_direction=False,
        is_rna=True,
    )

    traj_path = basedir / "output.dat"
    traj_info = trajectory.TrajectoryInfo(
        top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False
    )
    traj_states = traj_info.get_states()
    traj_states = utils.tree_stack(traj_states)

    def compute_body_width(body, first_base, last_base):
        n = body.center.shape[0]

        Q = body.orientation
        back_base_vectors = utils.Q_to_back_base(Q) # space frame, normalized
        base_normals = utils.Q_to_base_normal(Q) # space frame, normalized

        com_to_backbone_x = -0.4
        com_to_backbone_y = 0.2
        back_sites = body.center + com_to_backbone_x*back_base_vectors + com_to_backbone_y*base_normals

        return get_widths(back_sites, n, first_base, last_base)


    # first = 1
    # last = 12
    offset = 1
    n_bp = 13
    first = offset
    # last = n_bp-offset-2
    last = n_bp-offset-1 # -1 for whatever reason (rather than -2) in geom vs. lp code

    # s0_widths = compute_body_width(traj_states[0], first, last)
    all_widths = vmap(compute_body_width, (0, None, None))(traj_states, first, last)
    # sigma_backbone = 0.7
    # all_widths += sigma_backbone
    pdb.set_trace()
    all_widths = all_widths*utils.nm_per_oxrna_length*10

    mean_width = all_widths.mean()
    print(f"Mean width (A): {mean_width}")
