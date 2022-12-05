import pdb
from functools import partial

import jax.numpy as jnp
from jax_md import space

from utils import Q_to_back_base
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import angstroms_to_oxdna_length


def get_backbone_distance_loss(pairs, displacement_fn,
                               target_distance=angstroms_to_oxdna_length(6.5)):

    d = space.map_bond(partial(displacement_fn))
    nbrs_i = pairs[:, 0]
    nbrs_j = pairs[:, 1]

    # note: so much repeat computatoin given that we've already computed this...
    def backbone_distance_loss(body):
        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q)
        back_sites = body.center + com_to_backbone * back_base_vectors
        dr_back = d(back_sites[nbrs_i], back_sites[nbrs_j])
        r_back = jnp.linalg.norm(dr_back, axis=1)

        return jnp.sum(((r_back - target_distance))**2)
        # return r_back, jnp.sum((100*(r_back - target_distance))**2)
        # avg_r_back = jnp.mean(r_back)
        # return (avg_r_back - target_distance)**2
    return backbone_distance_loss


if __name__ == "__main__":

    from topology import TopologyInfo
    from trajectory import TrajectoryInfo

    top_path = "data/simple-helix/generated.top"
    # traj_path = "data/simple-helix/output.dat"
    config_path = "data/simple-helix/start.conf"

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=config_path, reverse_direction=True)

    body = config_info.states[0]

    displacement_fn, _ = space.periodic(config_info.box_size)

    pairs = top_info.bonded_nbrs[2:4]
    loss_fn = get_backbone_distance_loss(pairs, displacement_fn)
    loss_fn(body)
    pdb.set_trace()
