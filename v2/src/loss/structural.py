import pdb

from jax.config import config as jax_config
import jax.numpy as jnp
from jax import jit, vmap

from jax_md import util
from jax_md.rigid_body import RigidBody

# import sys
# sys.path.append("/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/v2/src")
# pdb.set_trace()
from loss import propeller, pitch, geometry

from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo


Array = util.Array


def get_structural_loss_fn(
        backbone_dist_pairs, displacement_fn,
        pitch_base_quartets,
        propeller_base_pairs: Array):
    propeller_loss_fn = propeller.get_propeller_loss_fn(propeller_base_pairs)
    pitch_loss_fn = pitch.get_pitch_distance_loss(pitch_base_quartets)
    geometry_loss_fn = geometry.get_backbone_distance_loss(backbone_dist_pairs, displacement_fn)
    def structural_loss_fn(body):
        return geometry_loss_fn(body) + pitch_loss_fn(body) + propeller_loss_fn(body)

    return structural_loss_fn


if __name__ == "__main__":
    from jax_md import space

    top_path = "data/simple-helix/generated.top"
    # traj_path = "data/simple-helix/output.dat"
    config_path = "data/simple-helix/start.conf"

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=config_path, reverse_direction=True)

    body = config_info.states[0]

    displacement_fn, _ = space.periodic(config_info.box_size)
    backbone_dist_pairs = top_info.bonded_nbrs
    pitch_quartets = jnp.array([
        [0, 15, 1, 14],
        [1, 14, 2, 13],
        [2, 13, 3, 12],
        [3, 12, 4, 11],
        [4, 11, 5, 10],
        [5, 10, 6, 9],
        [6, 9, 7, 8]
    ])
    propeller_base_pairs = jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])


    loss_fn = get_structural_loss_fn(
        backbone_dist_pairs,
        displacement_fn,
        pitch_quartets,
        propeller_base_pairs
    )

    curr_loss = loss_fn(body)
    pdb.set_trace()
    print("done")
