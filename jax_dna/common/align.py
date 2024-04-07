import pdb
from pathlib import Path

import jax.numpy as jnp
from jax_md import space, rigid_body

from jax_dna.common import utils
from jax_dna.common.trajectory import TrajectoryInfo
from jax_dna.common.topology import TopologyInfo


def get_bp_pos(body, bp):
    return (body.center[bp[0]] + body.center[bp[1]]) / 2

def align_conf(top_info, config_info):
    # We assume a duplex
    assert(top_info.n % 2 == 0)
    n_bp = top_info.n // 2
    quartets = utils.get_all_quartets(n_nucs_per_strand=n_bp)
    bp1 = [0, top_info.n-1]
    bp2 = [n_bp-1, n_bp]

    config_info_states = config_info.get_states()
    assert(len(config_info_states) == 1)
    init_body = config_info_states[0]

    displacement_fn, shift_fn = space.free()

    bp1_midp = get_bp_pos(init_body, bp1)
    bp2_midp = get_bp_pos(init_body, bp2)
    helical_axis = displacement_fn(bp2_midp, bp1_midp)
    helical_axis_norm = helical_axis / jnp.linalg.norm(helical_axis)

    target_axis = jnp.array([0., 0., 1.]) # FIXME: make an argument

    crossed = jnp.cross(helical_axis_norm, target_axis)
    crossed = crossed / jnp.linalg.norm(crossed)
    dotted = jnp.dot(helical_axis_norm, target_axis)
    theta = jnp.arccos(dotted)
    cos_part = jnp.cos(theta / 2)
    sin_part = crossed * jnp.sin(theta/2)
    orientation = jnp.concatenate([cos_part.reshape(-1), sin_part])
    rot_q = rigid_body.Quaternion(orientation)

    new_center = rigid_body.quaternion_rotate(rot_q, init_body.center)
    new_rb = rigid_body.RigidBody(center=new_center,
                                  orientation=rot_q * init_body.orientation)
    box_size = config_info.box_size
    new_conf = TrajectoryInfo(top_info, read_from_states=True,
                              states=utils.tree_stack([new_rb]), box_size=box_size)

    return new_conf


def run(args, reverse_direction=False):
    top_path = args['top_path']
    conf_path = args['conf_path']

    top_info = TopologyInfo(top_path, reverse_direction=reverse_direction)
    config_info = TrajectoryInfo(top_info, read_from_file=True,
                                 traj_path=conf_path, reverse_direction=reverse_direction)

    aligned_traj = align_conf(top_info, config_info)

    aligned_conf_path = Path(conf_path).parent / f"{Path(conf_path).stem}_aligned.conf"
    aligned_traj.write(aligned_conf_path, reverse=True, write_topology=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Center an oxDNA trajectory")
    parser.add_argument('--top-path', type=str,
                        help='Path to topology file')
    parser.add_argument('--conf-path', type=str,
                        help='Path to configuration file')
    args = vars(parser.parse_args())
    run(args)
