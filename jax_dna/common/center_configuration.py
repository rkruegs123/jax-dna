# ruff: noqa
# fmt: off
import pdb
from pathlib import Path

import jax.numpy as jnp
from jax_md.rigid_body import RigidBody

from jax_dna.common.utils import bcolors
from jax_dna.common.trajectory import TrajectoryInfo
from jax_dna.common.topology import TopologyInfo



def center_conf(top_info, config_info):
    box_size = config_info.box_size
    config_info_states = config_info.get_states()
    assert(len(config_info_states) == 1)
    body = config_info_states[0]

    body_avg_pos = jnp.mean(body.center, axis=0)
    box_center = jnp.array([box_size / 2, box_size / 2, box_size / 2])
    disp = body_avg_pos - box_center
    center_adjusted = body.center - disp

    body_centered = RigidBody(center=center_adjusted, orientation=body.orientation)

    centered_traj = TrajectoryInfo(top_info, read_from_states=True,
                                   states=[body_centered], box_size=config_info.box_size)

    return centered_traj


def run(args):
    top_path = args['top_path']
    conf_path = args['conf_path']

    top_info = TopologyInfo(top_path, reverse_direction=True) # Note: we assume reversal for now
    config_info = TrajectoryInfo(top_info, read_from_file=True,
                                 traj_path=conf_path, reverse_direction=True)

    centered_traj = center_conf(top_info, config_info)

    centered_conf_path = Path(conf_path).parent / f"{Path(conf_path).stem}_centered.conf"
    centered_traj.write(centered_conf_path, reverse=True, write_topology=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Center an oxDNA trajectory")
    parser.add_argument('--top-path', type=str,
                        help='Path to topology file')
    parser.add_argument('--conf-path', type=str,
                        help='Path to configuration file')
    args = vars(parser.parse_args())
    run(args)
