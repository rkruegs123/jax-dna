import pdb
from pathlib import Path

import jax.numpy as jnp
from jax_md.rigid_body import RigidBody

from utils import bcolors
from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo


def run(args):
    top_path = args['top_path']
    conf_path = args['conf_path']

    top_info = TopologyInfo(top_path, reverse_direction=True) # Note: we assume reversal for now
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)

    box_size = config_info.box_size
    assert(len(config_info.states) == 1)
    body = config_info.states[0]

    body_avg_pos = jnp.mean(body.center, axis=0)
    box_center = jnp.array([box_size / 2, box_size / 2, box_size / 2])
    disp = body_avg_pos - box_center
    center_adjusted = body.center - disp

    body_centered = RigidBody(center=center_adjusted, orientation=body.orientation)

    centered_conf_path = Path(conf_path).parent / f"{Path(conf_path).stem}_centered.conf"
    centered_traj = TrajectoryInfo(top_info, states=[body_centered], box_size=config_info.box_size)
    centered_traj.write(centered_conf_path, reverse=True, write_topology=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Center an oxDNA trajectory")
    parser.add_argument('--top-path', type=str,
                        default="data/simple-helix/generated.top",
                        help='Path to topology file')
    parser.add_argument('--conf-path', type=str,
                        default="data/simple-helix/start.conf",
                        help='Path to configuration file')
    args = vars(parser.parse_args())
    run(args)
