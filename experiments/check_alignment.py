import pdb
from pathlib import Path
from tqdm import tqdm
import time
import numpy as onp
from copy import deepcopy
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
from jax_md import space, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.dna1 import model

from jax.config import config
config.update("jax_enable_x64", True)



displacement_fn, shift_fn = space.free()
n_bp = 40
quartets = utils.get_all_quartets(n_nucs_per_strand=n_bp)
bp1 = [0, 79]
bp2 = [39, 40]

rise_per_bp = 3.4 / utils.ang_per_oxdna_length # oxDNA length units
contour_length = quartets.shape[0] * rise_per_bp # oxDNA length units


sys_basedir = Path("data/templates/tors-mod-40bp")
# top_path = sys_basedir / "sys.top"
top_path = "generated.top"
top_info = topology.TopologyInfo(top_path, reverse_direction=True)

# conf_path = sys_basedir / "init.conf"
conf_path = "generated.conf"
conf_info = trajectory.TrajectoryInfo(
    top_info,
    read_from_file=True, traj_path=conf_path, reverse_direction=True
)

init_body = conf_info.get_states()[0]


def get_bp_pos(body, bp):
    return (body.center[bp[0]] + body.center[bp[1]]) / 2

bp1_midp = get_bp_pos(init_body, bp1)
bp2_midp = get_bp_pos(init_body, bp2)
helical_axis = displacement_fn(bp2_midp, bp1_midp)
helical_axis_norm = helical_axis / jnp.linalg.norm(helical_axis)


target_axis = jnp.array([0., 0., 1.])
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
new_conf = trajectory.TrajectoryInfo(top_info, read_from_states=True,
                                     states=utils.tree_stack([new_rb]), box_size=200.0)
new_conf.write("recentered.conf", reverse=False, write_topology=False)
pdb.set_trace()

init_body = new_rb


bp1_midp = get_bp_pos(init_body, bp1)
bp2_midp = get_bp_pos(init_body, bp2)
helical_axis = displacement_fn(bp2_midp, bp1_midp)
helical_axis_norm = helical_axis / jnp.linalg.norm(helical_axis)


all_q_axis = list()
for q in quartets:
    q_midp1 = get_bp_pos(init_body, q[:2])
    q_midp2 = get_bp_pos(init_body, q[2:])
    q_axis = displacement_fn(q_midp2, q_midp1)
    all_q_axis.append(q_axis)

    q_axis_norm = q_axis / jnp.linalg.norm(q_axis)
    diff = q_axis_norm - helical_axis_norm
    print(f"- Diff: {diff}")

# Plot x-z plane
for q, q_axis in zip(quartets, all_q_axis):
    q_midp1 = get_bp_pos(init_body, q[:2])
    plt.plot([q_midp1[0], q_midp1[0]+q_axis[0]], [q_midp1[2], q_midp1[2]+q_axis[2]])
    # plt.arrow(q_midp1[0], q_midp1[2], q_axis[0], q_axis[2], width=0.005, head_width=0.05)

plt.plot([bp1_midp[0], bp2_midp[0]], [bp1_midp[2], bp2_midp[2]], linestyle="--")

plt.show()
plt.clf()

pdb.set_trace()

# Plot y-z plane
for q, q_axis in zip(quartets, all_q_axis):
    q_midp1 = get_bp_pos(init_body, q[:2])
    plt.plot([q_midp1[1], q_midp1[1]+q_axis[1]], [q_midp1[2], q_midp1[2]+q_axis[2]])
    # plt.arrow(q_midp1[1], q_midp1[2], q_axis[1], q_axis[2], width=0.005, head_width=0.05)

plt.plot([bp1_midp[1], bp2_midp[1]], [bp1_midp[2], bp2_midp[2]], linestyle="--")
plt.show()
