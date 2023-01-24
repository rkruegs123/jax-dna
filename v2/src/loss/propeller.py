from math import radians
import pdb

from jax.config import config as jax_config
import jax.numpy as jnp
from jax import jit, vmap

from jax_md import util
from jax_md.rigid_body import RigidBody

# import sys
# sys.path.append("/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/v2/src")
# pdb.set_trace()
from utils import clamp
from utils import Q_to_base_normal
from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo

from jax.config import config
config.update("jax_enable_x64", True)


Array = util.Array
FLAGS = jax_config.FLAGS

f32 = util.f32
f64 = util.f64

DTYPE = [f32]
if FLAGS.jax_enable_x64:
    DTYPE += [f64]


# Tom's these, botrtom page 57. Potentially averaged from ref 162.
# Note: there seem to be conflicting values on this. Some other citations/values are the following:
# - https://people.bu.edu/mfk/restricted566/dnastructure.pdf -- 12.6 (Table 1)
TARGET_TWIST = radians(21.7)

# the propeller twist is defined as the angle between the normal vectors of stacked bases
def compute_single_propeller_twist(basepair, base_normals: Array):
    A, B = basepair #A, B are the indices of the relevant H-bonded nucleotides
    # get angle between base normal vectors
    theta  = jnp.arccos(clamp(jnp.dot(base_normals[A], base_normals[B])))
    return theta

def get_avg_propeller_twist(system: RigidBody, base_pairs: Array):
    base_normals = Q_to_base_normal(system.orientation)
    get_all_propellers = vmap(compute_single_propeller_twist, in_axes = [0, None])
    all_propellers = get_all_propellers(base_pairs, base_normals)
    return jnp.mean(all_propellers)

def get_propeller_loss_fn(base_pairs: Array, target_propeller_twist=TARGET_TWIST):
    def propeller_loss_fn(body: RigidBody):
        return (get_avg_propeller_twist(body, base_pairs) - target_propeller_twist)**2
    return propeller_loss_fn


if __name__ == "__main__":
    base_pairs = jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
    # base_pairs = jnp.array([[0, 15], [1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9], [7, 8]])

    top_path = "data/simple-helix/generated.top"
    config_path = "data/simple-helix/start.conf"

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=config_path, reverse_direction=True)

    body = config_info.states[0]

    loss_fn = get_propeller_loss_fn(base_pairs)
    curr_loss = loss_fn(body)
    pdb.set_trace()
    print("done")
