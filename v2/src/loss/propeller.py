import pdb
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../loader'))

from jax.config import config as jax_config
import jax.numpy as jnp
import numpy as onp
from jax import jit, vmap

from jax_md import simulate
from jax_md import space
from jax_md import util
from jax_md import rigid_body
from jax_md.rigid_body import RigidBody, Quaternion

from utils import DEFAULT_TEMP
from utils import back_site, stack_site, base_site
from utils import get_one_hot
from utils import clamp
from utils import q_to_base_normal, Q_to_base_normal

from trajectory import TrajectoryInfo
from topology import TopologyInfo
from get_params import get_default_params

from math import radians, degrees
from jax.config import config
config.update("jax_enable_x64", True)
Array=util.Array
FLAGS = jax_config.FLAGS
DYNAMICS_STEPS = 100

f32 = util.f32
f64 = util.f64

DTYPE = [f32]
if FLAGS.jax_enable_x64:
    DTYPE += [f64]

#the propeller twist is defined as the angle between the normal vectors of stacked bases 
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

def get_propeller_loss(system: RigidBody, base_pairs: Array, target_propeller_twist = radians(21.7)):
    return (get_avg_propeller_twist(system, base_pairs) - target_propeller_twist)**2

if __name__ == "__main__":
    from tqdm import tqdm

    top_path = "/Users/megancengel/Research_apps/jaxmd-oxdna/v2/data/test-data/simple-helix/generated.top"
    ##megan testing
    #config_path = "/Users/megancengel/Research_apps/jaxmd-oxdna/v2/data/test-data/simple-helix/start.conf"
    config_path = "/Users/megancengel/Research_apps/jaxmd-oxdna/v2/data/test-data/simple-helix/output.dat"

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=config_path, reverse_direction=True)

    body = config_info.states[-1]

    pdb.set_trace()
    base_pairs = jnp.array([[0, 15], [1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9], [7, 8]])
    twist = 180 - degrees(get_avg_propeller_twist(body, base_pairs))
    pdb.set_trace()
    print("done")
