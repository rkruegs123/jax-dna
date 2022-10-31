from math import radians
import pdb

from jax.config import config as jax_config
import jax.numpy as jnp
from jax import jit, vmap

from jax_md import util
from jax_md.rigid_body import RigidBody

from utils import clamp
from utils import Q_to_base_normal

from jax.config import config
config.update("jax_enable_x64", True)


Array = util.Array
FLAGS = jax_config.FLAGS
DYNAMICS_STEPS = 100

f32 = util.f32
f64 = util.f64

DTYPE = [f32]
if FLAGS.jax_enable_x64:
    DTYPE += [f64]

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

def get_propeller_loss(system: RigidBody, base_pairs: Array, target_propeller_twist=TARGET_TWIST):
    return (get_avg_propeller_twist(system, base_pairs) - target_propeller_twist)**2


if __name__ == "__main__":
    pass
