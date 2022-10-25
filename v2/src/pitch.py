import pdb

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

from nn_energy import nn_energy_fn_factory
from other_pairs_energy import other_pairs_energy_fn_factory_fixed
from trajectory import TrajectoryInfo
from topology import TopologyInfo
from get_params import get_default_params

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


# the pitch is defined as the angle between the projections of two base-base vectors in a plane perpendicular to the helical axis for two contiguous base pairs
def compute_single_pitch(quartet, system: RigidBody, base_sites: Array):
    # base pair #1 is comprised of nucs a1 and b1. Base pair #2 is a2, b2. i.e. a1 is H-bonded to b1, a2 is h-bonded to b2/
    a1, b1, a2, b2 = quartet #a1, b1, a2, b2 are the indices of the relevant nucleotides
    # get base-base vectors for each base pair, 1 and 2
    bb1 = base_sites[b1] - base_sites[a1]
    bb2 = base_sites[b2] - base_sites[a2]
    # get "average" helical axis
    a2a1 = base_sites[a1] - base_sites[a2]
    b2b1 = base_sites[b1] - base_sites[b2]
    local_helix = 0.5 * (a2a1 + b2b1)
    local_helix_dir = local_helix/jnp.linalg.norm(local_helix)
    # project each of the base-base vectors onto the plane perpendicular to the helical axis
    bb1_projected = bb1 - jnp.dot(bb1, local_helix_dir) * local_helix_dir
    bb2_projected = bb2 - jnp.dot(bb2, local_helix_dir) * local_helix_dir

    bb1_projected_dir = bb1_projected/jnp.linalg.norm(bb1_projected)
    bb2_projected_dir = bb2_projected/jnp.linalg.norm(bb2_projected)
    # find the angle between the projections of the base-base vectors in the plane perpendicular to the "local/average" helical axis
    theta = jnp.arccos(clamp(jnp.dot(bb1_projected_dir, bb2_projected_dir)))
    return theta

def get_pitches(system: RigidBody, base_quartets: Array):
    base_sites = system.center + rigid_body.quaternion_rotate(system.orientation, base_site)
    get_all_pitches = vmap(compute_single_pitch, in_axes = [0, None, None])
    all_pitches = get_all_pitches(base_quartets, system, base_sites)
    return all_pitches


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    from tqdm import tqdm

    # traj_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/test.dat"
    # top_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/polyA_10bp/generated.top"

    top_path = "data/simple-helix/generated.top"
    config_path = "data/simple-helix/start.conf"

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=config_path, reverse_direction=True)

    params = get_default_params()
    seq_oh = jnp.array(get_one_hot(config_info.top_info.seq), dtype=f64)

    body = config_info.states[0]

    pdb.set_trace()
    quartets = jnp.array([[0, 15, 1, 14], [1, 14, 2, 13], [2, 13, 3, 12], [3, 12, 4, 11],
                          [4, 11, 5, 10], [5, 10, 6, 9], [6, 9, 7, 8]])
    base_sites = body.center + rigid_body.quaternion_rotate(body.orientation, base_site)
    # pitch_test = compute_single_pitch(quartet, body, base_sites)
    pitches = get_pitches(body, quartets)
    num_turns = jnp.sum(pitches) / (2*jnp.pi)
    av_pitch = (len(quartets)+1) / (jnp.sum(pitches) / (2*jnp.pi))
    pdb.set_trace()
    print("done")
