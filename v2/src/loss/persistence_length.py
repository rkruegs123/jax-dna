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

import sys
sys.path.append("v2/src/")
# pdb.set_trace()

from utils import DEFAULT_TEMP
from utils import back_site, stack_site, base_site
from utils import get_one_hot
from utils import clamp


from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo
from loader.get_params import get_default_params

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

def vector_autocorrelate(arr):
  n_vectors = arr.shape[0]
  # correlate each component indipendently
  acorr = jnp.array([jnp.correlate(arr[:,i],arr[:,i],'full') for i in jnp.arange(3)])[:,n_vectors-1:] #we should  really vmap over this, but for simplicity, we unroll a for loop for now
  # sum the correlations for each component
  acorr = jnp.sum(acorr, axis = 0)
  # divide by the number of values actually measured and return
  acorr /= (n_vectors - jnp.arange(n_vectors))
  return acorr

# Tom's thesis: 130-150 base pairs
TARGET_PERSISTENCE_LENGTH_DSDNA = 140

def compute_l_vector(quartet, system: RigidBody, base_sites: Array):
    # base pair #1 is comprised of nucs a1 and b1. Base pair #2 is a2, b2. i.e. a1 is H-bonded to b1, a2 is h-bonded to b2/
    a1, b1, a2, b2 = quartet #a1, b1, a2, b2 are the indices of the relevant nucleotides
    # get midpoints for each base pair, 1 and 2
    mp1 = (base_sites[b1] + base_sites[a1]) / 2.
    mp2 = (base_sites[b2] + base_sites[a2]) / 2.
    # get vector between midpoint
    l = mp2 - mp1
    l0 = jnp.linalg.norm(l)
    return l, l0

# vector autocorrelate from https://stackoverflow.com/questions/48844295/computing-autocorrelation-of-vectors-with-numpy

def get_correlation_curve(system: RigidBody, base_quartets: Array):
    base_sites = system.center + rigid_body.quaternion_rotate(system.orientation, base_site)
    get_all_l_vectors = vmap(compute_l_vector, in_axes = [0, None, None])
    all_l_vectors, l0_vals = get_all_l_vectors(base_quartets, system, base_sites)
    autocorr = vector_autocorrelate(all_l_vectors)
    return autocorr, jnp.mean(l0_vals)

def persistence_length_fit(autocorr, l0_av):

    y = jnp.log(autocorr)
    # x = jnp.linspace(0, autocorr.shape[0], 1)
    x = jnp.arange(autocorr.shape[0])
    x = jnp.stack([jnp.ones_like(x), x], axis=1)
    ### fit line:fit_ =	jax.numpy.linalg.lstsq(x, y)
    fit_ = jnp.linalg.lstsq(x, y)
    ### extract slope = -l0_av/Lp ---> Lp = -l0_av/slope
    slope = fit_[0][1]
    Lp = -l0_av/slope
    return Lp

def get_persistence_length_loss(base_quartets, target_avg_pitch=TARGET_PERSISTENCE_LENGTH_DSDNA):
    n_quartets = base_quartets.shape[0]
    def Lp_loss_fn(body):
        correlation_curve, l0_avg = get_correlation_curve(body, base_quartets)
        return correlation_curve, persistence_length_fit(correlation_curve, l0_avg)
        # return (target_avg_pitch - avg_pitch)**2
    return Lp_loss_fn



if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from pathlib import Path

    # top_path = "data/simple-helix/generated.top"
    # config_path = "data/simple-helix/start.conf"
    # top_path = "data/persistence-length/init.top"
    # config_path = "data/persistence-length/relaxed.dat"

    # bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/langevin_2023-01-31_16-23-04")
    # bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/langevin_2023-01-31_16-38-50")
    bpath = Path("/home/ryan/Documents/Harvard/research/brenner/tmp-oxdna/langevin_2023-01-31_01-20-52")
    top_path = bpath / "init.top"
    config_path = bpath / "output.dat"

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=config_path, reverse_direction=True)

    params = get_default_params()
    seq_oh = jnp.array(get_one_hot(config_info.top_info.seq), dtype=f64)

    body = config_info.states[0]

    def get_all_quartets(n_nucs_per_strand):
        s1_nucs = list(range(n_nucs_per_strand))
        s2_nucs = list(range(n_nucs_per_strand, n_nucs_per_strand*2))
        s2_nucs.reverse()

        bps = list(zip(s1_nucs, s2_nucs))
        n_bps = len(s1_nucs)
        all_quartets = list()
        for i in range(n_bps-1):
            bp1 = bps[i]
            bp2 = bps[i+1]
            all_quartets.append(bp1 + bp2)
        return jnp.array(all_quartets, dtype=jnp.int32)

    quartets = get_all_quartets(n_nucs_per_strand=body.center.shape[0] // 2)

    # chop off the ends
    quartets = quartets[25:]
    quartets = quartets[:-25]

    pdb.set_trace()

    """
    quartets = jnp.array([
        [0, 15, 1, 14],
        [1, 14, 2, 13],
        [2, 13, 3, 12],
        [3, 12, 4, 11],
        [4, 11, 5, 10],
        [5, 10, 6, 9],
        [6, 9, 7, 8]
    ])
    """

    all_curves = list()
    all_l0_avg = list()

    # for b_idx in tqdm(range(0, len(config_info.states), 10)):
    for b_idx in tqdm(range(0, len(config_info.states), 1)):
        body = config_info.states[b_idx]
        correlation_curve, l0_avg = get_correlation_curve(body, quartets)
        Lp = persistence_length_fit(correlation_curve, l0_avg)

        # plt.plot(range(len(correlation_curve)), correlation_curve)
        # plt.show()
        # plt.clf()

        all_curves.append(correlation_curve)
        all_l0_avg.append(l0_avg)

    all_curves = jnp.array(all_curves)
    all_l0_avg = jnp.array(all_l0_avg)
    mean_correlation_curve = jnp.mean(all_curves, axis=0)
    mean_l0_avg = jnp.mean(all_l0_avg)

    plt.plot(range(len(mean_correlation_curve)), mean_correlation_curve)
    plt.show()
    plt.clf()

    Lp = persistence_length_fit(mean_correlation_curve, mean_l0_avg)


    # loss_fn = get_persistence_length_loss(quartets)
    # curr_loss = loss_fn(body)


    pdb.set_trace()
    print("done")
