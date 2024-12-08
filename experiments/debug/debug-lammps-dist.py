from pathlib import Path
import shutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import subprocess
import pdb
from copy import deepcopy
import time
import functools
import numpy as onp
import pprint
import random
import pandas as pd
import socket
# import ray
from collections import Counter

from jax import jit, vmap, lax, value_and_grad
import jax.numpy as jnp
from jax_md import space, rigid_body
import optax

from jax_dna.common.read_seq_specific import read_ss_oxdna
from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.dna2 import model, lammps_utils
import jax_dna.input.trajectory as jdt


def get_bp_pos(body, bp):
    return (body.center[bp[0]] + body.center[bp[1]]) / 2



def single_pitch(quartet, base_sites, displacement_fn):
    # a1 is h-bonded to b1, a2 is h-bonded to b2
    a1, b1, a2, b2 = quartet

    # get normalized base-base vectors for each base pair, 1 and 2
    bb1 = displacement_fn(base_sites[b1], base_sites[a1])
    bb2 = displacement_fn(base_sites[b2], base_sites[a2])

    bb1 = bb1[:2]
    bb2 = bb2[:2]

    bb1 = bb1 / jnp.linalg.norm(bb1)
    bb2 = bb2 / jnp.linalg.norm(bb2)

    theta = jnp.arccos(utils.clamp(jnp.dot(bb1, bb2)))

    return theta


def compute_pitches(body, quartets, displacement_fn, com_to_hb):
    # Construct the base site position in the body frame
    base_site_bf = jnp.array([com_to_hb, 0.0, 0.0])

    # Compute the space-frame base sites
    base_sites = body.center + rigid_body.quaternion_rotate(
        body.orientation, base_site_bf)

    # Compute the pitches for all quartets
    all_pitches = vmap(single_pitch, (0, None, None))(
        quartets, base_sites, displacement_fn)

    return all_pitches





displacement_fn, shift_fn = space.free() # FIXME: could use box size from top_info, but not sure how the centering works.


t_kelvin = 300.0
kT = utils.get_kt(t_kelvin)
beta = 1 / kT
salt_conc = 0.15
q_eff = 0.815


sys_basedir = Path("data/templates/lammps-stretch-tors")
lammps_data_rel_path = sys_basedir / "data"
lammps_data_abs_path = os.getcwd() / lammps_data_rel_path



top_fpath = Path("/home/ryan/Downloads/debug-lammps/sys.top")


top_info = topology.TopologyInfo(top_fpath, reverse_direction=False)
seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
seq = top_info.seq
n = seq_oh.shape[0]
assert(n % 2 == 0)
n_bp = n // 2
strand_length = int(seq_oh.shape[0] // 2)

strand1_start = 0
strand1_end = n_bp-1
strand2_start = n_bp
strand2_end = n_bp*2-1

bp1_meas = [4, strand2_end-4]
bp2_meas = [strand1_end-4, strand2_start+4]

quartets = utils.get_all_quartets(n_nucs_per_strand=n_bp)
quartets = quartets[4:n_bp-5]

pdb.set_trace()


@jit
def compute_distance(body, bp1=bp1_meas, bp2=bp2_meas):
    bp1_meas_pos = get_bp_pos(body, bp1)
    bp2_meas_pos = get_bp_pos(body, bp2)
    dist = jnp.abs(bp1_meas_pos[2] - bp2_meas_pos[2])
    return dist


@jit
def compute_theta(body):
    pitches = compute_pitches(body, quartets, displacement_fn, model.com_to_hb)
    return pitches.sum()



sim_dir = Path("/home/ryan/Downloads/debug-lammps/sim-f0.0")
n_threads = 12
"""
traj_ = jdt.from_file(
    sim_dir / "output.dat",
    [strand_length, strand_length],
    is_oxdna=False,
    n_processes=n_threads,
)
full_traj_states = [ns.to_rigid_body() for ns in traj_.states]
"""

traj_info = trajectory.TrajectoryInfo(
    top_info, read_from_file=True, reindex=True,
    traj_path=sim_dir / "output.dat",
    # reverse_direction=True)
    reverse_direction=False)
full_traj_states = traj_info.get_states()

print(compute_distance(full_traj_states[0]))
print(compute_theta(full_traj_states[0]))


pdb.set_trace()
