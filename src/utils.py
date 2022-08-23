from pathlib import Path
import pdb
import numpy as np
from itertools import combinations
from io import StringIO
import pandas as pd

from jax import vmap
from jax_md.rigid_body import Quaternion, RigidBody
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)



DEFAULT_TEMP = 300

# Probabilistic sequence utilities
DNA_MAPPER = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1]
}
DNA_BASES = set("ACGT")

# seq is a string
def get_one_hot(seq):
    seq = seq.upper()
    if not set(seq).issubset(DNA_BASES):
        raise RuntimeError(f"Sequence contains bases other than ACGT: {seq}")
    seq_one_hot = [DNA_MAPPER[b] for b in seq]
    return np.array(seq_one_hot, dtype=np.float64) # float so they can become probabilistic


# oxDNA unit conversions

# Tom's thesis, page 23, bottom
ang_per_oxdna_length = 8.518
def angstroms_to_oxdna_length(ang):
    return ang / ang_per_oxdna_length
def oxdna_length_to_angstroms(l):
    return l * ang_per_oxdna_length
# Tom's thesis, page 24, top
joules_per_oxdna_energy = 4.142e-20
def joules_to_oxdna_energy(j):
    return j / joules_per_oxdna_energy
def oxdna_energy_to_joules(e):
    return e * joules_per_oxdna_energy
kb = 1.380649e-23 # joules per kelvin
kb_oxdna = kb / joules_per_oxdna_energy # oxdna energy per kelvin

"""
Option 1: Exact, but source of minor error
def get_kt(t): # t is temperature in kelvin
    return kb_oxdna * t
"""
# Option 2: Inexact, but exactly what was intended (i.e. kt=0.1E @ 300 K)
def get_kt(t):
    return 0.1 * t/300.0
# Tom's thesis, page 36, bottom (Section 2.5)
amu_per_oxdna_mass = 100
def amu_to_oxdna_mass(amu):
    return amu / amu_per_oxdna_mass
def oxdna_mass_to_amu(m):
    return m * amu_per_oxdna_mass

"""
nucleotide_mass = 3.1575 # 3.1575 M
moment_of_inertia = 0.43512
"""

# Peter's thesis changes mass and moment of inertia to 1
nucleotide_mass = 1.0
moment_of_inertia = [1.0, 1.0, 1.0]

backbone_to_com = 0.24 # 0.24 l. Tom's thesis, page 36, bottom (Section 2.5)
backbone_to_stacking_angstroms = 6.3 # 6.3 A. Tom's thesis, page 23
backbone_to_stacking = angstroms_to_oxdna_length(backbone_to_stacking_angstroms)
backbone_to_hb_angstroms = 6.8 # 6.8 A. Tom's thesis, page 23
backbone_to_hb = angstroms_to_oxdna_length(backbone_to_hb_angstroms)
"""
Diagram (not to scale):
(backbone)----[com]------(stacking)--(hb)
"""
"""
Note: for whatever reason, using the 0.24 value from Tom's thesis doesn't give the values from the oxDNA code and/or documentation. We use those directly, without understanding why they are the way they are
# FIXME: the computed values here for `com_to_X` conflict with the values in "Geometry of the Model": https://dna.physics.ox.ac.uk/index.php/Documentation
com_to_stacking = backbone_to_stacking - backbone_to_com
com_to_hb = backbone_to_hb - backbone_to_com
com_to_backbone = -backbone_to_com
"""
com_to_stacking = 0.34
com_to_hb = 0.4
com_to_backbone = -0.4


# Transform quaternions to nucleotide orientations

## backbone-bsae orientation
def q_to_back_base(q):
    q0, q1, q2, q3 = q.vec
    return jnp.array([
        q0**2 + q1**2 - q2**2 - q3**2,
        2*(q1*q2 + q0*q3),
        2*(q1*q3 - q0*q2)
    ])
Q_to_back_base = vmap(q_to_back_base) # Q is system of quaternions, q is an individual quaternion

## normal orientation
def q_to_base_normal(q):
    q0, q1, q2, q3 = q.vec
    return jnp.array([
        2*(q1*q3 + q0*q2),
        2*(q2*q3 - q0*q1),
        q0**2 - q1**2 - q2**2 + q3**2
    ])
Q_to_base_normal = vmap(q_to_base_normal)

## third axis (n x b)
def q_to_cross_prod(q):
    q0, q1, q2, q3 = q.vec
    return jnp.array([
        2*(q1*q2 - q0*q3),
        q0**2 - q1**2 + q2**2 - q3**2,
        2*(q2*q3 + q0*q1)
    ])
Q_to_cross_prod = vmap(q_to_cross_prod)



if __name__ == "__main__":
    # final_params = get_params()

    top_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/simple-helix/generated.top"
    conf_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/simple-helix/start.conf"
    traj_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/simple-helix/output.dat"

    read_trajectory(traj_path, top_path, reverse=True)
    # convert_3to5_to_5to3(top_path, traj_path)


    # body, box_size, n_strands, bonded_nbrs, unbonded_nbrs, seq = read_config(conf_path, top_path)

    pdb.set_trace()

    # jax_traj_to_oxdna_traj([body], box_size[0], output_name="recovered.dat")

    pdb.set_trace()
    print("done")
