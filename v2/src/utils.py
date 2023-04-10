from pathlib import Path
import pdb
import numpy as np
from itertools import combinations
from io import StringIO
import pandas as pd

from jax import vmap, jit
from jax_md.rigid_body import Quaternion, RigidBody
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)



# DEFAULT_TEMP = 300
DEFAULT_TEMP = 296.15

@jit
def clamp(x, lo=-1.0, hi=1.0):
    """
    correction = 1e-10
    min_ = jnp.where(x + 1e-10 > hi, hi, x)
    max_ = jnp.where(min_ - 1e-10 < lo, lo, min_)
    """

    min_ = jnp.where(x >= hi, hi, x)
    max_ = jnp.where(min_ <= lo, lo, min_)
    return max_

    # return jnp.clip(x, lo, hi)



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
## Tom's thesis, page 23, bottom
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

# the following are the site positions in the BODY frame!
base_site = jnp.array(
    [com_to_hb, 0.0, 0.0]
)
stack_site = jnp.array(
    [com_to_stacking, 0.0, 0.0]
)
back_site = jnp.array(
    [com_to_backbone, 0.0, 0.0]
)

# Transform quaternions to nucleotide orientations

## backbone-base orientation
@jit
def q_to_back_base(q):
    q0, q1, q2, q3 = q.vec
    return jnp.array([
        q0**2 + q1**2 - q2**2 - q3**2,
        2*(q1*q2 + q0*q3),
        2*(q1*q3 - q0*q2)
    ])
Q_to_back_base = jit(vmap(q_to_back_base)) # Q is system of quaternions, q is an individual quaternion
"""
def Q_to_back_base_direct(q):
    q0 = q.vec[:, 0]
    q1 = q.vec[:, 1]
    q2 = q.vec[:, 2]
    q3 = q.vec[:, 3]

    x = q0**2 + q1**2 - q2**2 - q3**2
    y = 2*(q1*q2 + q0*q3)
    z = 2*(q1*q3 - q0*q2)
    return jnp.stack([x, y, z], axis=1)
"""

## normal orientation
@jit
def q_to_base_normal(q):
    q0, q1, q2, q3 = q.vec
    return jnp.array([
        2*(q1*q3 + q0*q2),
        2*(q2*q3 - q0*q1),
        q0**2 - q1**2 - q2**2 + q3**2
    ])
Q_to_base_normal = jit(vmap(q_to_base_normal))

## third axis (n x b)
@jit
def q_to_cross_prod(q):
    q0, q1, q2, q3 = q.vec
    return jnp.array([
        2*(q1*q2 - q0*q3),
        q0**2 - q1**2 + q2**2 - q3**2,
        2*(q2*q3 + q0*q1)
    ])
Q_to_cross_prod = jit(vmap(q_to_cross_prod))

def smooth_max(xs, k=1):
    return jnp.log(jnp.sum(jnp.exp(k*xs))) / k
# min(x, y) = -max(-x, -y)
def smooth_min(xs, k=1):
    return -smooth_max(-xs, k)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Kron: AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT
HB_WEIGHTS = jnp.array([
    0.0, 0.0, 0.0, 1.0, # AX
    0.0, 0.0, 1.0, 0.0, # CX
    0.0, 1.0, 0.0, 0.0, # GX
    1.0, 0.0, 0.0, 0.0  # TX
])
get_hb_probs = vmap(lambda seq, i, j: jnp.kron(seq[i], seq[j]), in_axes=(None, 0, 0), out_axes=0)


stacking_param_names = [
    # f1(dr_stack)
    "eps_stack_base",
    "eps_stack_kt_coeff",
    "a_stack",
    "dr0_stack",
    "dr_c_stack",
    "dr_low_stack",
    "dr_high_stack",

    # f4(theta_4)
    "a_stack_4",
    "theta0_stack_4",
    "delta_theta_star_stack_4",

    # f4(theta_5p)
    "a_stack_5",
    "theta0_stack_5",
    "delta_theta_star_stack_5",

    # f4(theta_6p)
    "a_stack_6",
    "theta0_stack_6",
    "delta_theta_star_stack_6",

    # f5(-cos(phi1))
    "a_stack_1",
    "neg_cos_phi1_star_stack",

    # f5(-cos(phi2))
    "a_stack_2",
    "neg_cos_phi2_star_stack"
]


if __name__ == "__main__":

    top_path = "data/simple-helix/generated.top"
    conf_path = "data/simple-helix/start.conf"
    traj_path = "data/simple-helix/output.dat"

    pdb.set_trace()
    print("done")
