import pdb
import numpy as onp

import jax
from jax import vmap, jit, tree_util
import jax.numpy as jnp

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)


DNA_ALPHA = "ACGT"
RNA_ALPHA = "ACGU"

DNA_MAPPER = dict()
for b_idx, base in enumerate(DNA_ALPHA):
    base_oh = [0, 0, 0, 0]
    base_oh[b_idx] = 1
    DNA_MAPPER[base] = base_oh

RNA_MAPPER = dict()
for b_idx, base in enumerate(RNA_ALPHA):
    base_oh = [0, 0, 0, 0]
    base_oh[b_idx] = 1
    RNA_MAPPER[base] = base_oh

# Flattened HB/stack weights yield the correct Kron. product
# Kron: AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT
HB_WEIGHTS_SA = jnp.array([
    [0.0, 0.0, 0.0, 1.0], # AX
    [0.0, 0.0, 1.0, 0.0], # CX
    [0.0, 1.0, 0.0, 0.0], # GX
    [1.0, 0.0, 0.0, 0.0]  # TX
])
STACK_WEIGHTS_SA = jnp.array([
    [1.0, 1.0, 1.0, 1.0], # AX
    [1.0, 1.0, 1.0, 1.0], # CX
    [1.0, 1.0, 1.0, 1.0], # GX
    [1.0, 1.0, 1.0, 1.0]  # TX
])
get_pair_probs = vmap(lambda seq, i, j: jnp.kron(seq[i], seq[j]), in_axes=(None, 0, 0), out_axes=0)

def get_one_hot(seq: str, is_rna=False):

    if is_rna:
        alphabet = RNA_ALPHA
        mapper = RNA_MAPPER
    else:
        alphabet = DNA_ALPHA
        mapper = DNA_MAPPER

    seq = seq.upper()
    if not set(seq).issubset(set(alphabet)):
        raise RuntimeError(f"Sequence contains bases other than ACGT: {seq}")
    seq_one_hot = [mapper[b] for b in seq]
    return onp.array(seq_one_hot, dtype=onp.float64) # float so they can become probabilistic

DEFAULT_TEMP = 296.15 # Kelvin
def get_kt(t_kelvin):
    return 0.1 * t_kelvin / 300.0

def get_t_kelvin(kt):
    return (300.0 * kt) / 0.1

# oxDNA unit conversions

## Tom's thesis, page 23, bottom
ang_per_oxdna_length = 8.518
nm_per_oxdna_length = 0.8518
def angstroms_to_oxdna_length(ang):
    return ang / ang_per_oxdna_length
def oxdna_length_to_angstroms(l):
    return l * ang_per_oxdna_length

oxdna_force_to_pn = 48.63


# Tom's thesis, page 24, top
joules_per_oxdna_energy = 4.142e-20
def joules_to_oxdna_energy(j):
    return j / joules_per_oxdna_energy
def oxdna_energy_to_joules(e):
    return e * joules_per_oxdna_energy

# nucleotide_mass = 3.1575 # 3.1575 M
# moment_of_inertia = 0.43512

# note: Petr's thesis changes mass and moment of inertia to 1
nucleotide_mass = 1.0
moment_of_inertia = [1.0, 1.0, 1.0]

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

def tree_stack(trees):
    return tree_util.tree_map(lambda *v: jnp.stack(v), *trees)

def kelvin_to_celsius(t_kelvin):
    return t_kelvin - 273.15

def celsius_to_kelvin(t_celsius):
    return t_celsius + 273.15


def get_all_bps(n_nucs_per_strand):
    s1_nucs = list(range(n_nucs_per_strand))
    s2_nucs = list(range(n_nucs_per_strand, n_nucs_per_strand*2))
    s2_nucs.reverse()

    bps = list(zip(s1_nucs, s2_nucs))
    return jnp.array(bps)

# Assumes a duplex
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



if __name__ == "__main__":
    pass
