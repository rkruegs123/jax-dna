import pdb

from jax import vmap, jit
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)


DEFAULT_TEMP = 296.15
DNA_BASES = set("ACGT")

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
