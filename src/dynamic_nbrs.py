import numpy as onp
import pdb
import toml
from functools import partial

from jax import jit
from jax import vmap
from jax import random
from jax import lax

from jax.config import config as jax_config
import jax.numpy as jnp

from jax_md import quantity
from jax_md import simulate
from jax_md import space
from jax_md import smap
from jax_md import energy
from jax_md import test_util
from jax_md import partition
from jax_md import util
from jax_md import rigid_body
from jax_md.rigid_body import RigidBody, Quaternion

from potential import exc_vol_unbonded, hydrogen_bonding, cross_stacking, coaxial_stacking
from utils import read_config, jax_traj_to_oxdna_traj
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import nucleotide_mass, get_kt
from utils import Q_to_back_base, Q_to_cross_prod, Q_to_base_normal


from jax.config import config
config.update("jax_enable_x64", True)

FLAGS = jax_config.FLAGS
DYNAMICS_STEPS = 100


f32 = util.f32
f64 = util.f64

DTYPE = [f32]
if FLAGS.jax_enable_x64:
    DTYPE += [f64]

@partial(vmap, in_axes=(0, None))
def rand_quat(key, dtype):
    return rigid_body.random_quaternion(key, dtype)


def clamp(x, lo=-1.0, hi=1.0):
    return jnp.clip(x, lo, hi)

# Kron: AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT
HB_WEIGHTS = jnp.array([
    0.0, 0.0, 0.0, 1.0, # AX
    0.0, 0.0, 1.0, 0.0, # CX
    0.0, 1.0, 0.0, 0.0, # GX
    1.0, 0.0, 0.0, 0.0  # TX
])
get_hb_probs = vmap(lambda seq, i, j: jnp.kron(seq[i], seq[j]), in_axes=(None, 0, 0), out_axes=0)

def dynamic_energy_fn_factory_fixed(displacement_fn, back_site, stack_site, base_site, neighbors):

    d = space.map_bond(partial(displacement_fn))
    nbs_i = neighbors[:, 0]
    nbs_j = neighbors[:, 1]

    def _compute_subterms(body: RigidBody, seq:util.Array):
        Q = body.orientation

        back_sites = body.center + rigid_body.quaternion_rotate(Q, back_site) # (N, 3)
        stack_sites = body.center + rigid_body.quaternion_rotate(Q, stack_site)
        base_sites = body.center + rigid_body.quaternion_rotate(Q, base_site)

        # Excluded volume (unbonded)
        # dr_base = d(base_sites[nbs_i], base_sites[nbs_j])
        dr_base = d(base_sites[nbs_j], base_sites[nbs_i]) # Note the flip here
        dr_backbone = d(back_sites[nbs_i], back_sites[nbs_j])
        dr_back_base = d(back_sites[nbs_i], base_sites[nbs_j])
        dr_base_back = d(base_sites[nbs_i], back_sites[nbs_j])
        exc_vol = exc_vol_unbonded(dr_base, dr_backbone, dr_back_base, dr_base_back)

        # FIXME: add the others

        # Hydrogen bonding
        back_bases = Q_to_back_base(Q) # space frame, normalized
        base_normals = Q_to_base_normal(Q) # space frame, normalized
        cross_prods = Q_to_cross_prod(Q) # space frame, normalized

        # Note: order of theta2 and theta3, and theta7 and theta8, I didn't think about. Does'nt matter for correctness as its the same
        theta1 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -back_bases[nbs_i], back_bases[nbs_j])))
        theta2 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -back_bases[nbs_j], dr_base) / jnp.linalg.norm(dr_base, axis=1)))
        theta3 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', back_bases[nbs_i], dr_base) / jnp.linalg.norm(dr_base, axis=1)))
        theta4 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nbs_i], base_normals[nbs_j])))
        # Note: are these swapped in Lorenzo's code?
        theta7 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -base_normals[nbs_j], dr_base) / jnp.linalg.norm(dr_base, axis=1)))
        theta8 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nbs_i], dr_base) / jnp.linalg.norm(dr_base, axis=1)))
        v_hb = hydrogen_bonding(dr_base, theta1, theta2, theta3, theta4, theta7, theta8)

        ## Dot product to only be between appropriate bases
        hb_probs = get_hb_probs(seq, nbs_i, nbs_j) # get the probabilities of all possibile hydrogen bonds for all neighbors
        hb_weights = jnp.dot(hb_probs, HB_WEIGHTS)
        v_hb = jnp.dot(hb_weights, v_hb)



        # Cross stacking
        cross_stack = cross_stacking(dr_base, theta1, theta2, theta3, theta4, theta7, theta8)


        return jnp.sum(exc_vol), jnp.sum(v_hb), jnp.sum(cross_stack)

    def energy_fn(body: RigidBody, seq: util.Array) -> float:
        exc_vol, v_hb, cross_stack = _compute_subterms(body, seq)
        return exc_vol + v_hb + cross_stack

    return energy_fn, _compute_subterms





# For variable neighbors
def dynamic_energy_fn_factory(displacement_fn, back_site, stack_site, base_site):

    d = space.map_bond(partial(displacement_fn))

    def energy_fn(body: RigidBody, neighbor: partition.NeighborList) -> float:
        Q = body.orientation
        back_sites = body.center + rigid_body.quaternion_rotate(Q, back_site) # (N, 3)
        stack_sites = body.center + rigid_body.quaternion_rotate(Q, stack_site)
        base_sites = body.center + rigid_body.quaternion_rotate(Q, base_site)

        pdb.set_trace()
        nbs_i = neighbor[:, 0]
        nbs_j = neighbor[:, 1]

        # FIXME: completey wrong potential
        # FENE
        dr_back = d(back_sites[nbs_i], back_sites[nbs_j]) # N x N x 3
        r_back = jnp.linalg.norm(dr_back, axis=1)
        fene = v_fene(r_back)

        return jnp.sum(fene)

    return energy_fn


if __name__ == "__main__":
    pass
