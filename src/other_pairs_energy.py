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
from jax.tree_util import Partial

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

from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import nucleotide_mass, get_kt
from utils import Q_to_back_base, Q_to_cross_prod, Q_to_base_normal
from utils import clamp
from potential import exc_vol_unbonded, hydrogen_bonding, cross_stacking, coaxial_stacking

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


# Kron: AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT
HB_WEIGHTS = jnp.array([
    0.0, 0.0, 0.0, 1.0, # AX
    0.0, 0.0, 1.0, 0.0, # CX
    0.0, 1.0, 0.0, 0.0, # GX
    1.0, 0.0, 0.0, 0.0  # TX
])
get_hb_probs = vmap(lambda seq, i, j: jnp.kron(seq[i], seq[j]), in_axes=(None, 0, 0), out_axes=0)

def other_pairs_energy_fn_factory_fixed(displacement_fn, back_site, stack_site, base_site, neighbors):

    d = space.map_bond(partial(displacement_fn))
    nbs_i = neighbors[:, 0]
    nbs_j = neighbors[:, 1]

    def _compute_subterms(body: RigidBody, seq: util.Array, params):
        params_exc_vol_unbonded = Partial(exc_vol_unbonded, params=params)
        params_hydrogen_bonding = Partial(hydrogen_bonding, params=params)
        params_cross_stacking = Partial(cross_stacking, params=params)
        params_coaxial_stacking = Partial(coaxial_stacking, params=params)

        Q = body.orientation

        back_sites = body.center + rigid_body.quaternion_rotate(Q, back_site) # (N, 3)
        stack_sites = body.center + rigid_body.quaternion_rotate(Q, stack_site)
        base_sites = body.center + rigid_body.quaternion_rotate(Q, base_site)

        # Excluded volume (unbonded)
        # dr_base = d(base_sites[nbs_i], base_sites[nbs_j])
        dr_base = d(base_sites[nbs_j], base_sites[nbs_i]) # Note the flip here
        dr_backbone = d(back_sites[nbs_j], back_sites[nbs_i]) # Note the flip here
        dr_back_base = d(back_sites[nbs_i], base_sites[nbs_j]) # Note: didn't flip this one (and others) because no need, but should look into at some point
        dr_base_back = d(base_sites[nbs_i], back_sites[nbs_j])
        exc_vol = params_exc_vol_unbonded(dr_base, dr_backbone, dr_back_base, dr_base_back)


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
        v_hb = params_hydrogen_bonding(dr_base, theta1, theta2, theta3, theta4, theta7, theta8)

        ## Dot product to only be between appropriate bases
        hb_probs = get_hb_probs(seq, nbs_i, nbs_j) # get the probabilities of all possibile hydrogen bonds for all neighbors
        hb_weights = jnp.dot(hb_probs, HB_WEIGHTS)
        v_hb = jnp.dot(hb_weights, v_hb)



        # Cross stacking
        cross_stack = params_cross_stacking(dr_base, theta1, theta2, theta3, theta4, theta7, theta8)

        # Coaxial stacking
        dr_stack = d(stack_sites[nbs_j], stack_sites[nbs_i]) # note: reversed
        dr_stack_norm = dr_stack / jnp.linalg.norm(dr_stack, axis=1, keepdims=True)
        dr_backbone_norm = dr_backbone / jnp.linalg.norm(dr_backbone, axis=1, keepdims=True)
        theta5 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nbs_i], dr_stack_norm)))
        theta6 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -base_normals[nbs_j], dr_stack_norm)))
        cosphi3 = jnp.einsum('ij, ij->i', dr_stack_norm, jnp.cross(dr_backbone_norm, back_bases[nbs_j]))
        cosphi4 = jnp.einsum('ij, ij->i', dr_stack_norm, jnp.cross(dr_backbone_norm, back_bases[nbs_i]))
        coax_stack = params_coaxial_stacking(dr_stack, theta4, theta1, theta5, theta6, cosphi3, cosphi4)


        return jnp.sum(exc_vol), jnp.sum(v_hb), jnp.sum(cross_stack), jnp.sum(coax_stack)

    # Philosophy: use `functools.partial` to fix either `seq` or `potential_fns` depending on what is being optimized
    def energy_fn(body: RigidBody, seq: util.Array, params, **kwargs) -> float:
        exc_vol, v_hb, cross_stack, coax_stack = _compute_subterms(body, seq, params)
        return exc_vol + v_hb + cross_stack + coax_stack

    return energy_fn, _compute_subterms



if __name__ == "__main__":
    pass
