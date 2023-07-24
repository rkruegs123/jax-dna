import numpy as onp
import pdb
import toml
from functools import partial
from copy import deepcopy

from jax import jit
from jax import vmap
from jax import random
from jax import lax
from jax.tree_util import Partial
from jax import debug

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

from utils import back_site, stack_site, base_site
from utils import nucleotide_mass, get_kt, moment_of_inertia
from utils import Q_to_back_base, Q_to_cross_prod, Q_to_base_normal
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import clamp
from trajectory import TrajectoryInfo
from topology import TopologyInfo
# from potential import v_fene, exc_vol_bonded, stacking
from potential_hard import v_fene, v_fene2, exc_vol_bonded, stacking, exc_vol_bonded2, stacking2, \
    hydrogen_bonding2, exc_vol_unbonded2
from get_params import get_default_params

from jax.config import config
config.update("jax_enable_x64", True)



# Kron: AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT
HB_WEIGHTS = jnp.array([
    0.0, 0.0, 0.0, 1.0, # AX
    0.0, 0.0, 1.0, 0.0, # CX
    0.0, 1.0, 0.0, 0.0, # GX
    1.0, 0.0, 0.0, 0.0  # TX
])
get_hb_probs = vmap(lambda seq, i, j: jnp.kron(seq[i], seq[j]), in_axes=(None, 0, 0), out_axes=0)


def energy_fn_factory(displacement_fn,
                      back_site, stack_site, base_site,
                      bonded_neighbors, unbonded_neighbors):

    params = get_default_params(t=300, no_smoothing=True) # FIXME: hardcoded temperature for now
    hb_params = params["hydrogen_bonding"]
    exc_vol_unbonded_params = params["excluded_volume"]
    exc_vol_bonded_params = deepcopy(exc_vol_unbonded_params)
    del exc_vol_bonded_params["dr_star_backbone"]
    del exc_vol_bonded_params["sigma_backbone"]
    stacking_params = params["stacking"]

    params_hb = Partial(hydrogen_bonding2, **hb_params)
    params_exc_vol_unbonded = Partial(exc_vol_unbonded2, **exc_vol_unbonded_params)
    params_exc_vol_bonded = Partial(exc_vol_bonded2, **exc_vol_bonded_params)
    params_stacking = Partial(stacking2, **stacking_params)

    d = space.map_bond(partial(displacement_fn))
    nn_i = bonded_neighbors[:, 0]
    nn_j = bonded_neighbors[:, 1]
    op_i = unbonded_neighbors[:, 0]
    op_j = unbonded_neighbors[:, 1]

    def energy_fn(body: RigidBody, seq: util.Array, params, **kwargs) -> float:

        # params_v_fene = Partial(v_fene2, eps_backbone=params[0], delta_backbone=params[1],
                                # r0_backbone=params[2])

        fene_params = dict(zip(["eps_backbone", "delta_backbone", "r0_backbone"], params[:3]))
        params_v_fene = Partial(v_fene2, **fene_params)

        """
        exc_vol_bonded_params = dict(zip([
            "eps_exc", "dr_star_base", "sigma_base", "dr_star_back_base", "sigma_back_base",
            "dr_star_base_back", "sigma_base_back"], params[3:10]))
        params_exc_vol_bonded = Partial(exc_vol_bonded2, **exc_vol_bonded_params)
        """

        """
        stacking_params = dict(zip([
            "dr_low_stack", "dr_high_stack", "eps_stack", "a_stack", "dr0_stack", "dr_c_stack",
            "theta0_stack_4", "delta_theta_star_stack_4", "a_stack_4",
            "theta0_stack_5", "delta_theta_star_stack_5", "a_stack_5",
            "theta0_stack_6", "delta_theta_star_stack_6", "a_stack_6",
            "neg_cos_phi1_star_stack", "a_stack_1",
            "neg_cos_phi2_star_stack", "a_stack_2"], params[10:29]))
        params_stacking = Partial(stacking2, **stacking_params)
        """


        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q)
        back_sites = body.center + com_to_backbone * back_base_vectors
        stack_sites = body.center + com_to_stacking * back_base_vectors
        base_sites = body.center + com_to_hb * back_base_vectors

        # back_sites = body.center + rigid_body.quaternion_rotate(Q, back_site) # (N, 3)
        # stack_sites = body.center + rigid_body.quaternion_rotate(Q, stack_site)
        # base_sites = body.center + rigid_body.quaternion_rotate(Q, base_site)

        # FENE
        dr_back = d(back_sites[nn_i], back_sites[nn_j]) # N x N x 3
        r_back = jnp.linalg.norm(dr_back, axis=1)
        fene = params_v_fene(r_back)


        # Excluded volume (bonded)
        dr_base = d(base_sites[nn_i], base_sites[nn_j])
        dr_back_base = d(back_sites[nn_i], base_sites[nn_j])
        dr_base_back = d(base_sites[nn_i], back_sites[nn_j])
        exc_vol_bonded = params_exc_vol_bonded(dr_base, dr_back_base, dr_base_back)


        # Stacking
        dr_stack = d(stack_sites[nn_i], stack_sites[nn_j])
        r_stack = jnp.linalg.norm(dr_stack, axis=1)
        base_normals = Q_to_base_normal(Q) # space frame, normalized
        cross_prods = Q_to_cross_prod(Q) # space frame, normalized

        # theta4 = jnp.zeros(len(nn_i))
        # theta5 = jnp.zeros(len(nn_i))
        # theta6 = jnp.zeros(len(nn_i))
        # cosphi1 = jnp.zeros(len(nn_i))
        # cosphi2 = jnp.zeros(len(nn_i))


        theta4 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nn_i], base_normals[nn_j])))
        theta5 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', dr_stack, base_normals[nn_j]) / r_stack))
        theta6 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nn_i], dr_stack) / r_stack))
        cosphi1 = -jnp.einsum('ij, ij->i', cross_prods[nn_i], dr_back) / r_back
        cosphi2 = -jnp.einsum('ij, ij->i', cross_prods[nn_j], dr_back) / r_back

        stack = params_stacking(r_stack, theta4, theta5, theta6, cosphi1, cosphi2)


        # Now, "other pairs" stuff

        ## Excluded volume (unbonded)
        dr_base_op = d(base_sites[op_j], base_sites[op_i]) # Note the flip here
        dr_backbone_op = d(back_sites[op_j], back_sites[op_i]) # Note the flip here
        dr_back_base_op = d(back_sites[op_i], base_sites[op_j]) # Note: didn't flip this one (and others) because no need, but should look into at some point
        dr_base_back_op = d(base_sites[op_i], back_sites[op_j])
        exc_vol_unbonded = params_exc_vol_unbonded(dr_base_op, dr_backbone_op, dr_back_base_op,
                                                   dr_base_back_op)

        ## Hydrogen bonding
        r_base_op = jnp.linalg.norm(dr_base_op, axis=1)
        back_bases = Q_to_back_base(Q) # space frame, normalized
        theta1_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -back_bases[op_i], back_bases[op_j])))
        theta2_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -back_bases[op_j], dr_base_op) / r_base_op))
        theta3_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', back_bases[op_i], dr_base_op) / r_base_op))
        theta4_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[op_i], base_normals[op_j])))
        # Note: are these swapped in Lorenzo's code?
        theta7_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -base_normals[op_j], dr_base_op) / r_base_op))
        theta8_op = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[op_i], dr_base_op) / r_base_op))
        v_hb = params_hb(dr_base_op, theta1_op, theta2_op, theta3_op, theta4_op,
                                       theta7_op, theta8_op)
        ### Dot product to only be between appropriate bases
        hb_probs = get_hb_probs(seq, op_i, op_j) # get the probabilities of all possibile hydrogen bonds for all neighbors
        hb_weights = jnp.dot(hb_probs, HB_WEIGHTS)
        v_hb = jnp.dot(hb_weights, v_hb)


        # return params["fene"]["eps_backbone"]
        # return jnp.sum(params) * jnp.sum(r_back)
        return jnp.sum(fene) + jnp.sum(exc_vol_bonded) + jnp.sum(stack) \
            + jnp.sum(exc_vol_unbonded) + jnp.sum(v_hb)


    return energy_fn


if __name__ == "__main__":
    pass
