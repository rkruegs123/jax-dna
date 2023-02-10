import pdb
from functools import partial
from jax import vmap, jit
from jax.tree_util import Partial
from jax import debug
import jax.numpy as jnp
from copy import deepcopy

# import sys
# sys.path.append('v2/src/')

from jax_md.rigid_body import RigidBody
from jax_md import util, space

from loader import get_params, smoothing
from energy.interactions import v_fene, exc_vol_bonded, stacking, \
    hydrogen_bonding, exc_vol_unbonded, cross_stacking, coaxial_stacking
from utils import Q_to_back_base, Q_to_cross_prod, Q_to_base_normal
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import clamp, get_kt

f64 = util.f64


# Kron: AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT
HB_WEIGHTS = jnp.array([
    0.0, 0.0, 0.0, 1.0, # AX
    0.0, 0.0, 1.0, 0.0, # CX
    0.0, 1.0, 0.0, 0.0, # GX
    1.0, 0.0, 0.0, 0.0  # TX
])
get_hb_probs = vmap(lambda seq, i, j: jnp.kron(seq[i], seq[j]), in_axes=(None, 0, 0), out_axes=0)


# FIXME: Duplicate logic from when we actually read in the parameters...
def process_stacking_params(unprocessed_params, kt):
    a_stack = unprocessed_params['a_stack']
    dr0_stack = unprocessed_params['dr0_stack']
    dr_c_stack = unprocessed_params['dr_c_stack']
    dr_low_stack = unprocessed_params['dr_low_stack']
    dr_high_stack = unprocessed_params['dr_high_stack']
    eps_stack_base = unprocessed_params['eps_stack_base']
    eps_stack_kt_coeff = unprocessed_params['eps_stack_kt_coeff']
    eps_stack = eps_stack_base + eps_stack_kt_coeff * kt

    b_low_stack, dr_c_low_stack, b_high_stack, dr_c_high_stack = smoothing.get_f1_smoothing_params(
        eps_stack, dr0_stack, a_stack, dr_c_stack,
        dr_low_stack, dr_high_stack)

    a_stack_4 = unprocessed_params['a_stack_4']
    theta0_stack_4 = unprocessed_params['theta0_stack_4']
    delta_theta_star_stack_4 = unprocessed_params['delta_theta_star_stack_4']
    b_stack_4, delta_theta_stack_4_c = smoothing.get_f4_smoothing_params(
        a_stack_4,
        theta0_stack_4,
        delta_theta_star_stack_4)

    a_stack_5 = unprocessed_params['a_stack_5']
    theta0_stack_5 = unprocessed_params['theta0_stack_5']
    delta_theta_star_stack_5 = unprocessed_params['delta_theta_star_stack_5']
    b_stack_5, delta_theta_stack_5_c = smoothing.get_f4_smoothing_params(
        a_stack_5,
        theta0_stack_5,
        delta_theta_star_stack_5)

    a_stack_6 = unprocessed_params['a_stack_6']
    theta0_stack_6 = unprocessed_params['theta0_stack_6']
    delta_theta_star_stack_6 = unprocessed_params['delta_theta_star_stack_6']
    b_stack_6, delta_theta_stack_6_c = smoothing.get_f4_smoothing_params(
        a_stack_6,
        theta0_stack_6,
        delta_theta_star_stack_6)

    a_stack_1 = unprocessed_params['a_stack_1']
    neg_cos_phi1_star_stack = unprocessed_params['neg_cos_phi1_star_stack']
    b_neg_cos_phi1_stack, neg_cos_phi1_c_stack = smoothing.get_f5_smoothing_params(
        a_stack_1,
        neg_cos_phi1_star_stack)

    a_stack_2 = unprocessed_params['a_stack_2']
    neg_cos_phi2_star_stack = unprocessed_params['neg_cos_phi2_star_stack']
    b_neg_cos_phi2_stack, neg_cos_phi2_c_stack = smoothing.get_f5_smoothing_params(
        a_stack_2,
        neg_cos_phi2_star_stack)

    processed_params = {
        # f1(dr_stack)
        "eps_stack": eps_stack,
        "dr0_stack": dr0_stack,
        "a_stack": a_stack,
        "dr_c_stack": dr_c_stack,
        "dr_low_stack": dr_low_stack,
        "dr_high_stack": dr_high_stack,
        "b_low_stack": b_low_stack,
        "dr_c_low_stack": dr_c_low_stack,
        "b_high_stack": b_high_stack,
        "dr_c_high_stack": dr_c_high_stack,

        # f4(theta_4)
        "a_stack_4": a_stack_4,
        "theta0_stack_4": theta0_stack_4,
        "delta_theta_star_stack_4": delta_theta_star_stack_4,
        "b_stack_4": b_stack_4,
        "delta_theta_stack_4_c": delta_theta_stack_4_c,

        # f4(theta_5p)
        "a_stack_5": a_stack_5,
        "theta0_stack_5": theta0_stack_5,
        "delta_theta_star_stack_5": delta_theta_star_stack_5,
        "b_stack_5": b_stack_5,
        "delta_theta_stack_5_c": delta_theta_stack_5_c,

        # f4(theta_6p)
        "a_stack_6": a_stack_6,
        "theta0_stack_6": theta0_stack_6,
        "delta_theta_star_stack_6": delta_theta_star_stack_6,
        "b_stack_6": b_stack_6,
        "delta_theta_stack_6_c": delta_theta_stack_6_c,

        ## f5(-cos(phi1))
        "a_stack_1": a_stack_1,
        "neg_cos_phi1_star_stack": neg_cos_phi1_star_stack,
        "b_neg_cos_phi1_stack": b_neg_cos_phi1_stack,
        "neg_cos_phi1_c_stack": neg_cos_phi1_c_stack,

        ## f5(-cos(phi2))
        "a_stack_2": a_stack_2,
        "neg_cos_phi2_star_stack": neg_cos_phi2_star_stack,
        "b_neg_cos_phi2_stack": b_neg_cos_phi2_stack,
        "neg_cos_phi2_c_stack": neg_cos_phi2_c_stack

    }
    return processed_params



def energy_fn_factory(displacement_fn,
                      back_site, stack_site, base_site,
                      bonded_neighbors, unbonded_neighbors,
                      temp=300):

    kt = get_kt(temp) # For use when optimizing over stacking parameters
    params = get_params.get_default_params(t=temp, no_smoothing=False) # FIXME: hardcoded temperature for now

    # Define which functions we will *not* be optimizing over
    # stacking_fn = Partial(stacking, **params["stacking"])
    hb_fn = Partial(hydrogen_bonding, **params["hydrogen_bonding"])
    exc_vol_unbonded_params = params["excluded_volume"]
    exc_vol_bonded_params = deepcopy(exc_vol_unbonded_params)
    del exc_vol_bonded_params["dr_star_backbone"]
    del exc_vol_bonded_params["sigma_backbone"]
    del exc_vol_bonded_params["b_backbone"]
    del exc_vol_bonded_params["dr_c_backbone"]
    exc_vol_unbonded_fn = Partial(exc_vol_unbonded, **exc_vol_unbonded_params)
    exc_vol_bonded_fn = Partial(exc_vol_bonded, **exc_vol_bonded_params)

    cross_stacking_fn = Partial(cross_stacking, **params["cross_stacking"])
    coaxial_stacking_fn = Partial(coaxial_stacking, **params["coaxial_stacking"])

    # Extract relevant neighbor information and define our pairwise displacement function
    d = space.map_bond(partial(displacement_fn))
    nn_i = bonded_neighbors[:, 0]
    nn_j = bonded_neighbors[:, 1]
    op_i = unbonded_neighbors[:, 0]
    op_j = unbonded_neighbors[:, 1]

    def _compute_subterms(body: RigidBody, seq: util.Array, params):

        # Use our the parameters to construct the relevant energy functions
        # Note: for each, there are two options -- corresponding to whether we are optimizing over arrays or dicts
        ## FENE
        # fene_params = dict(zip(["eps_backbone", "delta_backbone", "r0_backbone"], params[:3]))
        fene_params = params["fene"]
        fene_fn = Partial(v_fene, **fene_params)

        """
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
        unprocessed_stacking_params = dict(zip(stacking_param_names, params[3:23]))
        stacking_params = process_stacking_params(unprocessed_stacking_params, kt)
        """
        stacking_params = get_params.process_stacking(params['stacking'], kt)
        stacking_fn = Partial(stacking, **stacking_params)


        # Compute relevant variables for our potential
        # Note: `_op` corresponds to "other pairs"
        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q)
        back_sites = body.center + com_to_backbone * back_base_vectors
        stack_sites = body.center + com_to_stacking * back_base_vectors
        base_sites = body.center + com_to_hb * back_base_vectors

        ## Fene variables
        dr_back = d(back_sites[nn_i], back_sites[nn_j]) # N x N x 3
        r_back = jnp.linalg.norm(dr_back, axis=1)

        ## Exc. vol bonded variables
        dr_base = d(base_sites[nn_i], base_sites[nn_j])
        dr_back_base = d(back_sites[nn_i], base_sites[nn_j])
        dr_base_back = d(base_sites[nn_i], back_sites[nn_j])

        ## Stacking variables
        dr_stack = d(stack_sites[nn_i], stack_sites[nn_j])
        r_stack = jnp.linalg.norm(dr_stack, axis=1)
        base_normals = Q_to_base_normal(Q) # space frame, normalized
        cross_prods = Q_to_cross_prod(Q) # space frame, normalized

        theta4 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nn_i], base_normals[nn_j])))
        theta5 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', dr_stack, base_normals[nn_j]) / r_stack))
        theta6 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nn_i], dr_stack) / r_stack))
        cosphi1 = -jnp.einsum('ij, ij->i', cross_prods[nn_i], dr_back) / r_back
        cosphi2 = -jnp.einsum('ij, ij->i', cross_prods[nn_j], dr_back) / r_back

        ## Exc. vol unbonded variables
        dr_base_op = d(base_sites[op_j], base_sites[op_i]) # Note the flip here
        dr_backbone_op = d(back_sites[op_j], back_sites[op_i]) # Note the flip here
        dr_back_base_op = d(back_sites[op_i], base_sites[op_j]) # Note: didn't flip this one (and others) because no need, but should look into at some point
        dr_base_back_op = d(base_sites[op_i], back_sites[op_j])

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

        ## Cross stacking variables -- all already computed

        ## Coaxial stacking
        dr_stack_op = d(stack_sites[op_j], stack_sites[op_i]) # note: reversed
        dr_stack_norm_op = dr_stack_op / jnp.linalg.norm(dr_stack_op, axis=1, keepdims=True)
        dr_backbone_norm_op = dr_backbone_op / jnp.linalg.norm(dr_backbone_op, axis=1, keepdims=True)
        theta5_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[op_i], dr_stack_norm_op)))
        theta6_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -base_normals[op_j], dr_stack_norm_op)))
        cosphi3_op = jnp.einsum('ij, ij->i', dr_stack_norm_op,
                                jnp.cross(dr_backbone_norm_op, back_bases[op_j]))
        cosphi4_op = jnp.einsum('ij, ij->i', dr_stack_norm_op,
                                jnp.cross(dr_backbone_norm_op, back_bases[op_i]))


        # Compute the contributions from each interaction
        fene_dg = fene_fn(r_back)
        exc_vol_bonded_dg = exc_vol_bonded_fn(dr_base, dr_back_base, dr_base_back)
        stack_dg = stacking_fn(r_stack, theta4, theta5, theta6, cosphi1, cosphi2)
        exc_vol_unbonded_dg = exc_vol_unbonded_fn(dr_base_op, dr_backbone_op, dr_back_base_op,
                                                  dr_base_back_op)
        v_hb = hb_fn(dr_base_op, theta1_op, theta2_op, theta3_op, theta4_op,
                     theta7_op, theta8_op)
        ## For HB, dot product to only be between appropriate bases
        hb_probs = get_hb_probs(seq, op_i, op_j) # get the probabilities of all possibile hydrogen bonds for all neighbors
        hb_weights = jnp.dot(hb_probs, HB_WEIGHTS)
        hb_dg = jnp.dot(hb_weights, v_hb)

        cr_stack_dg = cross_stacking_fn(r_base_op, theta1_op, theta2_op, theta3_op,
                                        theta4_op, theta7_op, theta8_op)
        cx_stack_dg = coaxial_stacking_fn(dr_stack_op, theta4_op, theta1_op, theta5_op,
                                          theta6_op, cosphi3_op, cosphi4_op)

        return (jnp.sum(fene_dg), jnp.sum(exc_vol_bonded_dg), jnp.sum(stack_dg), \
                jnp.sum(exc_vol_unbonded_dg), hb_dg, jnp.sum(cr_stack_dg),
                jnp.sum(cx_stack_dg)) # hb_dg is already a scalar

    def energy_fn(body: RigidBody, seq: util.Array, params, **kwargs) -> float:
        dgs = _compute_subterms(body, seq, params)
        fene_dg, b_exc_dg, stack_dg, n_exc_dg, hb_dg, cr_stack, cx_stack = dgs
        return fene_dg + b_exc_dg + stack_dg + n_exc_dg + hb_dg + cr_stack + cx_stack

    return energy_fn, _compute_subterms



if __name__ == "__main__":

    from loader.trajectory import TrajectoryInfo
    from loader.topology import TopologyInfo
    from utils import base_site, stack_site, back_site, get_one_hot

    top_path = "data/persistence-length/init.top"
    # conf_path = "data/persistence-length/init.conf"
    conf_path = "data/persistence-length/relaxed.dat"

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)

    body = config_info.states[0]

    displacement_fn, _ = space.periodic(config_info.box_size)
    pairs = top_info.bonded_nbrs

    energy_fn, compute_subterms =  energy_fn_factory(
        displacement_fn,
        back_site, stack_site, base_site,
        top_info.bonded_nbrs, top_info.unbonded_nbrs)

    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)

    # starting with the correct parameters
    init_fene_params = [2.0, 0.25, 0.7525]
    init_stacking_params = [
        1.3448, 2.6568, 6.0, 0.4, 0.9, 0.32, 0.75, # f1(dr_stack)
        1.30, 0.0, 0.8, # f4(theta_4)
        0.90, 0.0, 0.95, # f4(theta_5p)
        0.90, 0.0, 0.95, # f4(theta_6p)
        2.0, -0.65, # f5(-cos(phi1))
        2.0, -0.65 # f5(-cos(phi2))
    ]
    params = init_fene_params + init_stacking_params

    hi = compute_subterms(body, seq, params)

    pdb.set_trace()
