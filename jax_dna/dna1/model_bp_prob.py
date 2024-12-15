"""
need better testing

right now, i think only the 1st and 4th cases matter

should add things like the bulge case where something is still technically unpaired but it still has hydrogen bonding interact with things

have to do this from both directions

should also add some one-hot examples
"""

import pdb
import unittest
from functools import partial
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import numpy as onp
import itertools

import jax
from jax import jit, random, lax, grad, value_and_grad, vmap
import jax.numpy as jnp
from jax_md import space, simulate, rigid_body

from jax_dna.common.read_seq_specific import read_ss_oxdna
from jax_dna.common.utils import DEFAULT_TEMP, clamp
from jax_dna.common.utils import Q_to_back_base, Q_to_base_normal, Q_to_cross_prod
from jax_dna.common.interactions import v_fene_smooth, stacking, exc_vol_bonded, \
    exc_vol_unbonded, cross_stacking, coaxial_stacking, hydrogen_bonding
from jax_dna.common import utils, topology, trajectory
from jax_dna.dna1.load_params import load, _process

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)


BP_TYPES = ["AT", "TA", "GC", "CG"]
N_BP_TYPES = len(BP_TYPES)
N_NT = len(utils.DNA_ALPHA)
BP_IDXS = list()
for nt1, nt2 in BP_TYPES:
    BP_IDXS.append([utils.DNA_ALPHA.index(nt1), utils.DNA_ALPHA.index(nt2)])
BP_IDXS = jnp.array(BP_IDXS)


DEFAULT_BASE_PARAMS = load(process=False) # Note: only processing depends on temperature
EMPTY_BASE_PARAMS = {
    "fene": dict(),
    "excluded_volume": dict(),
    "stacking": dict(),
    "hydrogen_bonding": dict(),
    "cross_stacking": dict(),
    "coaxial_stacking": dict()
}
com_to_stacking = 0.34
com_to_hb = 0.4
com_to_backbone = -0.4

def add_coupling(base_params):
    # Stacking
    base_params["stacking"]["a_stack_6"] = base_params["stacking"]["a_stack_5"]
    base_params["stacking"]["theta0_stack_6"] = base_params["stacking"]["theta0_stack_5"]
    base_params["stacking"]["delta_theta_star_stack_6"] = base_params["stacking"]["delta_theta_star_stack_5"]

    # Hydrogen Bonding
    base_params["hydrogen_bonding"]["a_hb_3"] = base_params["hydrogen_bonding"]["a_hb_2"]
    base_params["hydrogen_bonding"]["theta0_hb_3"] = base_params["hydrogen_bonding"]["theta0_hb_2"]
    base_params["hydrogen_bonding"]["delta_theta_star_hb_3"] = base_params["hydrogen_bonding"]["delta_theta_star_hb_2"]

    base_params["hydrogen_bonding"]["a_hb_8"] = base_params["hydrogen_bonding"]["a_hb_7"]
    base_params["hydrogen_bonding"]["theta0_hb_8"] = base_params["hydrogen_bonding"]["theta0_hb_7"]
    base_params["hydrogen_bonding"]["delta_theta_star_hb_8"] = base_params["hydrogen_bonding"]["delta_theta_star_hb_7"]

    # Cross stacking
    base_params["cross_stacking"]["a_cross_3"] = base_params["cross_stacking"]["a_cross_2"]
    base_params["cross_stacking"]["theta0_cross_3"] = base_params["cross_stacking"]["theta0_cross_2"]
    base_params["cross_stacking"]["delta_theta_star_cross_3"] = base_params["cross_stacking"]["delta_theta_star_cross_2"]

def get_full_base_params(override_base_params):
    fene_params = DEFAULT_BASE_PARAMS["fene"] | override_base_params["fene"]
    exc_vol_params = DEFAULT_BASE_PARAMS["excluded_volume"] | override_base_params["excluded_volume"]
    stacking_params = DEFAULT_BASE_PARAMS["stacking"] | override_base_params["stacking"]
    hb_params = DEFAULT_BASE_PARAMS["hydrogen_bonding"] | override_base_params["hydrogen_bonding"]
    cr_params = DEFAULT_BASE_PARAMS["cross_stacking"] | override_base_params["cross_stacking"]
    cx_params = DEFAULT_BASE_PARAMS["coaxial_stacking"] | override_base_params["coaxial_stacking"]

    base_params = {
        "fene": fene_params,
        "excluded_volume": exc_vol_params,
        "stacking": stacking_params,
        "hydrogen_bonding": hb_params,
        "cross_stacking": cr_params,
        "coaxial_stacking": cx_params
    }
    add_coupling(base_params)
    return base_params


class EnergyModel:
    def __init__(self, displacement_fn, bps, unpaired, is_unpaired, idx_to_unpaired_idx, idx_to_bp_idx,
                 override_base_params=EMPTY_BASE_PARAMS, t_kelvin=DEFAULT_TEMP,
                 ss_hb_weights=utils.HB_WEIGHTS_SA, ss_stack_weights=utils.STACK_WEIGHTS_SA
    ):
        self.displacement_fn = displacement_fn
        self.displacement_mapped = jit(space.map_bond(partial(displacement_fn)))
        self.t_kelvin = t_kelvin

        self.ss_hb_weights = ss_hb_weights
        self.ss_hb_weights_flat = self.ss_hb_weights.flatten()

        self.ss_stack_weights = ss_stack_weights
        self.ss_stack_weights_flat = self.ss_stack_weights.flatten()

        self.base_params = get_full_base_params(override_base_params)
        self.params = _process(self.base_params, self.t_kelvin)

        self.bps = bps
        self.n_bps = bps.shape[0]
        self.unpaired = unpaired
        self.n_unpaired = unpaired.shape[0]

        self.is_unpaired = is_unpaired
        self.idx_to_unpaired_idx = idx_to_unpaired_idx
        self.idx_to_bp_idx = idx_to_bp_idx

    def compute_subterms(self, body, unpaired_pseq, bp_pseq, seq, bonded_nbrs, unbonded_nbrs):
        nn_i = bonded_nbrs[:, 0]
        nn_j = bonded_nbrs[:, 1]

        op_i = unbonded_nbrs[0]
        op_j = unbonded_nbrs[1]
        mask = jnp.array(op_i < body.center.shape[0], dtype=jnp.int32)

        # Compute relevant variables for our potential
        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q) # space frame, normalized
        base_normals = Q_to_base_normal(Q) # space frame, normalized
        cross_prods = Q_to_cross_prod(Q) # space frame, normalized

        back_sites = body.center + com_to_backbone * back_base_vectors
        stack_sites = body.center + com_to_stacking * back_base_vectors
        base_sites = body.center + com_to_hb * back_base_vectors

        ## Fene variables
        dr_back_nn = self.displacement_mapped(back_sites[nn_i], back_sites[nn_j]) # N x N x 3
        r_back_nn = jnp.linalg.norm(dr_back_nn, axis=1)

        ## Exc. vol bonded variables
        dr_base_nn = self.displacement_mapped(base_sites[nn_i], base_sites[nn_j])
        dr_back_base_nn = self.displacement_mapped(back_sites[nn_i], base_sites[nn_j])
        dr_base_back_nn = self.displacement_mapped(base_sites[nn_i], back_sites[nn_j])

        ## Stacking variables
        dr_stack_nn = self.displacement_mapped(stack_sites[nn_i], stack_sites[nn_j])
        r_stack_nn = jnp.linalg.norm(dr_stack_nn, axis=1)

        theta4 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nn_i], base_normals[nn_j])))
        theta5 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', dr_stack_nn, base_normals[nn_j]) / r_stack_nn))
        theta6 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nn_i], dr_stack_nn) / r_stack_nn))
        # theta6 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', dr_stack_nn, base_normals[nn_j]) / r_stack_nn))
        # theta5 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nn_i], dr_stack_nn) / r_stack_nn))
        cosphi1 = -jnp.einsum('ij, ij->i', cross_prods[nn_i], dr_back_nn) / r_back_nn
        cosphi2 = -jnp.einsum('ij, ij->i', cross_prods[nn_j], dr_back_nn) / r_back_nn

        ## Exc. vol unbonded variables
        dr_base_op = self.displacement_mapped(base_sites[op_j], base_sites[op_i]) # Note the flip here
        dr_backbone_op = self.displacement_mapped(back_sites[op_j], back_sites[op_i]) # Note the flip here
        dr_back_base_op = self.displacement_mapped(back_sites[op_i], base_sites[op_j]) # Note: didn't flip this one (and others) because no need, but should look into at some point
        dr_base_back_op = self.displacement_mapped(base_sites[op_i], back_sites[op_j])

        ## Hydrogen bonding
        r_base_op = jnp.linalg.norm(dr_base_op, axis=1)
        theta1_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -back_base_vectors[op_i], back_base_vectors[op_j])))
        theta2_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -back_base_vectors[op_j], dr_base_op) / r_base_op))
        theta3_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', back_base_vectors[op_i], dr_base_op) / r_base_op))
        theta4_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[op_i], base_normals[op_j])))
        # note: are these swapped in Lorenzo's code?
        theta7_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -base_normals[op_j], dr_base_op) / r_base_op))
        # theta8_op = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[op_i], dr_base_op) / r_base_op))
        theta8_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[op_i], dr_base_op) / r_base_op))

        ## Cross stacking variables -- all already computed

        ## Coaxial stacking
        dr_stack_op = self.displacement_mapped(stack_sites[op_j], stack_sites[op_i]) # note: reversed
        dr_stack_norm_op = dr_stack_op / jnp.linalg.norm(dr_stack_op, axis=1, keepdims=True)
        dr_backbone_norm_op = dr_backbone_op / jnp.linalg.norm(dr_backbone_op, axis=1, keepdims=True)
        theta5_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[op_i], dr_stack_norm_op)))
        theta6_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -base_normals[op_j], dr_stack_norm_op)))
        cosphi3_op = jnp.einsum('ij, ij->i', dr_stack_norm_op,
                                jnp.cross(dr_backbone_norm_op, back_base_vectors[op_j]))
        cosphi4_op = jnp.einsum('ij, ij->i', dr_stack_norm_op,
                                jnp.cross(dr_backbone_norm_op, back_base_vectors[op_i]))

        # Compute the contributions from each interaction
        fene_dg = v_fene_smooth(r_back_nn, **self.params["fene"]).sum()
        exc_vol_bonded_dg = exc_vol_bonded(dr_base_nn, dr_back_base_nn, dr_base_back_nn,
                                           **self.params["excluded_volume_bonded"]).sum()

        v_stack = stacking(r_stack_nn, theta4, theta5, theta6, cosphi1, cosphi2, **self.params["stacking"])
        # stack_probs = utils.get_pair_probs(seq, nn_i, nn_j)
        # stack_weights = jnp.dot(stack_probs, self.ss_stack_weights_flat)


        def compute_seq_dep_weight(op1, op2, weights_table, flattened_weights_table):
            op1_unpaired = self.is_unpaired[op1]
            op2_unpaired = self.is_unpaired[op2]

            # Case 1: Both unpaired
            pair_probs = jnp.kron(unpaired_pseq[self.idx_to_unpaired_idx[op1]], unpaired_pseq[self.idx_to_unpaired_idx[op2]])
            pair_weight_unpaired = jnp.dot(pair_probs, flattened_weights_table)

            # Case 2: op1 unpaired, op2 base paired
            op1_nt_probs = unpaired_pseq[self.idx_to_unpaired_idx[op1]]
            op2_bp_idx, within_op2_bp_idx = self.idx_to_bp_idx[op2]
            bp_probs = bp_pseq[op2_bp_idx]

            def op1_up_fn(op1_nt, op2_bp_type_idx):
                op2_nt = BP_IDXS[op2_bp_type_idx][within_op2_bp_idx]
                return op1_nt_probs[op1_nt] * bp_probs[op2_bp_type_idx] * weights_table[op1_nt, op2_nt]
            pair_weight_op1_up = vmap(vmap(op1_up_fn, (None, 0)), (0, None))(jnp.arange(N_NT), jnp.arange(N_BP_TYPES)).sum()

            # Case 3: op2 unpaired, op1 base paired

            op2_nt_probs = unpaired_pseq[self.idx_to_unpaired_idx[op2]]
            op1_bp_idx, within_op1_bp_idx = self.idx_to_bp_idx[op1]
            bp_probs = bp_pseq[op1_bp_idx]

            def op2_up_fn(op2_nt, op1_bp_type_idx):
                op1_nt = BP_IDXS[op1_bp_type_idx][within_op1_bp_idx]
                return op2_nt_probs[op2_nt] * bp_probs[op1_bp_type_idx] * weights_table[op1_nt, op2_nt]
            pair_weight_op2_up = vmap(vmap(op2_up_fn, (None, 0)), (0, None))(jnp.arange(N_NT), jnp.arange(N_BP_TYPES)).sum()

            # Case 4: both op1 and op2 are base paired

            op1_bp_idx, within_op1_bp_idx = self.idx_to_bp_idx[op1]
            op2_bp_idx, within_op2_bp_idx = self.idx_to_bp_idx[op2]

            ## Case 4.I: op1 and op2 are in the same base pair
            bp_probs = bp_pseq[op1_bp_idx]
            def same_bp_fn(bp_idx):
                bp_prob = bp_probs[bp_idx]
                bp_nt1, bp_nt2 = BP_IDXS[bp_idx][jnp.array([within_op1_bp_idx, within_op2_bp_idx])]
                return bp_prob * weights_table[bp_nt1, bp_nt2]
            pair_weight_same_bp = vmap(same_bp_fn)(jnp.arange(N_BP_TYPES)).sum()

            ## Case 4.II: op1 and op2 are in different base pairs
            bp1_probs = bp_pseq[op1_bp_idx]
            bp2_probs = bp_pseq[op2_bp_idx]
            def diff_bps_fn(bp1_idx, bp2_idx):
                bp1_prob = bp1_probs[bp1_idx]
                op1_nt = BP_IDXS[bp1_idx][within_op1_bp_idx]

                bp2_prob = bp2_probs[bp2_idx]
                op2_nt = BP_IDXS[bp2_idx][within_op2_bp_idx]

                return bp1_prob * bp2_prob * weights_table[op1_nt, op2_nt]
            pair_weight_diff_bps = vmap(vmap(diff_bps_fn, (None, 0)), (0, None))(jnp.arange(N_BP_TYPES), jnp.arange(N_BP_TYPES)).sum()

            pair_weight_both_paired = jnp.where(op1_bp_idx == op2_bp_idx, pair_weight_same_bp, pair_weight_diff_bps)

            return jnp.where(jnp.logical_and(op1_unpaired, op2_unpaired), pair_weight_unpaired,
                             jnp.where(op1_unpaired, pair_weight_op1_up,
                                       jnp.where(op2_unpaired, pair_weight_op2_up, pair_weight_both_paired)))

        stack_weights = vmap(compute_seq_dep_weight, (0, 0, None, None))(nn_i, nn_j, self.ss_stack_weights, self.ss_stack_weights_flat)
        stack_dg = jnp.dot(stack_weights, v_stack)
        # stack_dg = stacking(r_stack_nn, theta4, theta5, theta6, cosphi1, cosphi2, **self.params["stacking"]).sum()

        exc_vol_unbonded_dg = exc_vol_unbonded(
            dr_base_op, dr_backbone_op, dr_back_base_op, dr_base_back_op,
            **self.params["excluded_volume"]
        )
        exc_vol_unbonded_dg = jnp.where(mask, exc_vol_unbonded_dg, 0.0).sum() # Mask for neighbors


        v_hb = hydrogen_bonding(
            dr_base_op, theta1_op, theta2_op, theta3_op, theta4_op,
            theta7_op, theta8_op, **self.params["hydrogen_bonding"])

        v_hb = jnp.where(mask, v_hb, 0.0) # Mask for neighbors
        hb_weights = vmap(compute_seq_dep_weight, (0, 0, None, None))(op_i, op_j, self.ss_hb_weights, self.ss_hb_weights_flat)
        hb_dg = jnp.dot(hb_weights, v_hb)



        cr_stack_dg = cross_stacking(
            r_base_op, theta1_op, theta2_op, theta3_op,
            theta4_op, theta7_op, theta8_op, **self.params["cross_stacking"])
        cr_stack_dg = jnp.where(mask, cr_stack_dg, 0.0).sum() # Mask for neighbors

        cx_stack_dg = coaxial_stacking(
            dr_stack_op, theta4_op, theta1_op, theta5_op,
            theta6_op, cosphi3_op, cosphi4_op, **self.params["coaxial_stacking"])
        cx_stack_dg = jnp.where(mask, cx_stack_dg, 0.0).sum() # Mask for neighbors

        return fene_dg, exc_vol_bonded_dg, stack_dg, \
            exc_vol_unbonded_dg, hb_dg, cr_stack_dg, cx_stack_dg

    def energy_fn(self, body, unpaired_pseq, bp_pseq, seq, bonded_nbrs, unbonded_nbrs):
        dgs = self.compute_subterms(body, unpaired_pseq, bp_pseq, seq, bonded_nbrs, unbonded_nbrs)
        fene_dg, b_exc_dg, stack_dg, n_exc_dg, hb_dg, cr_stack, cx_stack = dgs
        return fene_dg + b_exc_dg + stack_dg + n_exc_dg + hb_dg + cr_stack + cx_stack


class TestDna1(unittest.TestCase):
    test_data_basedir = Path("data/test-data")


    def test_brute_force_new(self):
        from jax_dna.dna1 import model as orig_model

        ss_path = "data/seq-specific/seq_oxdna1.txt"

        ss_hb_weights, ss_stack_weights = read_ss_oxdna(ss_path)
        # ss_hb_weights, ss_stack_weights = utils.HB_WEIGHTS_SA, utils.STACK_WEIGHTS_SA

        # ss_hb_weights, _ = read_ss_oxdna(ss_path)
        # _, ss_stack_weights = utils.HB_WEIGHTS_SA, utils.STACK_WEIGHTS_SA

        ss_hb_weights = jnp.array(ss_hb_weights)
        ss_stack_weights = jnp.array(ss_stack_weights)

        basedir = self.test_data_basedir / "helix-4bp"
        t_kelvin = utils.DEFAULT_TEMP

        top_path = basedir / "sys.top"
        if not top_path.exists():
            raise RuntimeError(f"No topology file at location: {top_path}")
        traj_path = basedir / "output.dat"
        if not traj_path.exists():
            raise RuntimeError(f"No trajectory file at location: {traj_path}")

        top_info = topology.TopologyInfo(top_path, reverse_direction=False)
        # seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
        n = len(top_info.seq)
        assert(n == 8)
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
        traj_states = traj_info.get_states()

        displacement_fn, shift_fn = space.periodic(traj_info.box_size)
        model_base = orig_model.EnergyModel(displacement_fn, t_kelvin=t_kelvin,
                                            ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights)

        neighbors_idx = top_info.unbonded_nbrs.T

        n_eval_strucs = 1

        energy_fn_base = model_base.energy_fn
        energy_fn_base = jit(energy_fn_base)

        bps = jnp.array([
            [0, 7],
            [1, 6],
            # [2, 5], we should ignore one
            [3, 4]
        ])
        n_bps = bps.shape[0]
        bp_logits = onp.random.rand(n_bps, 4)
        bp_pseq = bp_logits / bp_logits.sum(axis=1, keepdims=True)
        bp_pseq = jnp.array(bp_pseq)

        unpaired = jnp.array([2, 5])
        is_unpaired = jnp.array([(i in set(onp.array(unpaired))) for i in range(n)]).astype(jnp.int32)
        n_unpaired = unpaired.shape[0]
        unpaired_logits = onp.random.rand(n_unpaired, 4)
        unpaired_pseq = unpaired_logits / unpaired_logits.sum(axis=1, keepdims=True)
        unpaired_pseq = jnp.array(unpaired_pseq)


        idx_to_unpaired_idx = onp.arange(n)
        for up_idx, idx in enumerate(unpaired):
            idx_to_unpaired_idx[idx] = up_idx
        idx_to_unpaired_idx = jnp.array(idx_to_unpaired_idx)


        # idx_to_bp_idx = onp.arange(n)
        idx_to_bp_idx = onp.zeros((n, 2), dtype=onp.int32)
        for bp_idx, (nt1, nt2) in enumerate(bps):
            idx_to_bp_idx[nt1] = [bp_idx, 0]
            idx_to_bp_idx[nt2] = [bp_idx, 1]
        idx_to_bp_idx = jnp.array(idx_to_bp_idx)


        logits = onp.random.rand(n, 4)
        pseq = logits / logits.sum(axis=1, keepdims=True)
        pseq = jnp.array(pseq)

        model = EnergyModel(displacement_fn, bps, unpaired, is_unpaired, idx_to_unpaired_idx, idx_to_bp_idx,
                            t_kelvin=t_kelvin, ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights)
        energy_fn = model.energy_fn
        # energy_fn = jit(energy_fn)

        def sequence_probability(sequence, unpaired_pseq, bp_pseq):
            # Initialize probability to 1 (neutral for multiplication)
            probability = 1.0

            for n_up_idx, up_idx in enumerate(unpaired):
                up_nt_idx = utils.DNA_ALPHA.index(sequence[up_idx])
                probability *= unpaired_pseq[n_up_idx, up_nt_idx]

            for bp_idx, (nt1, nt2) in enumerate(bps):
                bp_type_idx = BP_TYPES.index(sequence[nt1] + seq[nt2])
                probability *= bp_pseq[bp_idx, bp_type_idx]

            return probability



        for struc_idx in range(n_eval_strucs):
            state = traj_states[struc_idx]

            expected_energy_calc = energy_fn(
                state, unpaired_pseq, bp_pseq, pseq, top_info.bonded_nbrs, neighbors_idx)

            expected_energy_brute = 0.0

            ## Get the sequences
            assert(len(BP_TYPES) == len(utils.DNA_ALPHA))
            all_seq_idxs = itertools.product(onp.arange(4), repeat=n_unpaired + n_bps)

            for seq_idxs in tqdm(all_seq_idxs, desc="Seq idxs."):
                sampled_unpaired_seq_idxs = seq_idxs[:n_unpaired]
                sampled_bp_type_idxs = seq_idxs[n_unpaired:]

                seq = ["X"] * n
                for unpaired_idx, nt_idx in zip(unpaired, sampled_unpaired_seq_idxs):
                    seq[unpaired_idx] = utils.DNA_ALPHA[nt_idx]

                for (nt1_idx, nt2_idx), bp_type_idx in zip(bps, sampled_bp_type_idxs):
                    bp1, bp2 = BP_TYPES[bp_type_idx]
                    seq[nt1_idx] = bp1
                    seq[nt2_idx] = bp2

                seq = ''.join(seq)

                seq_oh = jnp.array(utils.get_one_hot(seq), dtype=jnp.float64)
                seq_energy_calc = energy_fn_base(
                    state, seq_oh, top_info.bonded_nbrs, neighbors_idx)
                seq_prob = sequence_probability(seq, unpaired_pseq, bp_pseq)

                expected_energy_brute += seq_prob*seq_energy_calc

        pdb.set_trace()






if __name__ == "__main__":
    unittest.main()
