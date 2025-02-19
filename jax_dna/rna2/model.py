import pdb
import unittest
from functools import partial
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import numpy as onp
from io import StringIO

import jax
jax.config.update("jax_enable_x64", True)
from jax import jit, random, lax, grad, value_and_grad, vmap
import jax.numpy as jnp
from jax_md import space, simulate, rigid_body

from jax_dna.common.read_seq_specific import read_ss_oxdna
from jax_dna.common.utils import clamp
from jax_dna.common.base_functions import v_fene
from jax_dna.common.interactions import v_fene_smooth, stacking2, exc_vol_bonded, \
    exc_vol_unbonded, cross_stacking2, coaxial_stacking, hydrogen_bonding, \
    coaxial_stacking3, coaxial_stacking4
from jax_dna.common import utils, topology, trajectory, smoothing
from jax_dna.rna2.load_params import load, _process, read_seq_specific, \
    DEFAULT_BASE_PARAMS, EMPTY_BASE_PARAMS, get_full_base_params



class EnergyModel:
    def __init__(
        self,
        displacement_fn,
        override_base_params=EMPTY_BASE_PARAMS,
        t_kelvin=utils.DEFAULT_TEMP,
        ss_hb_weights=utils.HB_WEIGHTS_SA,
        ss_stack_weights=utils.STACK_WEIGHTS_SA,
        salt_conc=0.5,
        q_eff=0.815,
        use_symm_coax=False
    ):
        self.displacement_fn = displacement_fn
        self.displacement_mapped = jit(space.map_bond(partial(displacement_fn)))
        self.t_kelvin = t_kelvin
        self.salt_conc = salt_conc
        kT = utils.get_kt(self.t_kelvin)

        self.use_symm_coax = use_symm_coax

        self.ss_hb_weights = ss_hb_weights
        self.ss_hb_weights_flat = self.ss_hb_weights.flatten()

        self.ss_stack_weights = ss_stack_weights
        self.ss_stack_weights_flat = self.ss_stack_weights.flatten()

        self.base_params = get_full_base_params(override_base_params)
        self.params = _process(self.base_params, self.t_kelvin, self.salt_conc)

        if self.use_symm_coax:
            b_coax_1_bonus, delta_theta_coax_1_c_bonus = smoothing.get_f4_smoothing_params(
                self.params['coaxial_stacking']['a_coax_1'],
                self.params['coaxial_stacking']['theta0_coax_1_bonus'],
                self.params['coaxial_stacking']['delta_theta_star_coax_1'])
            self.params['coaxial_stacking']['b_coax_1_bonus'] = b_coax_1_bonus
            self.params['coaxial_stacking']['delta_theta_coax_1_c_bonus'] = delta_theta_coax_1_c_bonus

        self.com_to_backbone_x = self.params["geometry"]["pos_back_a1"]
        self.com_to_backbone_y = self.params["geometry"]["pos_back_a3"]
        self.com_to_stacking = self.params["geometry"]["pos_stack"]
        self.com_to_hb = self.params["geometry"]["pos_base"]

        self.p3_x = self.params["geometry"]["p3_x"]
        self.p3_y = self.params["geometry"]["p3_y"]
        self.p3_z = self.params["geometry"]["p3_z"]
        self.p5_x = self.params["geometry"]["p5_x"]
        self.p5_y = self.params["geometry"]["p5_y"]
        self.p5_z = self.params["geometry"]["p5_z"]

        self.pos_stack_3_a1 = self.params["geometry"]["pos_stack_3_a1"]
        self.pos_stack_3_a2 = self.params["geometry"]["pos_stack_3_a2"]
        self.pos_stack_5_a1 = self.params["geometry"]["pos_stack_5_a1"]
        self.pos_stack_5_a2 = self.params["geometry"]["pos_stack_5_a2"]


    def compute_subterms(self, body, seq, bonded_nbrs, unbonded_nbrs, is_end=None, unique_hb_pairs=None):
        nn_i = bonded_nbrs[:, 0]
        nn_j = bonded_nbrs[:, 1]

        op_i = unbonded_nbrs[0]
        op_j = unbonded_nbrs[1]
        mask = jnp.array(op_i < body.center.shape[0], dtype=jnp.int32)

        if is_end is None:
            dh_mults = jnp.ones(op_i.shape[0])
        else:
            dh_mults_op_i = jnp.where(is_end[op_i], 0.5, 1.0)
            dh_mults_op_j = jnp.where(is_end[op_j], 0.5, 1.0)
            dh_mults = jnp.multiply(dh_mults_op_i, dh_mults_op_j)

        # Compute relevant variables for our potential
        Q = body.orientation
        back_base_vectors = utils.Q_to_back_base(Q) # space frame, normalized
        base_normals = utils.Q_to_base_normal(Q) # space frame, normalized
        cross_prods = utils.Q_to_cross_prod(Q) # space frame, normalized

        back_sites = body.center + self.com_to_backbone_x*back_base_vectors + self.com_to_backbone_y*base_normals
        stack_sites = body.center + self.com_to_stacking * back_base_vectors
        base_sites = body.center + self.com_to_hb * back_base_vectors

        # bb_p3_sites = body.center + self.p3_x*back_base_vectors + self.p3_y*cross_prods + self.p3_z*base_normals
        # bb_p3_sites = self.p3_x*back_base_vectors # + self.p3_y*cross_prods + self.p3_z*base_normals
        bb_p3_sites = self.p3_x*back_base_vectors + self.p3_y*cross_prods + self.p3_z*base_normals
        # bb_p5_sites = body.center + self.p5_x*back_base_vectors # + self.p5_y*cross_prods + self.p5_z*base_normals
        bb_p5_sites = self.p5_x*back_base_vectors + self.p5_y*cross_prods + self.p5_z*base_normals

        stack3_sites = body.center + self.pos_stack_3_a1*back_base_vectors + self.pos_stack_3_a2*cross_prods
        stack5_sites = body.center + self.pos_stack_5_a1*back_base_vectors + self.pos_stack_5_a2*cross_prods


        ## Fene variables
        dr_back_nn = self.displacement_mapped(back_sites[nn_i], back_sites[nn_j]) # N x N x 3
        r_back_nn = jnp.linalg.norm(dr_back_nn, axis=1)

        ## Exc. vol bonded variables
        dr_base_nn = self.displacement_mapped(base_sites[nn_i], base_sites[nn_j])
        dr_back_base_nn = self.displacement_mapped(back_sites[nn_i], base_sites[nn_j])
        dr_base_back_nn = self.displacement_mapped(base_sites[nn_i], back_sites[nn_j])

        ## Stacking variables
        dr_stack_nn = self.displacement_mapped(stack5_sites[nn_i], stack3_sites[nn_j])
        r_stack_nn = jnp.linalg.norm(dr_stack_nn, axis=1)

        theta4 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nn_i], base_normals[nn_j])))
        theta5 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', dr_stack_nn, base_normals[nn_j]) / r_stack_nn))
        theta6 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nn_i], dr_stack_nn) / r_stack_nn))


        # theta9 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', bb_p3_sites[nn_i], dr_back_nn) / r_back_nn))
        theta9 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', dr_back_nn, bb_p3_sites[nn_i]) / r_back_nn))
        # costB1 = jnp.einsum('ij, ij->i', dr_back_nn, bb_p3_sites[nn_j]) / r_back_nn
        # costB1 = bb_p3_sites[nn_j][:, 2]
        # costB1 = (dr_back_nn / jnp.expand_dims(r_back_nn, axis=1))[:, 2]
        costB1 = jnp.einsum('ij, ij->i', -bb_p3_sites[nn_j], dr_back_nn) / r_back_nn
        # costB1 = vmap(jnp.dot, (0, 0))(-bb_p3_sites[nn_j], (dr_back_nn / jnp.expand_dims(r_back_nn, axis=1)))
        # theta9 = jnp.arccos(costB1)
        theta9 = jnp.arccos(clamp(costB1))
        # theta10 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', bb_p5_sites[nn_j], dr_back_nn) / r_back_nn))
        costB2 = jnp.einsum('ij, ij->i', -bb_p5_sites[nn_i], dr_back_nn) / r_back_nn
        # theta10 = jnp.arccos(costB2)
        theta10 = jnp.arccos(clamp(costB2))

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
        # fene_dg = v_fene(r_back_nn, self.params["fene"]["eps_backbone"],
        #                  self.params["fene"]["r0_backbone"],
        #                  self.params["fene"]["delta_backbone"]).sum()
        exc_vol_bonded_dg = exc_vol_bonded(dr_base_nn, dr_back_base_nn, dr_base_back_nn,
                                           **self.params["excluded_volume_bonded"]).sum()

        v_stack = stacking2(r_stack_nn, theta5, theta6, theta9, theta10, cosphi1, cosphi2, **self.params["stacking"])
        stack_probs = utils.get_pair_probs(seq, nn_i, nn_j)
        stack_weights = jnp.dot(stack_probs, self.ss_stack_weights_flat)
        stack_dg = jnp.dot(stack_weights, v_stack)
        # stack_dg = costB1.sum()
        # stack_dg = v_stack.sum()

        exc_vol_unbonded_dg = exc_vol_unbonded(
            dr_base_op, dr_backbone_op, dr_back_base_op, dr_base_back_op,
            **self.params["excluded_volume"]
        )
        exc_vol_unbonded_dg = jnp.where(mask, exc_vol_unbonded_dg, 0.0).sum() # Mask for neighbors

        v_hb = hydrogen_bonding(
            dr_base_op, theta1_op, theta2_op, theta3_op, theta4_op,
            theta7_op, theta8_op, **self.params["hydrogen_bonding"])
        v_hb = jnp.where(mask, v_hb, 0.0) # Mask for neighbors
        hb_probs = utils.get_pair_probs(seq, op_i, op_j) # get the probabilities of all possibile hydrogen bonds for all neighbors
        hb_weights = jnp.dot(hb_probs, self.ss_hb_weights_flat)
        if unique_hb_pairs is not None:
            unique_weights = vmap(lambda i, j: unique_hb_pairs[i, j], (0, 0))(op_i, op_j)
            hb_weights = jnp.multiply(hb_weights, unique_weights)
        hb_dg = jnp.dot(hb_weights, v_hb)

        cr_stack_dg = cross_stacking2(
            r_base_op, theta1_op, theta2_op, theta3_op,
            theta7_op, theta8_op, **self.params["cross_stacking"])
        cr_stack_dg = jnp.where(mask, cr_stack_dg, 0.0).sum() # Mask for neighbors

        if self.use_symm_coax:
            # cx_stack_dg = coaxial_stacking3(
            cx_stack_dg = coaxial_stacking4(
                dr_stack_op, theta4_op, theta1_op, theta5_op,
                theta6_op, cosphi3_op, cosphi4_op, **self.params["coaxial_stacking"])
        else:
            cx_stack_dg = coaxial_stacking(
                dr_stack_op, theta4_op, theta1_op, theta5_op,
                theta6_op, cosphi3_op, cosphi4_op, **self.params["coaxial_stacking"])
        cx_stack_dg = jnp.where(mask, cx_stack_dg, 0.0).sum() # Mask for neighbors

        # Compute debye_dg
        r_back_op = jnp.linalg.norm(dr_backbone_op, axis=1)
        def db_term(r):
            energy_full = jnp.exp(r * self.params['debye']['minus_kappa']) \
                          * (self.params['debye']['prefactor'] / r)
            energy_smooth =  self.params['debye']['B'] \
                             * (r - self.params['debye']['rcut'])**2
            cond = r < self.params['debye']['r_high']
            energy = jnp.where(cond, energy_full, energy_smooth)
            return jnp.where(r < self.params['debye']['rcut'], energy, 0.0)
        db_dgs = vmap(db_term)(r_back_op)
        db_dgs = jnp.where(mask, db_dgs, 0.0)
        db_dg = jnp.dot(db_dgs, dh_mults)


        return fene_dg, exc_vol_bonded_dg, stack_dg, exc_vol_unbonded_dg, hb_dg, cr_stack_dg, cx_stack_dg, db_dg


    def energy_fn(self, body, seq, bonded_nbrs, unbonded_nbrs, is_end=None, unique_hb_pairs=None):
        dgs = self.compute_subterms(body, seq, bonded_nbrs, unbonded_nbrs, is_end, unique_hb_pairs)
        fene_dg, exc_vol_bonded_dg, stack_dg, exc_vol_unbonded_dg, hb_dg, cr_stack_dg, cx_stack_dg, db_dg = dgs
        val = fene_dg + stack_dg + exc_vol_unbonded_dg + hb_dg + cr_stack_dg + cx_stack_dg + db_dg
        return val

class TestRna2(unittest.TestCase):
    test_data_basedir = Path("data/test-data")


    def check_energy_subterms(
        self,
        basedir,
        top_fname,
        traj_fname,
        t_kelvin,
        salt_conc,
        r_cutoff=10.0,
        dr_threshold=0.2,
        tol_places=4,
        verbose=True,
        avg_seq=True,
        hb_tol_places=3,
        params=None,
        use_neighbors=False,
        half_charged_ends=False,
        is_circle=False,
        unique_topology=False
    ):


        if avg_seq:
            ss_hb_weights = utils.HB_WEIGHTS_SA
            ss_stack_weights = utils.STACK_WEIGHTS_SA
        else:
            ss_hb_weights, ss_stack_weights, ss_cross_weights = read_seq_specific(DEFAULT_BASE_PARAMS)

        print(f"\n---- Checking energy breakdown agreement for base directory: {basedir} ----")

        basedir = Path(basedir)
        if not basedir.exists():
            raise RuntimeError(f"No directory exists at location: {basedir}")

        # First, load the oxDNA subterms
        split_energy_fname = basedir / "split_energy.dat"
        if not split_energy_fname.exists():
            raise RuntimeError(f"No energy subterm file exists at location: {split_energy_fname}")
        split_energy_df = pd.read_csv(
            split_energy_fname,
            names=["t", "fene", "b_exc", "stack", "n_exc", "hb",
                   "cr_stack", "cx_stack", "debye"],
                   # "cr_stack", "cx_stack"],
            delim_whitespace=True)
        oxdna_subterms = split_energy_df.iloc[1:, :]

        # Then, compute subterms via our energy model
        top_path = basedir / top_fname
        if not top_path.exists():
            raise RuntimeError(f"No topology file at location: {top_path}")
        traj_path = basedir / traj_fname
        if not traj_path.exists():
            raise RuntimeError(f"No trajectory file at location: {traj_path}")

        ## note: we don't reverse direction to keep ordering the same
        top_info = topology.TopologyInfo(
            top_path,
            reverse_direction=False,
            is_rna=True,
            allow_circle=is_circle,
            allow_arbitrary_alphabet=unique_topology
        )
        if half_charged_ends:
            is_end = top_info.is_end
        else:
            is_end = None
        if unique_topology:
            pairs = onp.argwhere(top_info.unique_hb_pairs == 1)
            unique_hb_pairs = {tuple(sorted(pair)) for pair in pairs}
            dummy_seq = ["A" for _ in range(top_info.n)]
            for i, j in unique_hb_pairs:
                dummy_seq[j] = "U"
            dummy_seq = ''.join(dummy_seq)
            seq_oh = jnp.array(utils.get_one_hot(dummy_seq, is_rna=True), dtype=jnp.float64)
            unique_hb_pairs = jnp.array(top_info.unique_hb_pairs)
        else:
            seq_oh = jnp.array(utils.get_one_hot(top_info.seq, is_rna=True), dtype=jnp.float64)
            unique_hb_pairs = None
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
        traj_states = traj_info.get_states()

        displacement_fn, shift_fn = space.periodic(traj_info.box_size)
        if params is None:
            params = deepcopy(EMPTY_BASE_PARAMS)
        model = EnergyModel(displacement_fn, params, t_kelvin=t_kelvin, salt_conc=salt_conc,
                            ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights)

        ## setup neighbors, if necessary
        if use_neighbors:
            neighbor_fn = top_info.get_neighbor_list_fn(
                displacement_fn, traj_info.box_size, r_cutoff, dr_threshold)
            neighbor_fn = jit(neighbor_fn)
            neighbors = neighbor_fn.allocate(traj_states[0].center) # We use the COMs
        else:
            neighbors_idx = top_info.unbonded_nbrs.T


        compute_subterms_fn = model.compute_subterms
        compute_subterms_fn = jit(compute_subterms_fn)
        computed_subterms = list()
        for state in tqdm(traj_states):

            if use_neighbors:
                neighbors = jit(neighbors.update)(state.center)
                neighbors_idx = neighbors.idx

            dgs = compute_subterms_fn(
                state, seq_oh, top_info.bonded_nbrs, neighbors_idx, is_end, unique_hb_pairs
            )
            avg_subterms = onp.array(dgs) / top_info.n # average per nucleotide
            computed_subterms.append(avg_subterms)
            # computed_subterms.append(avg_subterms[:-1])

        computed_subterms = onp.array(computed_subterms)

        round_places = 6
        if round_places < tol_places:
            raise RuntimeError(f"We round for printing purposes, but this must be higher precision than the tolerance")

        # Check for equality
        for i, (idx, row) in enumerate(oxdna_subterms.iterrows()): # note: i does not necessarily equal idx

            ith_oxdna_subterms = row.to_numpy()[1:]
            ith_computed_subterms = computed_subterms[i]
            ith_computed_subterms = onp.round(ith_computed_subterms, 6)

            if verbose:
                print(f"\tState {i}:")
                print(f"\t\tComputed subterms: {ith_computed_subterms}")
                print(f"\t\toxRNA subterms: {ith_oxdna_subterms}")
                print(f"\t\t|Difference|: {onp.abs(ith_computed_subterms - ith_oxdna_subterms)}")
                print(f"\t\t|HB Difference|: {onp.abs(ith_computed_subterms[4] - ith_oxdna_subterms[4])}")

            for subterm_idx, (oxdna_subterm, computed_subterm) in enumerate(zip(ith_oxdna_subterms, ith_computed_subterms)):
                if subterm_idx == 4:
                    self.assertAlmostEqual(oxdna_subterm, computed_subterm, places=hb_tol_places)
                else:
                    self.assertAlmostEqual(oxdna_subterm, computed_subterm, places=tol_places)

    def test_subterms(self):
        print(utils.bcolors.WARNING + "\nWARNING: errors for hydrogen bonding and cross stacking are subject to approximation of pi in parameter file\n" + utils.bcolors.ENDC)

        subterm_tests = [
            (self.test_data_basedir / "simple-helix-rna2-12bp-half-charged-ends", "sys.top", "output.dat", 296.15, 1.0, True, True, False, False),
            (self.test_data_basedir / "simple-helix-rna2-12bp", "sys.top", "output.dat", 296.15, 1.0, True, False, False, False),
            (self.test_data_basedir / "simple-helix-rna2-12bp-ss", "sys.top", "output.dat", 296.15, 1.0, False, False, False, False),
            (self.test_data_basedir / "simple-coax-rna2", "generated.top", "output.dat", 296.15, 1.0, True, False, False, False),
            (self.test_data_basedir / "simple-helix-rna2-12bp-ss-290.15", "sys.top", "output.dat", 290.15, 1.0, False, False, False, False),
            (self.test_data_basedir / "regr-rna2-2ht-293.15-ss", "sys.top", "output.dat", 293.15, 1.0, False, False, False, False),
            (self.test_data_basedir / "regr-rna2-2ht-293.15-sa", "sys.top", "output.dat", 293.15, 1.0, True, False, False, False),
            (self.test_data_basedir / "regr-rna2-2ht-296.15-ss", "sys.top", "output.dat", 296.15, 1.0, False, False, False, False),
            (self.test_data_basedir / "regr-rna2-2ht-296.15-sa", "sys.top", "output.dat", 296.15, 1.0, True, False, False, False),
            (self.test_data_basedir / "regr-circle-rna", "sys.top", "output.dat", 296.15, 1.0, True, True, True, False),

            # # (self.test_data_basedir / "regr-rna2-5ht-293.15-sa", "sys.top", "output.dat", 293.15, 1.0, True, False)
            (self.test_data_basedir / "simple-helix-rna2-12bp-unique", "sys.top", "output.dat", 296.15, 1.0, True, False, False, True),
        ]


        for basedir, top_fname, traj_fname, t_kelvin, salt_conc, avg_seq, half_charged_ends, is_circle, unique_topology in subterm_tests:
            for use_neighbors in [False, True]:
                self.check_energy_subterms(
                    basedir,
                    top_fname,
                    traj_fname,
                    t_kelvin,
                    salt_conc,
                    avg_seq=avg_seq,
                    params=None,
                    use_neighbors=use_neighbors,
                    half_charged_ends=half_charged_ends,
                    is_circle=is_circle,
                    unique_topology=unique_topology
                )


if __name__ == "__main__":
    unittest.main()
