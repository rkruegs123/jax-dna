import pdb
import unittest
from functools import partial
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import numpy as onp
from io import StringIO

from jax import jit, random, lax, grad, value_and_grad, vmap
import jax.numpy as jnp
from jax_md import space, simulate, rigid_body

from jax_dna.common.read_seq_specific import read_ss_oxdna
from jax_dna.common.utils import DEFAULT_TEMP, clamp
from jax_dna.common.utils import Q_to_back_base, Q_to_base_normal, Q_to_cross_prod
from jax_dna.common.interactions import v_fene_smooth, stacking, exc_vol_bonded, \
    exc_vol_unbonded, cross_stacking, coaxial_stacking, hydrogen_bonding, \
    coaxial_stacking2
from jax_dna.common import utils, topology, trajectory
from jax_dna.dna2.load_params import load, _process
from jax_dna.dna2 import lammps_utils

from jax.config import config
config.update("jax_enable_x64", True)


default_base_params_seq_avg = load(seq_avg=True, process=False)
default_base_params_seq_dep = load(seq_avg=False, process=False)

EMPTY_BASE_PARAMS = {
    "fene": dict(),
    "excluded_volume": dict(),
    "stacking": dict(),
    "hydrogen_bonding": dict(),
    "cross_stacking": dict(),
    "coaxial_stacking": dict(),
    "debye": dict()
}


"""
Missing couplings:
- back_base vs. base_back in excluded_volume. In LAMMPS, we are just using back_base.

"""
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

def get_full_base_params(override_base_params, seq_avg=True):
    if seq_avg:
        default_base_params = default_base_params_seq_avg
    else:
        default_base_params = default_base_params_seq_dep

    fene_params = default_base_params["fene"] | override_base_params["fene"]
    exc_vol_params = default_base_params["excluded_volume"] | override_base_params["excluded_volume"]
    stacking_params = default_base_params["stacking"] | override_base_params["stacking"]
    hb_params = default_base_params["hydrogen_bonding"] | override_base_params["hydrogen_bonding"]
    cr_params = default_base_params["cross_stacking"] | override_base_params["cross_stacking"]
    cx_params = default_base_params["coaxial_stacking"] | override_base_params["coaxial_stacking"]
    debye_params = default_base_params["debye"] | override_base_params["debye"]

    base_params = {
        "fene": fene_params,
        "excluded_volume": exc_vol_params,
        "stacking": stacking_params,
        "hydrogen_bonding": hb_params,
        "cross_stacking": cr_params,
        "coaxial_stacking": cx_params,
        "debye": debye_params
    }
    add_coupling(base_params)
    return base_params

com_to_backbone_x = -0.3400
com_to_backbone_y = 0.3408
com_to_backbone_oxdna1 = -0.4
com_to_stacking = 0.34
com_to_hb = 0.4
class EnergyModel:
    def __init__(self, displacement_fn, override_base_params=EMPTY_BASE_PARAMS,
                 t_kelvin=DEFAULT_TEMP, ss_hb_weights=utils.HB_WEIGHTS_SA,
                 ss_stack_weights=utils.STACK_WEIGHTS_SA,
                 salt_conc=0.5, q_eff=0.815, seq_avg=True
    ):
        self.displacement_fn = displacement_fn
        self.displacement_mapped = jit(space.map_bond(partial(displacement_fn)))
        self.t_kelvin = t_kelvin
        self.salt_conc = salt_conc
        kT = utils.get_kt(self.t_kelvin)

        self.ss_hb_weights = ss_hb_weights
        self.ss_hb_weights_flat = self.ss_hb_weights.flatten()

        self.ss_stack_weights = ss_stack_weights
        self.ss_stack_weights_flat = self.ss_stack_weights.flatten()

        self.base_params = get_full_base_params(override_base_params, seq_avg)
        self.params = _process(self.base_params, self.t_kelvin, self.salt_conc)

    def compute_subterms(self, body, seq, bonded_nbrs, unbonded_nbrs):
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

        back_sites = body.center + com_to_backbone_x*back_base_vectors + com_to_backbone_y*cross_prods
        back_sites_dna1 = body.center + com_to_backbone_oxdna1 * back_base_vectors
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

        dr_back_dna1_nn = self.displacement_mapped(back_sites_dna1[nn_i], back_sites_dna1[nn_j]) # N x N x 3
        r_back_dna1_nn = jnp.linalg.norm(dr_back_dna1_nn, axis=1)

        theta4 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nn_i], base_normals[nn_j])))
        theta5 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', dr_stack_nn, base_normals[nn_j]) / r_stack_nn))
        theta6 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nn_i], dr_stack_nn) / r_stack_nn))
        cosphi1 = -jnp.einsum('ij, ij->i', cross_prods[nn_i], dr_back_dna1_nn) / r_back_dna1_nn
        cosphi2 = -jnp.einsum('ij, ij->i', cross_prods[nn_j], dr_back_dna1_nn) / r_back_dna1_nn


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
        theta8_op = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[op_i], dr_base_op) / r_base_op))

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
        stack_probs = utils.get_pair_probs(seq, nn_i, nn_j)
        stack_weights = jnp.dot(stack_probs, self.ss_stack_weights_flat)
        stack_dg = jnp.dot(stack_weights, v_stack)

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
        hb_dg = jnp.dot(hb_weights, v_hb)

        cr_stack_dg = cross_stacking(
            r_base_op, theta1_op, theta2_op, theta3_op,
            theta4_op, theta7_op, theta8_op, **self.params["cross_stacking"])
        cr_stack_dg = jnp.where(mask, cr_stack_dg, 0.0).sum() # Mask for neighbors

        cx_stack_dg = coaxial_stacking2(
            dr_stack_op, theta4_op, theta1_op, theta5_op,
            theta6_op, **self.params["coaxial_stacking"])
        cx_stack_dg = jnp.where(mask, cx_stack_dg, 0.0).sum() # Mask for neighbors


        # Compute debye_dg
        r_base_op = jnp.linalg.norm(dr_backbone_op, axis=1)
        def db_term(r):
            energy_full = jnp.exp(r * self.params['debye']['minus_kappa']) \
                          * (self.params['debye']['prefactor'] / r)
            energy_smooth =  self.params['debye']['B'] \
                             * (r - self.params['debye']['rcut'])**2
            cond = r < self.params['debye']['r_high']
            energy = jnp.where(cond, energy_full, energy_smooth)
            return jnp.where(r < self.params['debye']['rcut'], energy, 0.0)
        db_dg = vmap(db_term)(r_base_op).sum()

        return fene_dg, exc_vol_bonded_dg, stack_dg, exc_vol_unbonded_dg, hb_dg, cr_stack_dg, cx_stack_dg, db_dg


    def energy_fn(self, body, seq, bonded_nbrs, unbonded_nbrs):
        dgs = self.compute_subterms(body, seq, bonded_nbrs, unbonded_nbrs)
        fene_dg, exc_vol_bonded_dg, stack_dg, exc_vol_unbonded_dg, hb_dg, cr_stack_dg, cx_stack_dg, db_dg = dgs
        return fene_dg + exc_vol_bonded_dg + stack_dg + exc_vol_unbonded_dg + hb_dg + cr_stack_dg + cx_stack_dg + db_dg


class TestDna2(unittest.TestCase):
    test_data_basedir = Path("data/test-data")

    def test_lammps(self, tol_places=4):
        t_kelvin = 300.0

        ss_path = "data/seq-specific/seq_oxdna2.txt"
        ss_hb_weights, ss_stack_weights = read_ss_oxdna(
            ss_path,
            default_base_params_seq_dep['hydrogen_bonding']['eps_hb'],
            default_base_params_seq_dep['stacking']['eps_stack_base'],
            default_base_params_seq_dep['stacking']['eps_stack_kt_coeff'],
            enforce_symmetry=False,
            t_kelvin=t_kelvin
        )
        # ss_hb_weights = utils.HB_WEIGHTS_SA
        # ss_stack_weights = utils.STACK_WEIGHTS_SA

        basedir = Path("data/test-data/lammps-oxdna2-40bp/")
        log_path = basedir / "log.lammps"

        log_df = lammps_utils.read_log(log_path)

        top_path = basedir / "data.top"
        if not top_path.exists():
            raise RuntimeError(f"No topology file at location: {top_path}")
        traj_path = basedir / "data.oxdna"
        if not traj_path.exists():
            raise RuntimeError(f"No trajectory file at location: {traj_path}")

        ## note: we don't reverse direction to keep ordering the same
        top_info = topology.TopologyInfo(top_path, reverse_direction=False)
        seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
        traj_states = traj_info.get_states()

        displacement_fn, shift_fn = space.periodic(traj_info.box_size)
        model = EnergyModel(displacement_fn, t_kelvin=t_kelvin,
                            ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights,
                            salt_conc=0.15, seq_avg=False)

        neighbors_idx = top_info.unbonded_nbrs.T

        compute_subterms_fn = jit(model.compute_subterms)
        energy_fn = jit(model.energy_fn)
        computed_subterms = list()
        computed_pot_energies = list()
        for state in tqdm(traj_states):

            dgs = compute_subterms_fn(
                state, seq_oh, top_info.bonded_nbrs, neighbors_idx)
            avg_subterms = onp.array(dgs) / top_info.n # average per nucleotide
            computed_subterms.append(avg_subterms)

            pot_energy = energy_fn(state, seq_oh, top_info.bonded_nbrs, neighbors_idx)
            computed_pot_energies.append(pot_energy)

        computed_subterms = onp.array(computed_subterms)
        computed_pot_energies = onp.array(computed_pot_energies)


        for idx in range(len(traj_states)):
            print(f"\nState {idx}:")
            ith_subterms = computed_subterms[idx]
            ith_pot_energy = computed_pot_energies[idx]
            row = log_df.iloc[idx]

            # FENE
            computed_fene = ith_subterms[0]
            gt_fene = row.E_bond
            print(f"- |FENE diff|: {onp.abs(gt_fene - computed_fene)}")
            self.assertAlmostEqual(gt_fene, computed_fene, places=tol_places)

            # Excluded volume
            # Notee: lammps does not included bonded excluded volume
            # computed_excv = ith_subterms[1] + ith_subterms[3]
            computed_excv = ith_subterms[3]
            gt_excv = row.c_excvEnergy
            excv_diff = onp.abs(gt_excv - computed_excv)
            print(f"- |Exc. Vol. diff|: {excv_diff}")
            # if excv_diff > 10**(-tol_places):
            #     pdb.set_trace()
            self.assertAlmostEqual(gt_excv, computed_excv, places=tol_places)

            # Stack
            computed_stk = ith_subterms[2]
            gt_stk = row.c_stkEnergy
            print(f"- |Stack diff|: {onp.abs(gt_stk - computed_stk)}")
            self.assertAlmostEqual(gt_stk, computed_stk, places=tol_places)

            # Hydrogen bonding
            computed_hb = ith_subterms[4]
            gt_hb = row.c_hbondEnergy
            print(f"- |H.B. diff|: {onp.abs(gt_hb - computed_hb)}")
            self.assertAlmostEqual(gt_hb, computed_hb, places=tol_places)

            # Cross stack
            computed_xstk = ith_subterms[5]
            gt_xstk = row.c_xstkEnergy
            print(f"- |X Stack diff|: {onp.abs(gt_xstk - computed_xstk)}")
            self.assertAlmostEqual(gt_xstk, computed_xstk, places=tol_places)

            # Coaxial stack
            computed_coaxstk = ith_subterms[6]
            gt_coaxstk = row.c_coaxstkEnergy
            coax_diff = onp.abs(gt_coaxstk - computed_coaxstk)
            print(f"- |Coax. Stack diff|: {coax_diff}")
            self.assertAlmostEqual(gt_coaxstk, computed_coaxstk, places=tol_places)

            # Debye-Huckel
            computed_dh = ith_subterms[7]
            gt_dh = row.c_dhEnergy
            print(f"- |Debye diff|: {onp.abs(gt_dh - computed_dh)}")
            self.assertAlmostEqual(gt_dh, computed_dh, places=tol_places)

        return


    def check_energy_subterms(self, basedir, top_fname, traj_fname,
                              t_kelvin, salt_conc,
                              r_cutoff=10.0, dr_threshold=0.2,
                              tol_places=4, verbose=True, avg_seq=True):


        if avg_seq:
            ss_hb_weights = utils.HB_WEIGHTS_SA
            ss_stack_weights = utils.STACK_WEIGHTS_SA
        else:

            ss_path = "data/seq-specific/seq_oxdna2.txt"
            ss_hb_weights, ss_stack_weights = read_ss_oxdna(
                ss_path,
                default_base_params_seq_dep['hydrogen_bonding']['eps_hb'],
                default_base_params_seq_dep['stacking']['eps_stack_base'],
                default_base_params_seq_dep['stacking']['eps_stack_kt_coeff'],
                enforce_symmetry=False,
                t_kelvin=t_kelvin
            )


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
        top_info = topology.TopologyInfo(top_path, reverse_direction=False)
        seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
        traj_states = traj_info.get_states()

        displacement_fn, shift_fn = space.periodic(traj_info.box_size)
        model = EnergyModel(displacement_fn, t_kelvin=t_kelvin, salt_conc=salt_conc,
                            ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights,
                            seq_avg=avg_seq)

        ## setup neighbors, if necessary
        neighbors_idx = top_info.unbonded_nbrs.T

        compute_subterms_fn = jit(model.compute_subterms)
        computed_subterms = list()
        for state in tqdm(traj_states):

            dgs = compute_subterms_fn(
                state, seq_oh, top_info.bonded_nbrs, neighbors_idx)
            avg_subterms = onp.array(dgs) / top_info.n # average per nucleotide
            computed_subterms.append(avg_subterms)

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
                print(f"\t\toxDNA subterms: {ith_oxdna_subterms}")
                # print(f"\t\t|FENE Diff.|: {onp.abs(ith_computed_subterms[0] - ith_oxdna_subterms[0])}")
                # print(f"\t\t|Exc. Vol. Bonded Diff.|: {onp.abs(ith_computed_subterms[1] - ith_oxdna_subterms[1])}")
                # print(f"\t\t|Stack Diff.|: {onp.abs(ith_computed_subterms[2] - ith_oxdna_subterms[2])}")
                # print(f"\t\t|Exc. Vol. Non-bonded Diff.|: {onp.abs(ith_computed_subterms[3] - ith_oxdna_subterms[3])}")
                # print(f"\t\t|H.B. Diff.|: {onp.abs(ith_computed_subterms[4] - ith_oxdna_subterms[4])}")
                # print(f"\t\t|Cr. Stack Diff.|: {onp.abs(ith_computed_subterms[5] - ith_oxdna_subterms[5])}")
                # coax_diff = onp.abs(ith_computed_subterms[6] - ith_oxdna_subterms[6])
                # print(f"\t\t|Cx. Stack Diff.|: {coax_diff}")
                # print(f"\t\t|Debye Diff.|: {onp.abs(ith_computed_subterms[7] - ith_oxdna_subterms[7])}")
                print(f"\t\t|Difference|: {onp.abs(ith_computed_subterms - ith_oxdna_subterms)}")
                print(f"\t\t|HB Difference|: {onp.abs(ith_computed_subterms[4] - ith_oxdna_subterms[4])}")

            for oxdna_subterm, computed_subterm in zip(ith_oxdna_subterms, ith_computed_subterms):
                self.assertAlmostEqual(oxdna_subterm, computed_subterm, places=tol_places)

    def test_subterms(self):
        print(utils.bcolors.WARNING + "\nWARNING: errors for hydrogen bonding and cross stacking are subject to approximation of pi in parameter file\n" + utils.bcolors.ENDC)

        subterm_tests = [
            (self.test_data_basedir / "simple-helix-oxdna2", "generated.top", "output.dat", 296.15, 0.5, True),
            (self.test_data_basedir / "simple-helix-oxdna2-ss", "generated.top", "output.dat", 296.15, 0.5, False),
            (self.test_data_basedir / "simple-coax-oxdna2", "generated.top", "output.dat", 296.15, 0.5, True),
            (self.test_data_basedir / "simple-coax-oxdna2-rev", "generated.top", "output.dat", 296.15, 0.5, True)
        ]

        for basedir, top_fname, traj_fname, t_kelvin, salt_conc, avg_seq in subterm_tests:
            self.check_energy_subterms(basedir, top_fname, traj_fname, t_kelvin, salt_conc,
                                       avg_seq=avg_seq)


if __name__ == "__main__":
    unittest.main()
