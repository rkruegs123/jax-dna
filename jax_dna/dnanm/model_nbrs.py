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
jax.config.update("jax_enable_x64", True)
from jax import jit, random, lax, grad, value_and_grad
import jax.numpy as jnp
from jax_md import space, rigid_body, simulate

from jax_dna.common.utils import DEFAULT_TEMP
from jax_dna.common import utils, topology_protein_na, trajectory
from jax_dna.dna2 import model as model_dna2
from jax_dna.anm import model as model_anm
from jax_dna.common.read_seq_specific import read_ss_oxdna
from jax_dna.common.utils import Q_to_back_base, Q_to_base_normal, Q_to_cross_prod



class EnergyModel:
    def __init__(
        self,
        displacement_fn,
        is_protein_idx,
        is_nt_idx,
        # ANM
        network,
        eq_distances,
        spring_constants,
        # DNA2
        override_base_params=model_dna2.EMPTY_BASE_PARAMS,
        t_kelvin=DEFAULT_TEMP,
        ss_hb_weights=utils.HB_WEIGHTS_SA,
        ss_stack_weights=utils.STACK_WEIGHTS_SA,
        salt_conc=0.5,
        q_eff=0.815,
        seq_avg=True,
        ignore_exc_vol_bonded=False
    ):
        self.t_kelvin = t_kelvin
        self.displacement_fn = displacement_fn
        self.displacement_mapped = jit(space.map_bond(partial(displacement_fn)))
        self.network = network
        self.eq_distances = eq_distances
        self.spring_constants = spring_constants

        self.is_protein_idx = is_protein_idx
        self.is_nt_idx = is_nt_idx

        self.dna2_energy_model = model_dna2.EnergyModel(
            displacement_fn,
            override_base_params,
            t_kelvin,
            ss_hb_weights,
            ss_stack_weights,
            salt_conc,
            q_eff,
            seq_avg,
            ignore_exc_vol_bonded
        )


    def compute_subterms(
        self,
        body,
        aa_seq,
        nt_seq_oh,
        bonded_nbrs_nt,
        unbonded_nbrs,
        # unbonded_nbrs_nt,
        # unbonded_nbrs_protein_nt,
        is_end=None
    ):

        spring_dg, prot_exc_vol_dg = model_anm.compute_subterms(
            body,
            aa_seq,
            self.network,
            self.eq_distances,
            self.spring_constants,
            self.displacement_fn,
            t_kelvin=self.t_kelvin,
        )

        dna2_dgs = self.dna2_energy_model.compute_pairwise_dgs(
            body, nt_seq_oh, bonded_nbrs_nt, unbonded_nbrs.T, is_end
        )
        fene_dgs, exc_vol_bonded_dgs, stack_dgs, exc_vol_unbonded_dgs, hb_dgs, cr_stack_dgs, cx_stack_dgs, db_dgs, metadata = dna2_dgs
        fene_dg = fene_dgs.sum()
        exc_vol_bonded_dg = exc_vol_bonded_dgs.sum()
        stack_dg = stack_dgs.sum()

        is_nt_nt_pair = jax.vmap(lambda i, j: jnp.logical_and(self.is_nt_idx[i], self.is_nt_idx[j]))(unbonded_nbrs[:, 0], unbonded_nbrs[:, 1])
        exc_vol_unbonded_dg = jnp.where(is_nt_nt_pair, exc_vol_unbonded_dgs, 0.0).sum()
        hb_dg = jnp.where(is_nt_nt_pair, hb_dgs, 0.0).sum()
        cr_stack_dg = jnp.where(is_nt_nt_pair, cr_stack_dgs, 0.0).sum()
        cx_stack_dg = jnp.where(is_nt_nt_pair, cx_stack_dgs, 0.0).sum()
        db_dg = jnp.where(is_nt_nt_pair, db_dgs, 0.0).sum()


        # protein/dna excluded volume
        back_sites, _, _, base_sites = metadata

        def protein_na_pair_exc_vol_fn(i, j):

            # Get p_idx and nt_idx
            ## note: assumes that theere is one protein index and one nt index. Will 0 out at the end
            p_idx = jnp.where(self.is_protein_idx[i], i, j)
            nt_idx = jnp.where(self.is_protein_idx[i], j, i)
            is_protein_nt_pair = jnp.logical_and(self.is_protein_idx[p_idx], self.is_nt_idx[nt_idx])


            dr_back = self.displacement_fn(body.center[p_idx], back_sites[nt_idx])
            r_back = space.distance(dr_back)
            val_back = model_anm.excluded_volume(
                r_back,
                eps=2.0,
                sigma=0.570,
                r_c=0.573,
                r_star=0.569,
                b=17.9*10**7,
            )
            val_back = jnp.nan_to_num(jnp.where(p_idx == nt_idx, 0.0, val_back))

            dr_base = self.displacement_fn(body.center[p_idx], base_sites[nt_idx])
            r_base = space.distance(dr_base)
            val_base = model_anm.excluded_volume(
                r_base,
                eps=2.0,
                sigma=0.360,
                r_c=0.363,
                r_star=0.359,
                b=29.6*10**7,
            )
            val_base = jnp.nan_to_num(jnp.where(p_idx == nt_idx, 0.0, val_base))

            return jnp.where(is_protein_nt_pair, val_back + val_base, 0.0)
        prot_nt_excl_vol_dgs = jax.vmap(protein_na_pair_exc_vol_fn)(unbonded_nbrs[:, 0], unbonded_nbrs[:, 1])
        prot_nt_excl_vol_dg = prot_nt_excl_vol_dgs.sum()

        return fene_dg, exc_vol_bonded_dg, stack_dg, exc_vol_unbonded_dg, \
            hb_dg, cr_stack_dg, cx_stack_dg, db_dg, spring_dg, prot_exc_vol_dg, prot_nt_excl_vol_dg


    def energy_fn(
        self,
        body,
        aa_seq,
        nt_seq_oh,
        bonded_nbrs_nt,
        unbonded_nbrs,
        # unbonded_nbrs_nt,
        # unbonded_nbrs_protein_nt,
        is_end=None
    ):
        all_dgs = self.compute_subterms(body, aa_seq, nt_seq_oh, bonded_nbrs_nt, unbonded_nbrs, is_end)
        fene_dg, exc_vol_bonded_dg, stack_dg, exc_vol_unbonded_dg, hb_dg, cr_stack_dg, cx_stack_dg, db_dg, spring_dg, prot_exc_vol_dg, prot_nt_excl_vol_dg = all_dgs
        return prot_exc_vol_dg + spring_dg + prot_nt_excl_vol_dg + fene_dg + exc_vol_bonded_dg \
            + stack_dg + exc_vol_unbonded_dg + hb_dg + cr_stack_dg + cx_stack_dg + db_dg




class TestDNANM(unittest.TestCase):
    test_data_basedir = Path("data/test-data")

    def check_energy_subterms(
        self,
        basedir,
        top_fname,
        traj_fname,
        par_fname,
        t_kelvin,
        salt_conc,
        seq_avg,
        half_charged_ends,
        tol_places=4,
    ):

        print(f"\n---- Checking energy breakdown agreement for base directory: {basedir} ----")

        if seq_avg:
            ss_hb_weights = utils.HB_WEIGHTS_SA
            ss_stack_weights = utils.STACK_WEIGHTS_SA
        else:

            ss_path = "data/seq-specific/seq_oxdna2.txt"
            ss_hb_weights, ss_stack_weights = read_ss_oxdna(
                ss_path,
                model_dna2.default_base_params_seq_dep['hydrogen_bonding']['eps_hb'],
                model_dna2.default_base_params_seq_dep['stacking']['eps_stack_base'],
                model_dna2.default_base_params_seq_dep['stacking']['eps_stack_kt_coeff'],
                enforce_symmetry=False,
                t_kelvin=t_kelvin
            )

        basedir = Path(basedir)
        if not basedir.exists():
            raise RuntimeError(f"No directory exists at location: {basedir}")


        # First, load the oxDNA subterms
        split_energy_fname = basedir / "split_energy.dat"
        if not split_energy_fname.exists():
            raise RuntimeError(f"No energy subterm file exists at location: {split_energy_fname}")
        split_energy_df = pd.read_csv(
            split_energy_fname,
            names=[
                "t",
                "spring",
                "prot_exc",
                "prot_na_exc",
                "fene",
                "b_exc",
                "stack",
                "n_exc",
                "hb",
                "cr_stack",
                "cx_stack",
                "debye",
            ],
            delim_whitespace=True)
        oxdna_subterms = split_energy_df.iloc[1:, :]

        # Then, compute subterms via our energy model
        top_path = basedir / top_fname
        traj_path = basedir / traj_fname
        par_path = basedir / par_fname

        ## note: we don't reverse direction to keep ordering the same
        top_info = topology_protein_na.ProteinNucAcidTopology(top_path, par_path)

        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
        traj_states = traj_info.get_states()

        if half_charged_ends:
            is_end = jnp.array(top_info.is_end)
        else:
            is_end = None

        displacement_fn, shift_fn = space.periodic(traj_info.box_size)

        params = deepcopy(model_dna2.EMPTY_BASE_PARAMS)

        model = EnergyModel(
            displacement_fn,
            is_nt_idx=jnp.array(top_info.is_nt_idx),
            is_protein_idx=jnp.array(top_info.is_protein_idx),
            # ANM
            network=jnp.array(top_info.network),
            eq_distances=jnp.array(top_info.eq_distances),
            spring_constants=jnp.array(top_info.spring_constants),
            # DNA2
            override_base_params=params,
            t_kelvin=t_kelvin,
            salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            ss_stack_weights=ss_stack_weights,
            seq_avg=seq_avg
        )
        compute_subterms_fn = model.compute_subterms
        compute_subterms_fn = jit(compute_subterms_fn)

        computed_subterms = list()
        for state in tqdm(traj_states):

            dgs = compute_subterms_fn(
                state,
                jnp.array(top_info.aa_seq_idx),
                jnp.array(utils.get_one_hot(top_info.nt_seq), dtype=jnp.float64),
                jnp.array(top_info.bonded_nbrs),
                jnp.array(top_info.unbonded_nbrs),
                is_end
            )

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

            print(f"\tState {i}:")
            print(f"\t\tComputed subterms: {ith_computed_subterms}")
            print(f"\t\toxDNA subterms: {ith_oxdna_subterms}")
            print(f"\t\t|Difference|: {onp.abs(ith_computed_subterms - ith_oxdna_subterms)}")

            for oxdna_subterm, computed_subterm in zip(ith_oxdna_subterms, ith_computed_subterms):
                self.assertAlmostEqual(oxdna_subterm, computed_subterm, places=tol_places)

    def test_subterms(self):

        subterm_tests = [
            (self.test_data_basedir / "protein-top" / "HCAGE", 300.0, 1.0, "trajectory_cpu.dat", "hcage.par", "hcage.top", True, True),
        ]

        for basedir, t_kelvin, salt_conc, traj_fname, par_fname, top_fname, seq_avg, half_charged_ends in subterm_tests:
            self.check_energy_subterms(
                basedir, top_fname, traj_fname, par_fname, t_kelvin, salt_conc, seq_avg, half_charged_ends
            )

    def test_sim(self):
        # Simulate a zinc finger (1AAY)
        box_size = 30.0
        displacement_fn, shift_fn = space.periodic(box_size)
        dt = 1e-3
        t_kelvin = DEFAULT_TEMP
        kT = utils.get_kt(t_kelvin)

        gamma = rigid_body.RigidBody(center=jnp.array([kT/2.5], dtype=jnp.float64),
                                     orientation=jnp.array([kT/7.5], dtype=jnp.float64))
        mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                    orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))

        basedir = Path("data/templates/1AAY")
        top_path = basedir / "complex.top"
        par_path = basedir / "protein.par"
        top_info = topology_protein_na.ProteinNucAcidTopology(top_path, par_path)
        # conf_path = basedir / "complex.conf"
        conf_path = basedir / "relaxed.dat"
        conf_info = trajectory.TrajectoryInfo(
            top_info,
            read_from_file=True, traj_path=conf_path, reverse_direction=False
        )
        init_body = conf_info.get_states()[0]

        n_steps = 10000
        key = random.PRNGKey(0)
        salt_conc = 0.5

        seq_avg = False
        if seq_avg:
            ss_hb_weights = utils.HB_WEIGHTS_SA
            ss_stack_weights = utils.STACK_WEIGHTS_SA
        else:

            ss_path = "data/seq-specific/seq_oxdna2.txt"
            ss_hb_weights, ss_stack_weights = read_ss_oxdna(
                ss_path,
                model_dna2.default_base_params_seq_dep['hydrogen_bonding']['eps_hb'],
                model_dna2.default_base_params_seq_dep['stacking']['eps_stack_base'],
                model_dna2.default_base_params_seq_dep['stacking']['eps_stack_kt_coeff'],
                enforce_symmetry=False,
                t_kelvin=t_kelvin
            )


        half_charged_ends = True
        if half_charged_ends:
            is_end = jnp.array(top_info.is_end)
        else:
            is_end = None

        # Define a dummy set of parameters to override
        params = deepcopy(model_dna2.EMPTY_BASE_PARAMS)

        model = EnergyModel(
            displacement_fn,
            is_nt_idx=jnp.array(top_info.is_nt_idx),
            is_protein_idx=jnp.array(top_info.is_protein_idx),
            # ANM
            network=jnp.array(top_info.network),
            eq_distances=jnp.array(top_info.eq_distances),
            spring_constants=jnp.array(top_info.spring_constants),
            # DNA2
            override_base_params=params,
            t_kelvin=t_kelvin,
            salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            ss_stack_weights=ss_stack_weights,
            seq_avg=seq_avg
        )

        energy_fn = partial(
            model.energy_fn,
            aa_seq=jnp.array(top_info.aa_seq_idx),
            nt_seq_oh=jnp.array(utils.get_one_hot(top_info.nt_seq), dtype=jnp.float64),
            bonded_nbrs_nt=jnp.array(top_info.bonded_nbrs),
            unbonded_nbrs=jnp.array(top_info.unbonded_nbrs),
            is_end=is_end
        )

        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        step_fn = jit(step_fn)
        state = init_fn(key, init_body, mass=mass)

        traj = list()
        for _ in tqdm(range(n_steps)):
            state = step_fn(state)
            traj.append(state.position)
        traj = utils.tree_stack(traj)

        write_traj = True
        if write_traj:
            traj_to_write = traj[::100]
            traj_info = trajectory.TrajectoryInfo(
                top_info, read_from_states=True, states=traj_to_write, box_size=box_size)
            traj_info.write("dnanm_sanity.dat", reverse=False)

        # (pos_sum, traj), pos_sum_grad = jit(value_and_grad(sim_fn, has_aux=True))(test_param_dict)

        # self.assertNotEqual(pos_sum_grad, 0.0)


if __name__ == "__main__":
    unittest.main()
