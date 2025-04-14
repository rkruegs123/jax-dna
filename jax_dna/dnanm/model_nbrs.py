# ruff: noqa
import itertools
import pdb
import unittest
from copy import deepcopy
from functools import partial
from pathlib import Path

import jax
import numpy as onp
import pandas as pd
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, jit, lax, random, value_and_grad
from jax_md import energy, rigid_body, simulate, space

from jax_dna.anm import model as model_anm
from jax_dna.common import topology_protein_na, trajectory, utils
from jax_dna.common.read_seq_specific import read_ss_oxdna
from jax_dna.common.utils import DEFAULT_TEMP, Q_to_back_base, Q_to_base_normal, Q_to_cross_prod
from jax_dna.dna2 import model as model_dna2


class EnergyModel:
    def __init__(
        self,
        displacement_fn,
        is_protein_idx,
        is_nt_idx,
        aa_seq,
        nt_seq,
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
        ignore_exc_vol_bonded=False,
        # DNA/Protein interactions
        include_dna_protein_morse=False,
        dna_base_protein_sigma=None,
        dna_base_protein_epsilon=None,
        dna_base_protein_alpha=None,
        dna_back_protein_sigma=None,
        dna_back_protein_epsilon=None,
        dna_back_protein_alpha=None,
    ):
        self.include_dna_protein_morse = include_dna_protein_morse
        self.dna_base_protein_sigma = dna_base_protein_sigma
        self.dna_base_protein_epsilon = dna_base_protein_epsilon
        self.dna_base_protein_alpha = dna_base_protein_alpha
        self.dna_back_protein_sigma = dna_back_protein_sigma
        self.dna_back_protein_epsilon = dna_back_protein_epsilon
        self.dna_back_protein_alpha = dna_back_protein_alpha

        self.aa_seq = aa_seq
        self.nt_seq = nt_seq
        self.nt_seq_oh = jax.nn.one_hot(self.nt_seq, num_classes=4, dtype=jnp.int32)

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
            ignore_exc_vol_bonded,
        )

    def compute_subterms(
        self,
        body,
        bonded_nbrs_nt,
        unbonded_nbrs,
        # unbonded_nbrs_nt,
        # unbonded_nbrs_protein_nt,
        is_end=None,
    ):
        spring_dg, prot_exc_vol_dg = model_anm.compute_subterms(
            body,
            self.aa_seq,
            self.network,
            self.eq_distances,
            self.spring_constants,
            self.displacement_fn,
            t_kelvin=self.t_kelvin,
        )

        dna2_dgs = self.dna2_energy_model.compute_pairwise_dgs(
            body, self.nt_seq_oh, bonded_nbrs_nt, unbonded_nbrs.T, is_end
        )
        (
            fene_dgs,
            exc_vol_bonded_dgs,
            stack_dgs,
            exc_vol_unbonded_dgs,
            hb_dgs,
            cr_stack_dgs,
            cx_stack_dgs,
            db_dgs,
            metadata,
        ) = dna2_dgs
        fene_dg = fene_dgs.sum()
        exc_vol_bonded_dg = exc_vol_bonded_dgs.sum()
        stack_dg = stack_dgs.sum()

        is_nt_nt_pair = jax.vmap(lambda i, j: jnp.logical_and(self.is_nt_idx[i], self.is_nt_idx[j]))(
            unbonded_nbrs[:, 0], unbonded_nbrs[:, 1]
        )
        exc_vol_unbonded_dg = jnp.where(is_nt_nt_pair, exc_vol_unbonded_dgs, 0.0).sum()
        hb_dg = jnp.where(is_nt_nt_pair, hb_dgs, 0.0).sum()
        cr_stack_dg = jnp.where(is_nt_nt_pair, cr_stack_dgs, 0.0).sum()
        cx_stack_dg = jnp.where(is_nt_nt_pair, cx_stack_dgs, 0.0).sum()
        db_dg = jnp.where(is_nt_nt_pair, db_dgs, 0.0).sum()

        # protein/dna excluded volume
        back_sites, _, _, base_sites = metadata

        def protein_na_pair_exc_vol_fn(r_back, r_base):
            val_back = model_anm.excluded_volume(
                r_back,
                eps=2.0,
                sigma=0.570,
                r_c=0.573,
                r_star=0.569,
                b=17.9 * 10**7,
            )

            val_base = model_anm.excluded_volume(
                r_base,
                eps=2.0,
                sigma=0.360,
                r_c=0.363,
                r_star=0.359,
                b=29.6 * 10**7,
            )

            return val_back + val_base

        def protein_na_pair_morse_fn(r_back, r_base, aa_type, nt_type):
            val_back = energy.morse(
                r_back,
                sigma=self.dna_back_protein_sigma[aa_type, nt_type],
                epsilon=self.dna_back_protein_epsilon[aa_type, nt_type],
                alpha=self.dna_back_protein_alpha[aa_type, nt_type],
            )

            val_base = energy.morse(
                r_back,
                sigma=self.dna_base_protein_sigma[aa_type, nt_type],
                epsilon=self.dna_base_protein_epsilon[aa_type, nt_type],
                alpha=self.dna_base_protein_alpha[aa_type, nt_type],
            )

            return val_base + val_back

        def protein_na_unbonded_fn(i, j):
            # Get p_idx and nt_idx
            ## note: assumes that theere is one protein index and one nt index. Will 0 out at the end
            p_idx = jnp.where(self.is_protein_idx[i], i, j)
            nt_idx = jnp.where(self.is_protein_idx[i], j, i)
            is_protein_nt_pair = jnp.logical_and(self.is_protein_idx[p_idx], self.is_nt_idx[nt_idx])

            dr_back = self.displacement_fn(body.center[p_idx], back_sites[nt_idx])
            r_back = space.distance(dr_back)

            dr_base = self.displacement_fn(body.center[p_idx], base_sites[nt_idx])
            r_base = space.distance(dr_base)

            exc_vol_dg = protein_na_pair_exc_vol_fn(r_back, r_base)
            val = exc_vol_dg

            if self.include_dna_protein_morse:
                aa_type = self.aa_seq[p_idx]
                nt_type = self.nt_seq[nt_idx]
                morse_dg = protein_na_pair_morse_fn(r_back, r_base, aa_type, nt_type)
                val += morse_dg

            val = jnp.nan_to_num(jnp.where(p_idx == nt_idx, 0.0, val))
            return jnp.where(is_protein_nt_pair, val, 0.0)

        prot_nt_unbonded_dgs = jax.vmap(protein_na_unbonded_fn)(unbonded_nbrs[:, 0], unbonded_nbrs[:, 1])
        prot_nt_unbonded_dg = prot_nt_unbonded_dgs.sum()

        return (
            fene_dg,
            exc_vol_bonded_dg,
            stack_dg,
            exc_vol_unbonded_dg,
            hb_dg,
            cr_stack_dg,
            cx_stack_dg,
            db_dg,
            spring_dg,
            prot_exc_vol_dg,
            prot_nt_unbonded_dg,
        )

    def energy_fn(
        self,
        body,
        bonded_nbrs_nt,
        unbonded_nbrs,
        # unbonded_nbrs_nt,
        # unbonded_nbrs_protein_nt,
        is_end=None,
    ):
        all_dgs = self.compute_subterms(body, bonded_nbrs_nt, unbonded_nbrs, is_end)
        (
            fene_dg,
            exc_vol_bonded_dg,
            stack_dg,
            exc_vol_unbonded_dg,
            hb_dg,
            cr_stack_dg,
            cx_stack_dg,
            db_dg,
            spring_dg,
            prot_exc_vol_dg,
            prot_nt_unbonded_dgs,
        ) = all_dgs
        return (
            prot_exc_vol_dg
            + spring_dg
            + prot_nt_unbonded_dgs
            + fene_dg
            + exc_vol_bonded_dg
            + stack_dg
            + exc_vol_unbonded_dg
            + hb_dg
            + cr_stack_dg
            + cx_stack_dg
            + db_dg
        )


def get_init_morse_tables(default_alpha=10.0, default_epsilon=0.1):
    dna_base_protein_sigma = jnp.full((20, 4), 0.75)
    dna_base_protein_epsilon = jnp.full((20, 4), default_epsilon)
    dna_base_protein_alpha = jnp.full((20, 4), default_alpha)

    dna_back_protein_sigma = jnp.full((20, 4), 0.5)
    dna_back_protein_epsilon = jnp.full((20, 4), default_epsilon)
    dna_back_protein_alpha = jnp.full((20, 4), default_alpha)

    return (
        dna_base_protein_sigma,
        dna_base_protein_epsilon,
        dna_base_protein_alpha,
        dna_back_protein_sigma,
        dna_back_protein_epsilon,
        dna_back_protein_alpha,
    )


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
        use_neighbors,
        r_cutoff=10.0,
        dr_threshold=0.2,
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
                model_dna2.default_base_params_seq_dep["hydrogen_bonding"]["eps_hb"],
                model_dna2.default_base_params_seq_dep["stacking"]["eps_stack_base"],
                model_dna2.default_base_params_seq_dep["stacking"]["eps_stack_kt_coeff"],
                enforce_symmetry=False,
                t_kelvin=t_kelvin,
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
            delim_whitespace=True,
        )
        oxdna_subterms = split_energy_df.iloc[1:, :]

        # Then, compute subterms via our energy model
        top_path = basedir / top_fname
        traj_path = basedir / traj_fname
        par_path = basedir / par_fname

        ## note: we don't reverse direction to keep ordering the same
        top_info = topology_protein_na.ProteinNucAcidTopology(top_path, par_path)

        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False
        )
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
            aa_seq=jnp.array(top_info.aa_seq_idx),
            nt_seq=jnp.array(top_info.nt_seq_idx),
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
            seq_avg=seq_avg,
        )
        compute_subterms_fn = model.compute_subterms
        compute_subterms_fn = jit(compute_subterms_fn)

        if use_neighbors:
            neighbor_fn = top_info.get_neighbor_list_fn(displacement_fn, traj_info.box_size, r_cutoff, dr_threshold)
            neighbor_fn = jit(neighbor_fn)
            neighbors = neighbor_fn.allocate(traj_states[0].center)  # We use the COMs
        else:
            neighbors_idx = jnp.array(top_info.unbonded_nbrs.T)

        computed_subterms = list()
        for state in tqdm(traj_states):
            if use_neighbors:
                neighbors = neighbors.update(state.center)
                neighbors_idx = neighbors.idx

            dgs = compute_subterms_fn(state, jnp.array(top_info.bonded_nbrs), neighbors_idx.T, is_end)

            avg_subterms = onp.array(dgs) / top_info.n  # average per nucleotide
            computed_subterms.append(avg_subterms)

        computed_subterms = onp.array(computed_subterms)

        round_places = 6
        if round_places < tol_places:
            raise RuntimeError(f"We round for printing purposes, but this must be higher precision than the tolerance")

        # Check for equality
        for i, (idx, row) in enumerate(oxdna_subterms.iterrows()):  # note: i does not necessarily equal idx
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
            (
                self.test_data_basedir / "protein-top" / "HCAGE",
                300.0,
                1.0,
                "trajectory_cpu.dat",
                "hcage.par",
                "hcage.top",
                True,
                True,
            ),
        ]

        for basedir, t_kelvin, salt_conc, traj_fname, par_fname, top_fname, seq_avg, half_charged_ends in subterm_tests:
            for use_neighbors in [False, True]:
                self.check_energy_subterms(
                    basedir,
                    top_fname,
                    traj_fname,
                    par_fname,
                    t_kelvin,
                    salt_conc,
                    seq_avg,
                    half_charged_ends,
                    use_neighbors,
                )

    def test_sim(self):
        # Simulate a zinc finger (1AAY)
        box_size = 30.0
        displacement_fn, shift_fn = space.periodic(box_size)
        dt = 1e-3
        t_kelvin = DEFAULT_TEMP
        kT = utils.get_kt(t_kelvin)

        gamma = rigid_body.RigidBody(
            center=jnp.array([kT / 2.5], dtype=jnp.float64), orientation=jnp.array([kT / 7.5], dtype=jnp.float64)
        )
        mass = rigid_body.RigidBody(
            center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
            orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64),
        )

        # basedir = Path("data/templates/1AAY")
        # basedir = Path("data/templates/1A1L")
        basedir = Path("data/templates/1ZAA")
        top_path = basedir / "complex.top"
        par_path = basedir / "protein.par"
        top_info = topology_protein_na.ProteinNucAcidTopology(top_path, par_path)
        # conf_path = basedir / "complex.conf"
        conf_path = basedir / "relaxed.dat"
        conf_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True, traj_path=conf_path, reverse_direction=False
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
                model_dna2.default_base_params_seq_dep["hydrogen_bonding"]["eps_hb"],
                model_dna2.default_base_params_seq_dep["stacking"]["eps_stack_base"],
                model_dna2.default_base_params_seq_dep["stacking"]["eps_stack_kt_coeff"],
                enforce_symmetry=False,
                t_kelvin=t_kelvin,
            )

        half_charged_ends = True
        if half_charged_ends:
            is_end = jnp.array(top_info.is_end)
        else:
            is_end = None

        # Define a dummy set of parameters to override
        params = deepcopy(model_dna2.EMPTY_BASE_PARAMS)

        include_dna_protein_morse = True
        if include_dna_protein_morse:
            default_alpha = 10.0
            default_epsilon = 0.1

            (
                dna_base_protein_sigma,
                dna_base_protein_epsilon,
                dna_base_protein_alpha,
                dna_back_protein_sigma,
                dna_back_protein_epsilon,
                dna_back_protein_alpha,
            ) = get_init_morse_tables()

        else:
            dna_base_protein_sigma = None
            dna_base_protein_epsilon = None
            dna_base_protein_alpha = None

            dna_back_protein_sigma = None
            dna_back_protein_epsilon = None
            dna_back_protein_alpha = None

        model = EnergyModel(
            displacement_fn,
            is_nt_idx=jnp.array(top_info.is_nt_idx),
            is_protein_idx=jnp.array(top_info.is_protein_idx),
            aa_seq=jnp.array(top_info.aa_seq_idx),
            nt_seq=jnp.array(top_info.nt_seq_idx),
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
            seq_avg=seq_avg,
            # DNA/Protein interaction
            include_dna_protein_morse=include_dna_protein_morse,
            dna_base_protein_sigma=dna_base_protein_sigma,
            dna_base_protein_epsilon=dna_base_protein_epsilon,
            dna_base_protein_alpha=dna_base_protein_alpha,
            dna_back_protein_sigma=dna_back_protein_sigma,
            dna_back_protein_epsilon=dna_back_protein_epsilon,
            dna_back_protein_alpha=dna_back_protein_alpha,
        )

        energy_fn = partial(
            model.energy_fn,
            bonded_nbrs_nt=jnp.array(top_info.bonded_nbrs),
            unbonded_nbrs=jnp.array(top_info.unbonded_nbrs),
            is_end=is_end,
        )

        compute_subterms_fn = partial(
            model.compute_subterms,
            bonded_nbrs_nt=jnp.array(top_info.bonded_nbrs),
            unbonded_nbrs=jnp.array(top_info.unbonded_nbrs),
            is_end=is_end,
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
                top_info, read_from_states=True, states=traj_to_write, box_size=box_size
            )
            traj_info.write("dnanm_sanity.dat", reverse=False)

        # (pos_sum, traj), pos_sum_grad = jit(value_and_grad(sim_fn, has_aux=True))(test_param_dict)

        # self.assertNotEqual(pos_sum_grad, 0.0)


if __name__ == "__main__":
    unittest.main()
