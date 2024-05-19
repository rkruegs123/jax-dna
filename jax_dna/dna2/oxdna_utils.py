import pdb
from copy import deepcopy
import unittest
from pathlib import Path
from tqdm import tqdm
import numpy as onp

import jax.numpy as jnp
from jax import jit
from jax_md import space

from jax_dna.dna1 import oxdna_utils as dna1_utils
from jax_dna.dna2 import model
from jax_dna.common.utils import DEFAULT_TEMP
from jax_dna.common import utils, topology, trajectory


def recompile_oxdna(override_base_params, oxdna_path, t_kelvin, num_threads=4, seq_avg=True):
    variable_mapper = deepcopy(dna1_utils.DEFAULT_VARIABLE_MAPPER)
    override_base_params_copy = deepcopy(override_base_params)
    if seq_avg:

        # The variable mapper should always be correct
        variable_mapper['stacking']["eps_stack_base"] = "STCK_BASE_EPS_OXDNA2"
        variable_mapper['stacking']["eps_stack_kt_coeff"] = "STCK_FACT_EPS_OXDNA2"
        variable_mapper["hydrogen_bonding"]["eps_hb"] = "HYDR_EPS_OXDNA2"

        # However, since we're using dna1 to compile, we also have to supply the correct default values if we aren't overriding them manually
        if "eps_stack_base" not in override_base_params_copy["stacking"]:
            override_base_params_copy["stacking"]["eps_stack_base"] = model.default_base_params_seq_avg["stacking"]["eps_stack_base"]
        if "eps_stack_kt_coeff" not in override_base_params_copy["stacking"]:
            override_base_params_copy["stacking"]["eps_stack_kt_coeff"] = model.default_base_params_seq_avg["stacking"]["eps_stack_kt_coeff"]
        if "eps_hb" not in override_base_params_copy["hydrogen_bonding"]:
            override_base_params_copy["hydrogen_bonding"]["eps_hb"] = model.default_base_params_seq_avg["hydrogen_bonding"]["eps_hb"]

    return dna1_utils.recompile_oxdna(override_base_params_copy, oxdna_path, t_kelvin, num_threads=num_threads, variable_mapper=variable_mapper)


class TestOxdnaUtils(unittest.TestCase):

    test_data_basedir = Path("data/test-data")

    def test_subterms(self):

        import subprocess
        import pandas as pd

        oxdna_path = Path("/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/")
        oxdna_exec_path = oxdna_path / "build/bin/oxDNA"
        t_kelvin = DEFAULT_TEMP
        seq_avg = True

        override_base_params = deepcopy(model.EMPTY_BASE_PARAMS)
        if seq_avg:
            default_base_params = model.default_base_params_seq_avg
        else:
            default_base_params = model.default_base_params_seq_dep

        override_base_params["stacking"] = default_base_params["stacking"]

        recompile_oxdna(override_base_params, oxdna_path, t_kelvin, num_threads=4)

        basedir = self.test_data_basedir / "simple-helix-oxdna2-12bp"
        for fname in ["energy.dat", "last_conf.dat", "output.dat", "pair.dat", "potential.dat", "split_energy.dat"]:
            fpath = basedir / fname
            if fpath.exists():
                fpath.unlink()

        p = subprocess.Popen([oxdna_exec_path, "input"], cwd=basedir)
        p.wait()
        assert(p.returncode == 0)

        # Load the oxDNA subterms
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
        top_fname = "sys.top"
        top_path = basedir / top_fname
        if not top_path.exists():
            raise RuntimeError(f"No topology file at location: {top_path}")
        traj_fname = "output.dat"
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
        ss_hb_weights = utils.HB_WEIGHTS_SA
        ss_stack_weights = utils.STACK_WEIGHTS_SA
        salt_conc = 0.5
        em = model.EnergyModel(displacement_fn, override_base_params, t_kelvin=t_kelvin, salt_conc=salt_conc,
                               ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights,
                               seq_avg=seq_avg)

        ## setup neighbors, if necessary
        neighbors_idx = top_info.unbonded_nbrs.T

        compute_subterms_fn = jit(em.compute_subterms)
        computed_subterms = list()
        for state in tqdm(traj_states):

            dgs = compute_subterms_fn(
                state, seq_oh, top_info.bonded_nbrs, neighbors_idx)
            avg_subterms = onp.array(dgs) / top_info.n # average per nucleotide
            computed_subterms.append(avg_subterms)

        computed_subterms = onp.array(computed_subterms)

        tol_places = 4
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
            print(f"\t\t|HB Difference|: {onp.abs(ith_computed_subterms[4] - ith_oxdna_subterms[4])}")

            for oxdna_subterm, computed_subterm in zip(ith_oxdna_subterms, ith_computed_subterms):
                self.assertAlmostEqual(oxdna_subterm, computed_subterm, places=tol_places)


if __name__ == "__main__":
    unittest.main()
