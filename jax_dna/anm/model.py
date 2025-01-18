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
from jax_md import space

from jax_dna.common.utils import DEFAULT_TEMP
from jax_dna.common import utils, topology_protein, trajectory




def excluded_volume(
    r,
    eps=2.0,
    sigma=0.350,
    r_c=0.353,
    r_star=0.349,
    b=30.7*10**7,
):
    rep_val = 4*eps*(-sigma**6 / r**6 + sigma**12 / r**12)
    smoothing_val = b*eps*(r-r_c)**4

    return jnp.where(r < r_star, rep_val, jnp.where(r < r_c, smoothing_val, 0.0))


def compute_subterms(
    body,
    seq,
    network,
    eq_distances,
    spring_constants,
    displacement_fn,
    t_kelvin=DEFAULT_TEMP,
):
    n = body.center.shape[0]

    displacement_mapped = jit(space.map_bond(partial(displacement_fn)))

    network_i = network[:, 0]
    network_j = network[:, 1]


    def pair_spring_fn(i, j):
        r0 = eq_distances[i, j]
        k = spring_constants[i, j]

        dr = displacement_fn(body.center[i], body.center[j])
        r = space.distance(dr)
        return 0.5 * k * (r-r0)**2
        # return k * (r-r0)**2
    spring_dgs = jax.vmap(pair_spring_fn)(network_i, network_j)
    spring_dg = spring_dgs.sum()

    def pair_exc_vol_fn(i, j):
        dr = displacement_fn(body.center[i], body.center[j])
        r = space.distance(dr)
        val = jnp.where(i == j, 0.0, excluded_volume(r))
        return jnp.nan_to_num(val) # Note: I think we could actually get away without this
    exc_vol_dgs = jax.vmap(pair_exc_vol_fn)(jnp.arange(n), jnp.arange(n)) # FIXME: compute over all n?
    exc_vol_dg = exc_vol_dgs.sum()

    return spring_dg, exc_vol_dg


def energy_fn(
    body,
    seq,
    network,
    eq_distances,
    spring_constants,
    displacement_fn,
    t_kelvin=DEFAULT_TEMP,
):
    dgs = compute_subterms(body, seq, network, eq_distances, spring_constants, displacement_fn, t_kelvin=DEFAULT_TEMP)
    spring_dg, exc_vol_dg = dgs
    return exc_vol_dg + spring_dg
force_fn = jax.grad(energy_fn)



class TestANM(unittest.TestCase):
    test_data_basedir = Path("data/test-data")

    def check_energy_subterms(
        self,
        basedir,
        top_fname,
        traj_fname,
        par_fname,
        t_kelvin,
        tol_places=4,
        check_force_not_nan=True
    ):

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
            names=["t", "spring", "exc"],
            delim_whitespace=True)
        oxdna_subterms = split_energy_df.iloc[1:, :]

        # Then, compute subterms via our energy model
        top_path = basedir / top_fname
        traj_path = basedir / traj_fname
        par_path = basedir / par_fname

        ## note: we don't reverse direction to keep ordering the same
        top_info = topology_protein.ProteinTopology(top_path, par_path)

        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
        traj_states = traj_info.get_states()

        displacement_fn, shift_fn = space.periodic(traj_info.box_size)

        computed_subterms = list()
        for state in tqdm(traj_states):


            dgs = compute_subterms(
                state,
                jnp.array(top_info.seq_idx),
                jnp.array(top_info.network),
                jnp.array(top_info.eq_distances),
                jnp.array(top_info.spring_constants),
                displacement_fn,
            )

            if check_force_not_nan:
                forces = force_fn(
                     state,
                     jnp.array(top_info.seq_idx),
                     jnp.array(top_info.network),
                     jnp.array(top_info.eq_distances),
                     jnp.array(top_info.spring_constants),
                     displacement_fn,
                 )
                assert(not jnp.isnan(forces.center).any())
                assert(not jnp.isnan(forces.orientation.vec).any())

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
            (self.test_data_basedir / "protein-top" / "KDPG", 300.0, "trajectory.dat", "kdpg.par", "kdpg.top"),
        ]

        for basedir, t_kelvin, traj_fname, par_fname, top_fname in subterm_tests:
            self.check_energy_subterms(
                basedir, top_fname, traj_fname, par_fname, t_kelvin,
            )


if __name__ == "__main__":
    unittest.main()
