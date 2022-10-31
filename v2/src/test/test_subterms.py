import pdb
import jax.numpy as jnp
from jax.tree_util import Partial
import pandas as pd
from pathlib import Path
import numpy as np

from jax_md import space
from jax_md import util

from energy import factory
from utils import base_site, stack_site, back_site, DEFAULT_TEMP, get_one_hot, bcolors
from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo
from loader import get_params

import oxpy
from oxDNA_analysis_tools.UTILS.RyeReader import describe
from oxDNA_analysis_tools.output_bonds import output_bonds

from jax.config import config
config.update("jax_enable_x64", True)


f64 = util.f64

# `n` is the index into states to compute the subterms
def compute_subterms(top_path, traj_path, T=DEFAULT_TEMP):
    top_info = TopologyInfo(top_path, reverse_direction=False)
    n = top_info.n
    traj_info = TrajectoryInfo(top_info, traj_path=traj_path, reverse_direction=False)
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)

    displacement_fn, shift_fn = space.periodic(traj_info.box_size)
    params = get_params.get_default_params(t=T, no_smoothing=False)

    _, _compute_subterms = factory.energy_fn_factory(displacement_fn,
                                                     back_site, stack_site, base_site,
                                                     top_info.bonded_nbrs, top_info.unbonded_nbrs)
    _compute_subterms = Partial(_compute_subterms, seq=seq, params=params)

    trajectory_subterms = list()
    for s in traj_info.states:
        s_subterms = _compute_subterms(s)
        avg_s_subterms = np.array(s_subterms) / n # average per nucleotide
        trajectory_subterms.append(avg_s_subterms)
    return np.array(trajectory_subterms)

def get_oxpy_subterms(top_path, traj_path, input_path):
    # top_info, traj_info = describe(top_path, traj_path)
    # output_bonds(traj_info, top_info, input_path, visualize=False)
    basedir = Path(top_path).parent
    split_energy_fname = basedir / "split_energy.dat"
    if not split_energy_fname.exists():
        raise RuntimeError(f"No file exists at location: {split_energy_fname}")
    split_energy_df = pd.read_csv(
        split_energy_fname,
        names=["t", "fene", "b_exc", "stack", "n_exc", "hb",
                 "cr_stack", "cx_stack"],
        delim_whitespace=True)
    return split_energy_df



def run(top_path, traj_path, input_path, T=DEFAULT_TEMP):
    print(f"----Checking energy breakdown agreement for trajectory at location: {traj_path}----")

    computed_subterms = compute_subterms(top_path, traj_path, T)
    oxpy_subterms = get_oxpy_subterms(top_path, traj_path, input_path)

    # observables aren't computed for the initial state, so we drop the first row from split_energy.dat
    oxpy_subterms = oxpy_subterms.iloc[1: , :]

    # check for equality
    for i, (idx, row) in enumerate(oxpy_subterms.iterrows()): # note: i does not necessarily equal idx
        print(f"\tState {i}:")
        ith_oxpy_subterms = row.to_numpy()[1:]
        ith_computed_subterms = computed_subterms[i]
        ith_computed_subterms = np.round(ith_computed_subterms, 6)
        print(f"\t\tComputed subterms: {ith_computed_subterms}")
        print(f"\t\toxDNA subterms: {ith_oxpy_subterms}")
        print(f"\t\t|Difference|: {np.abs(ith_computed_subterms - ith_oxpy_subterms)}")
        print(f"\t\t|HB Difference|: {np.abs(ith_computed_subterms[4] - ith_oxpy_subterms[4])}")
        if not np.allclose(ith_oxpy_subterms, ith_computed_subterms, atol=1e-4, rtol=1e-8):
            print(bcolors.FAIL + "\t\tFail!\n" + bcolors.ENDC)
            pdb.set_trace()
        else:
            print(bcolors.OKGREEN + "\t\tSuccess!\n" + bcolors.ENDC)

    print(bcolors.WARNING + "WARNING: errors for hydrogen bonding and cross stalking are subject to approximation of pi in parameter file" + bcolors.ENDC)
    return
