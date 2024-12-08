import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp
import subprocess
import shutil
import shlex
import argparse
import pandas as pd
import random

import jax
import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax

from jax_dna.common import utils, topology, trajectory, center_configuration, checkpoint
from jax_dna.loss import persistence_length, rise
from jax_dna.dna1 import model, oxdna_utils
import jax_dna.input.trajectory as jdt

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)


checkpoint_every = 5
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


compute_all_curves = vmap(persistence_length.get_correlation_curve, (0, None, None))
compute_all_rises = vmap(rise.get_avg_rises, (0, None, None, None))


offset = 4
sys_basedir = Path("data/templates/simple-helix-60bp")
input_template_path = sys_basedir / "input"

top_path = sys_basedir / "sys.top"
top_info = topology.TopologyInfo(top_path, reverse_direction=False)
seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

quartets = utils.get_all_quartets(n_nucs_per_strand=seq_oh.shape[0] // 2)
quartets = quartets[offset:-offset-1]
base_site = jnp.array([model.com_to_hb, 0.0, 0.0])

conf_path = sys_basedir / "init.conf"
conf_info = trajectory.TrajectoryInfo(
    top_info,
    read_from_file=True, traj_path=conf_path,
    reverse_direction=False

)
centered_conf_info = center_configuration.center_conf(top_info, conf_info)
box_size = conf_info.box_size

displacement_fn, shift_fn = space.free()
dt = 3e-3
t_kelvin = utils.DEFAULT_TEMP
kT = utils.get_kt(t_kelvin)
beta = 1 / kT

# basedir = Path("/home/ryan/Downloads/test-60bp-lp-dt3e-3/ref_traj/iter0/")
basedir = Path("/home/ryan/Downloads/test-60bp-short/ref_traj/iter0/")
params = deepcopy(model.EMPTY_BASE_PARAMS)


# Test a given repeat
r_idx = 0
repeat_dir = basedir / f"r{r_idx}"

# test_data_basedir = Path("data/test-data")
# repeat_dir = test_data_basedir / "simple-helix-60bp"

traj_info = trajectory.TrajectoryInfo(
    top_info, read_from_file=True,
    traj_path=repeat_dir / "output.dat",
    reverse_direction=False)
traj_states = traj_info.get_states()
n_traj_states = len(traj_states)
traj_states = utils.tree_stack(traj_states)


energy_df_columns = ["time", "potential_energy", "kinetic_energy", "total_energy"]
energy_df = pd.read_csv(repeat_dir / "energy.dat", names=energy_df_columns,
                        delim_whitespace=True)[1:]
# energy_dfs = [pd.read_csv(iter_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
#                                   delim_whitespace=True)[1:] for r in range(n_sims)]
# energy_df = pd.concat(energy_dfs, ignore_index=True)


em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
energy_fn = lambda body: em.energy_fn(
    body,
    seq=seq_oh,
    bonded_nbrs=top_info.bonded_nbrs,
    unbonded_nbrs=top_info.unbonded_nbrs.T)
energy_fn = jit(energy_fn)


energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
_, calc_energies = scan(energy_scan_fn, None, traj_states)

gt_energies = energy_df.potential_energy.to_numpy() * seq_oh.shape[0]

energy_diffs = list()
for i, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
    print(f"State {i}:")
    print(f"\t- Calc. Energy: {calc}")
    print(f"\t- Reference. Energy: {gt}")
    diff = onp.abs(calc - gt)
    print(f"\t- Difference: {diff}")


computed_rises_ang = compute_all_rises(traj_states, quartets, displacement_fn, model.com_to_hb) * utils.ang_per_oxdna_length
# avg_rise = jnp.mean(all_rises)
running_avg_rises_ang = onp.cumsum(computed_rises_ang) / onp.arange(1, len(computed_rises_ang) + 1)
plt.plot(running_avg_rises_ang)
plt.title(f"{(offset+1)*2} skipped quartets")
plt.xlabel("num state")
plt.ylabel("avg. rise (A)")
plt.show()
plt.close()






pdb.set_trace()


# Test all repeats
n_sims = 8
energy_dfs = [pd.read_csv(basedir / f"r{r}" / "energy.dat", names=energy_df_columns,
                          delim_whitespace=True)[1:] for r in range(n_sims)]
energy_df = pd.concat(energy_dfs, ignore_index=True)

"""
traj_ = jdt.from_file(
    basedir / "output.dat",
    [strand_length, strand_length],
    is_oxdna=False,
    n_processes=n_threads,
)
traj_states = [ns.to_rigid_body() for ns in traj_.states]
n_traj_states = len(traj_states)
traj_states = utils.tree_stack(traj_states)
"""
traj_info = trajectory.TrajectoryInfo(
    top_info, read_from_file=True,
    traj_path=basedir / "output.dat",
    reverse_direction=False)
traj_states = traj_info.get_states()
n_traj_states = len(traj_states)
traj_states = utils.tree_stack(traj_states)

_, calc_energies = scan(energy_scan_fn, None, traj_states)

gt_energies = energy_df.potential_energy.to_numpy() * seq_oh.shape[0]

energy_diffs = list()
for i, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
    print(f"State {i}:")
    print(f"\t- Calc. Energy: {calc}")
    print(f"\t- Reference. Energy: {gt}")
    diff = onp.abs(calc - gt)
    print(f"\t- Difference: {diff}")
    energy_diffs.append(diff)

pdb.set_trace()
