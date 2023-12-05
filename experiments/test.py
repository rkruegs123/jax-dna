import pdb
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp
import subprocess
import shutil
import pandas as pd
import random
from copy import deepcopy
import seaborn as sns

import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.dna1 import model, oxdna_utils
from jax_dna.loss import tm

from jax.config import config
config.update("jax_enable_x64", True)



params = deepcopy(model.EMPTY_BASE_PARAMS)
params["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]

box_size = 15.0

displacement_fn, shift_fn = space.periodic(box_size)
t_kelvin = 312.15
kT = utils.get_kt(t_kelvin)
beta = 1 / kT

sys_basedir = Path("data/templates/tm-8bp")
top_path = sys_basedir / "sys.top"
top_info = topology.TopologyInfo(top_path,
                                 # reverse_direction=False
                                 reverse_direction=True
)
seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)


iter_dir = Path("/n/holylabs/LABS/brenner_lab/Users/rkrueger/jaxmd-oxdna/output/tm-sim-12_sims-5e7_steps-2.5e5_sample/ref_traj/iter0")


n_sims = 12

traj_info = trajectory.TrajectoryInfo(
    top_info, read_from_file=True,
    traj_path=iter_dir / "output.dat",
    # reverse_direction=False)
    reverse_direction=True
)
ref_states = traj_info.get_states()
n_ref_states = len(ref_states)
ref_states = utils.tree_stack(ref_states)

## Load the oxDNA energies
energy_df_columns = [
    "time", "potential_energy", "kinetic_energy", "total_energy",
    "op_idx", "op", "op_weight"
]
energy_dfs = [pd.read_csv(iter_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
                          delim_whitespace=True)[1:] for r in range(n_sims)]
energy_df = pd.concat(energy_dfs, ignore_index=True)


em_base = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
energy_fn = lambda body: em_base.energy_fn(
    body,
    seq=seq_oh,
    bonded_nbrs=top_info.bonded_nbrs,
    unbonded_nbrs=top_info.unbonded_nbrs.T)
energy_fn = jit(energy_fn)

ref_energies = list()
for rs_idx in tqdm(range(n_ref_states), desc="Calculating energies"):
    rs = ref_states[rs_idx]
    ref_energies.append(energy_fn(rs))
ref_energies = jnp.array(ref_energies)

gt_energies = energy_df.potential_energy.to_numpy() * seq_oh.shape[0]

atol_places = 3
tol = 10**(-atol_places)
energy_diffs = list()
for i, (calc, gt) in enumerate(zip(ref_energies, gt_energies)):
    diff = onp.abs(calc - gt)
    if diff > tol:
        print(f"WARNING: energy difference of abs({calc} - {gt}) = {diff}")
        # pdb.set_trace() # note: in practice, we wouldn't set a trace
    energy_diffs.append(diff)


sns.distplot(ref_energies, label="Calculated", color="red")
sns.distplot(gt_energies, label="Reference", color="green")
plt.legend()
# plt.show()
plt.savefig("energies.png")
plt.clf()

sns.histplot(energy_diffs)
plt.savefig("energy_diffs.png")
# plt.show()
plt.clf()
