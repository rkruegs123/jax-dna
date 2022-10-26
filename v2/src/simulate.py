import pdb
import jax
from jax import jit
import jax.numpy as jnp
from jax import random
from jax.tree_util import Partial
from tqdm import tqdm
import datetime
from pathlib import Path
import shutil

from jax_md.rigid_body import RigidBody
from jax_md import space, util

from utils import nucleotide_mass, get_kt, moment_of_inertia, get_one_hot, DEFAULT_TEMP
from utils import base_site, stack_site, back_site
from utils import bcolors
import langevin
from energy import factory
from loader import get_params
from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo


f64 = util.f64

def run_single_langevin(top_path, conf_path,
                        n_steps, key, T=DEFAULT_TEMP, dt=5e-3, save_every=10,
                        output_basedir="v2/data/output/", save_output=False):

    if not Path(top_path).exists():
        raise RuntimeError(f"Topology file does not exist at location: {top_path}")

    if not Path(conf_path).exists():
        raise RuntimeError(f"Configuration file does not exist at location: {conf_path}")

    if save_output:
        output_basedir = Path(output_basedir)
        if not output_basedir.exists():
            raise RuntimeError(f"Output base directory does not exist at location: {output_basedir}")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f"langevin_{timestamp}"
        run_dir = output_basedir / run_name
        run_dir.mkdir(parents=False, exist_ok=False)
        shutil.copy(top_path, run_dir)
        shutil.copy(conf_path, run_dir)
        params_str = f"topology file: {top_path}\nconfiguration file: {conf_path}\nn_steps: {n_steps}\nkey: {key}\ntemperature: {T}\ndt: {dt}\nsave_every: {save_every}\n"
        with open(run_dir / "params.txt", "w+") as f:
            f.write(params_str)
        print(bcolors.WARNING + f"Created directory and copied simulation information at location: {run_dir}" + bcolors.ENDC)

    print(bcolors.OKBLUE + f"Setting up simulation..." + bcolors.ENDC)
    mass = RigidBody(center=jnp.array([nucleotide_mass]),
                     orientation=jnp.array([moment_of_inertia]))
    gamma = RigidBody(center=jnp.array([DEFAULT_TEMP/2.5]),
                      orientation=jnp.array([DEFAULT_TEMP/7.5]))
    params = get_params.get_default_params(t=T, no_smoothing=False)

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    displacement_fn, shift_fn = space.periodic(config_info.box_size)

    body = config_info.states[0]
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT

    energy_fn, _ = factory.energy_fn_factory(displacement_fn,
                                             back_site, stack_site, base_site,
                                             top_info.bonded_nbrs, top_info.unbonded_nbrs)
    energy_fn = Partial(energy_fn, seq=seq, params=params)

    init_fn, step_fn = langevin.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
    step_fn = jit(step_fn)


    state = init_fn(key, body, mass=mass, seq=seq, params=params)

    trajectory = [state.position]
    energies = [energy_fn(state.position)]
    print(bcolors.OKBLUE + f"Starting simulation..." + bcolors.ENDC)
    for i in tqdm(range(n_steps), colour="red"): # note: colour can be one of [hex (#00ff00), BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE]
        state = step_fn(state, seq=seq, params=params)

        if i % save_every == 0:
            energies.append(energy_fn(state.position))
            trajectory.append(state.position)

    final_traj = TrajectoryInfo(top_info ,states=trajectory, box_size=config_info.box_size)
    if save_output:
        print(bcolors.OKBLUE + f"Writing trajectory to file..." + bcolors.ENDC)
        final_traj.write(run_dir / "output.dat", reverse=True, write_topology=False)
    return final_traj, energies



if __name__ == "__main__":
    import time
    import numpy as np

    top_path = "data/simple-helix/generated.top"
    conf_path = "data/simple-helix/start.conf"
    key = random.PRNGKey(0)

    start = time.time()
    run_single_langevin(top_path, conf_path, n_steps=1000, key=key, save_output=True)
    end = time.time()
    total_time = end - start
    print(bcolors.OKGREEN + f"Finished simulation in {np.round(total_time, 2)} seconds" + bcolors.ENDC)
