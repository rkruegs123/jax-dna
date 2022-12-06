from tqdm import tqdm
import pdb
from pathlib import Path
import shutil
import datetime
import matplotlib.pyplot as plt
import numpy as onp

from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax import jit
from jax.tree_util import Partial

from jax_md.rigid_body import RigidBody
from jax_md import space, util, simulate


from energy import factory # FIXME: will want to replace with a different energy function
# import langevin
from loader import get_params
from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo
from metadynamics import cv
from metadynamics.utils import get_height_fn
from metadynamics.utils import sum_of_gaussians
import metadynamics.energy as md_energy
from utils import bcolors
from utils import nucleotide_mass, get_kt, moment_of_inertia, get_one_hot, DEFAULT_TEMP
from utils import base_site, stack_site, back_site



# Note: may not have to run these in batches as the reconstructed free energy landscape may not be (too) stochastic

f64 = util.f64

def run_single_metad(top_path, conf_path, bps,
                     n_steps, stride, n_gaussians,
                     key, T=DEFAULT_TEMP, dt=5e-3, save_every=10, plot_every=None,
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
        run_name = f"metad_{timestamp}"
        run_dir = output_basedir / run_name
        run_dir.mkdir(parents=False, exist_ok=False)
        shutil.copy(top_path, run_dir)
        shutil.copy(conf_path, run_dir)
        params_str = f"topology file: {top_path}\nconfiguration file: {conf_path}\nn_steps: {n_steps}\nkey: {key}\ntemperature: {T}\ndt: {dt}\nsave_every: {save_every}\n"
        with open(run_dir / "params.txt", "w+") as f:
            f.write(params_str)
        print(bcolors.WARNING + f"Created directory and copied simulation information at location: {run_dir}" + bcolors.ENDC)


    print(bcolors.OKBLUE + f"Setting up simulation..." + bcolors.ENDC)
    # Typical information
    mass = RigidBody(center=jnp.array([nucleotide_mass]),
                     orientation=jnp.array([moment_of_inertia]))

    gamma = RigidBody(center=jnp.array([DEFAULT_TEMP/2.5]),
                      orientation=jnp.array([DEFAULT_TEMP/7.5]))

    params = [2.0, 0.25, 0.7525]

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    displacement_fn, shift_fn = space.periodic(config_info.box_size)

    body = config_info.states[0]
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT


    # Metapotential information
    height_0 = 1.0
    width_0 = 0.25
    centers = jnp.zeros(n_gaussians, dtype=f64)
    widths = jnp.full(n_gaussians, width_0, dtype=f64)
    heights = jnp.zeros(n_gaussians, dtype=f64)
    # height_fn = get_height_fn(height_0, well_tempered=False)
    height_fn = get_height_fn(height_0, well_tempered=True, kt=kT, delta_T=20.0)
    n_bp_fn = cv.get_n_bp_fn(bps, displacement_fn)

    height_fn = jit(height_fn)
    n_bp_fn = jit(n_bp_fn)


    # Wrap the energy function
    base_energy_fn, compute_subterms = factory.energy_fn_factory(displacement_fn,
                                                                 back_site, stack_site, base_site,
                                                                 top_info.bonded_nbrs, top_info.unbonded_nbrs)
    compute_subterms = jit(Partial(compute_subterms, seq=seq, params=params))
    base_energy_fn = Partial(base_energy_fn, seq=seq, params=params)
    # energy_fn = jit(Partial(energy_fn, seq=seq, params=params))

    md_energy_fn = md_energy.factory(base_energy_fn, n_bp_fn)
    md_energy_fn = jit(md_energy_fn)

    # init_fn, step_fn = langevin.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
    init_fn, step_fn = simulate.nvt_langevin(md_energy_fn, shift_fn, dt, kT, gamma)
    step_fn = jit(step_fn)

    # state = init_fn(key, body, mass=mass, seq=seq, params=params)
    state = init_fn(key, body, mass=mass,
                    heights=heights, centers=centers, widths=widths)


    trajectory = [state.position]
    energies = [md_energy_fn(state.position,
                             heights=heights, centers=centers, widths=widths)]
    subterms = [compute_subterms(state.position)]
    print(bcolors.OKBLUE + f"Starting simulation..." + bcolors.ENDC)
    for i in tqdm(range(n_steps), colour="blue"):
        # state = step_fn(state, seq=seq, params=params)
        state = step_fn(state, heights=heights, centers=centers, widths=widths)

        if i % stride == 0:
            iter_cv = n_bp_fn(state.position)
            iter_bias = sum_of_gaussians(heights, centers, widths, iter_cv)
            num_gauss = i // stride
            # widths = widths.at[num_guass].set(width_0)
            heights = heights.at[num_gauss].set(height_fn(iter_bias))
            centers = centers.at[num_gauss].set(iter_cv)

        if i % save_every == 0:
            energies.append(md_energy_fn(state.position,
                                         heights=heights, centers=centers, widths=widths))
            trajectory.append(state.position)
            subterms.append(compute_subterms(state.position))

        if plot_every and i % plot_every == 0:
            pdb.set_trace()
            # Plot the metapotential
            test_cvs = onp.linspace(-2, 10, 200)
            # test_cvs = onp.linspace(0, 10, 40)
            biases = [sum_of_gaussians(heights, centers, widths, tmp_cv) for tmp_cv in test_cvs]
            plt.plot(test_cvs, biases)
            plt.show()
            plt.clf()


    final_traj = TrajectoryInfo(top_info ,states=trajectory, box_size=config_info.box_size)
    if save_output:
        print(bcolors.OKBLUE + f"Writing trajectory to file..." + bcolors.ENDC)
        final_traj.write(run_dir / "output.dat", reverse=True, write_topology=False)

    pdb.set_trace()
    return final_traj, energies

if __name__ == "__main__":
    top_path = "data/simple-helix/generated.top"
    # conf_path = "data/simple-helix/start.conf"
    conf_path = "data/simple-helix/unbound.conf"
    key = random.PRNGKey(0)

    n_steps = int(1e5)
    stride = 100
    n_gaussians = n_steps // stride

    bps = jnp.array([
        [0, 15],
        [1, 14],
        [2, 13],
        [3, 12],
        [4, 11],
        [5, 10],
        [6, 9],
        [7, 8]
    ])

    run_single_metad(top_path, conf_path, bps,
                     n_steps, stride, n_gaussians,
                     key, save_every=100, save_output=True,
                     # plot_every=10000,
                     dt=5e-3
    )
