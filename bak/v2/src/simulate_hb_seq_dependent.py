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
from functools import partial

from jax_md.rigid_body import RigidBody
from jax_md import space, util, simulate

from utils import nucleotide_mass, get_kt, moment_of_inertia, get_one_hot, DEFAULT_TEMP
from utils import base_site, stack_site, back_site
from utils import bcolors
# import langevin
# from energy import factory, ext_force
from energy import ext_force
from energy import hb_seq_dependent_factory as factory
from loader import get_params
from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo

from jax.config import config
config.update("jax_enable_x64", True)


f64 = util.f64

def run_single_langevin(args,
                        output_basedir="v2/data/output/", save_output=False):

    top_path = args['top_path']
    conf_path = args['conf_path']
    n_steps = args['n_steps']
    key = random.PRNGKey(args['key'])
    init_method = args['params']
    save_every = args['save_every']
    dt = args['dt']
    boundary_conditions = args['boundary_conditions']
    T = args['temp']


    use_ext_force = args['use_ext_force']
    ext_force_magnitude = args['ext_force_magnitude']
    ext_force_bps1 = args['ext_force_bps1']
    ext_force_bps2 = args['ext_force_bps2']

    use_neighbor_lists = args['use_neighbor_lists']
    if use_neighbor_lists and boundary_conditions != "periodic":
        raise RuntimeError(f"Can only use neighbor lists with periodic boundary conditions")
    r_cutoff = args['r_cutoff']
    dr_threshold = args['dr_threshold']

    if use_ext_force and (not ext_force_magnitude or not ext_force_bps1 or not ext_force_bps2):
        raise RuntimeError(f"Must provide external force parameters if using an external force")

    if use_ext_force:
        for nuc_id in ext_force_bps1 + ext_force_bps2:
            if not nuc_id.isdigit():
                raise RuntimeError(f"External force nucleotide IDs must be integers")
        ext_force_bps1 = [int(nuc_id) for nuc_id in ext_force_bps1]
        ext_force_bps2 = [int(nuc_id) for nuc_id in ext_force_bps2]

    if not Path(top_path).exists():
        raise RuntimeError(f"Topology file does not exist at location: {top_path}")

    if not Path(conf_path).exists():
        raise RuntimeError(f"Configuration file does not exist at location: {conf_path}")

    if save_output:
        output_basedir = Path(output_basedir)
        if not output_basedir.exists():
            raise RuntimeError(f"Output base directory does not exist at location: {output_basedir}")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f"langevin_{timestamp}_n{n_steps}_dt{dt}_t{T}_k{args['key']}"
        run_dir = output_basedir / run_name
        run_dir.mkdir(parents=False, exist_ok=False)
        shutil.copy(top_path, run_dir)
        shutil.copy(conf_path, run_dir)

        params_str = ""
        for k, v in args.items():
            params_str += f"{k}: {v}\n"
        with open(run_dir / "params.txt", "w+") as f:
            f.write(params_str)
        print(bcolors.WARNING + f"Created directory and copied simulation information at location: {run_dir}" + bcolors.ENDC)

    print(bcolors.OKBLUE + f"Setting up simulation..." + bcolors.ENDC)
    mass = RigidBody(center=jnp.array([nucleotide_mass], dtype=f64),
                     orientation=jnp.array([moment_of_inertia], dtype=f64))

    params = get_params.get_init_optimize_params_hb_seq_dependent(init_method)
    params = jnp.array(params)

    top_info = TopologyInfo(top_path, reverse_direction=True)

    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    if boundary_conditions == "periodic":
        displacement_fn, shift_fn = space.periodic(config_info.box_size)
    elif boundary_conditions == "free":
        displacement_fn, shift_fn = space.free()
    else:
        raise RuntimeError(f"Invalid boundary conditions: {boundary_conditions}")

    body = config_info.states[0]
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT

    # gamma = RigidBody(center=jnp.array([DEFAULT_TEMP/2.5]),
                      # orientation=jnp.array([DEFAULT_TEMP/7.5]))
    gamma = RigidBody(center=jnp.array([kT/2.5], dtype=f64),
                      orientation=jnp.array([kT/7.5], dtype=f64))


    energy_fn, compute_subterms = factory.energy_fn_factory(
        displacement_fn,
        back_site, stack_site, base_site,
        top_info.bonded_nbrs, top_info.unbonded_nbrs
    )
    if use_neighbor_lists:
        energy_fn = jit(partial(energy_fn, seq=seq, params=params))
        compute_subterms = jit(partial(compute_subterms, seq=seq, params=params))

        neighbor_fn = top_info.get_neighbor_list_fn(displacement_fn, config_info.box_size,
                                                    r_cutoff, dr_threshold)
        neighbors = neighbor_fn.allocate(body.center) # We use the COMs
    else:
        energy_fn = jit(partial(energy_fn, seq=seq, params=params,
                                op_nbrs_idx=top_info.unbonded_nbrs.T))
        compute_subterms = jit(partial(compute_subterms, seq=seq, params=params,
                                       op_nbrs_idx=top_info.unbonded_nbrs.T))

    if use_ext_force:
        _, force_fn = ext_force.get_force_fn(energy_fn, top_info.n, displacement_fn,
                                             # [0, 15],
                                             # [201, 202],
                                             # [0],
                                             # [0, 219],
                                             # [109, 110], # (108, 111), (106, 113), (104, 115), (100, 119)
                                             ext_force_bps1,
                                             [0, 0, ext_force_magnitude], [0, 0, 0, 0])
        _, force_fn = ext_force.get_force_fn(force_fn, top_info.n, displacement_fn,
                                             # [7, 8],
                                             # [0, 403],
                                             # [7],
                                             # [109, 110],
                                             # [0, 219], # (1, 218), (3, 216), (5, 214), (7, 212), (9, 210)
                                             ext_force_bps2,
                                             [0, 0, -ext_force_magnitude], [0, 0, 0, 0])
        force_fn = jit(force_fn)
        init_fn, step_fn = simulate.nvt_langevin(force_fn, shift_fn, dt, kT, gamma)
    else:
        # init_fn, step_fn = langevin.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)


    step_fn = jit(step_fn)

    if use_neighbor_lists:
        state = init_fn(key, body, mass=mass, op_nbrs_idx=neighbors.idx) # , seq=seq, params=params) # FIXME: why include seq and params here?
    else:
        state = init_fn(key, body, mass=mass)


    trajectory = [state.position]
    if use_neighbor_lists:
        init_energy = energy_fn(state.position, op_nbrs_idx=neighbors.idx)
    else:
        init_energy = energy_fn(state.position)
    energies = [init_energy]
    print(bcolors.OKBLUE + f"Starting simulation..." + bcolors.ENDC)

    if use_neighbor_lists:
        def iter_fn(state, neighbors):
            state = step_fn(state, op_nbrs_idx=neighbors.idx)
            neighbors = neighbors.update(state.position.center)
            return state, neighbors
    else:
        neighbors = None
        def iter_fn(state, neighbors):
            state = step_fn(state)
            return state, neighbors
    iter_fn = jit(iter_fn)

    real_neighbors = set(zip(top_info.unbonded_nbrs.T[0], top_info.unbonded_nbrs.T[1]))
    for i in tqdm(range(n_steps), colour="red"): # note: colour can be one of [hex (#00ff00), BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE]
        state, neighbors = iter_fn(state, neighbors)

        if i % save_every == 0:
            if use_neighbor_lists:
                i_energy = energy_fn(state.position, op_nbrs_idx=neighbors.idx)
            else:
                i_energy = energy_fn(state.position)
            energies.append(i_energy)
            trajectory.append(state.position)

    final_traj = TrajectoryInfo(top_info, states=trajectory, box_size=config_info.box_size)
    if save_output:
        print(bcolors.OKBLUE + f"Writing trajectory to file..." + bcolors.ENDC)
        final_traj.write(run_dir / "output.dat", reverse=True, write_topology=False)
    return final_traj, energies



if __name__ == "__main__":
    import time
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser(description="Conduct an oxDNA simulation")
    parser.add_argument('--top-path', type=str,
                        default="data/simple-helix/generated.top",
                        help='Path to topology file')
    parser.add_argument('--conf-path', type=str,
                        default="data/simple-helix/start.conf",
                        help='Path to configuration file')
    parser.add_argument('-n', '--n-steps', type=int, default=1000,
                        help="Num. steps per simulation")
    parser.add_argument('--save-every', type=int, default=10,
                        help="Frequency of saving data from optimization")
    parser.add_argument('-k', '--key', type=int, default=0,
                        help="Random key")
    parser.add_argument('--dt', type=float, default=5e-3,
                        help="Time step")
    parser.add_argument('--temp', type=float, default=DEFAULT_TEMP,
                        help="Temperature (K)")
    parser.add_argument('--params', type=str,
                        default="oxdna",
                        choices=["random", "oxdna"],
                        help='Method for initializing parameters')
    parser.add_argument('--boundary-conditions', type=str,
                        default="periodic",
                        choices=["periodic", "free"],
                        help='Boundary conditions')
    parser.add_argument('--save-output', action='store_true')

    # External force
    # note: https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
    parser.add_argument('--ext-force-bps1', nargs='+', help='First list of bases for external force', required=False)
    parser.add_argument('--ext-force-bps2', nargs='+', help='Second list of bases for external force', required=False)
    parser.add_argument('--use-ext-force', action='store_true')
    parser.add_argument('--ext-force-magnitude', type=float, default=1.0,
                        help="Magnitude of the external force in the z-direction")

    # Neighbor lists
    parser.add_argument('--use-neighbor-lists', action='store_true')
    parser.add_argument('--r-cutoff', type=float, default=10.0, help="r_cutoff for Verlet lists")
    parser.add_argument('--dr-threshold', type=float, default=0.2, help="dr_threshold for Verlet lists")

    args = vars(parser.parse_args())
    save_output = args['save_output']
    if not save_output:
        print(bcolors.WARNING + "Are you sure you don't want to save the output? Press `c` to continue:" + bcolors.ENDC)
        pdb.set_trace()


    start = time.time()

    traj, energies = run_single_langevin(args, save_output=save_output)
    end = time.time()
    total_time = end - start
    print(bcolors.OKGREEN + f"Finished simulation in {np.round(total_time, 2)} seconds" + bcolors.ENDC)
