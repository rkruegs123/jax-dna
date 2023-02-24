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
from jax_md import space, util, simulate

from utils import nucleotide_mass, get_kt, moment_of_inertia, get_one_hot, DEFAULT_TEMP
from utils import base_site, stack_site, back_site
from utils import bcolors
# import langevin
from energy import factory, ext_force
from loader import get_params
from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo


f64 = util.f64

def run_single_langevin(args,
                        T=DEFAULT_TEMP, dt=5e-3,
                        output_basedir="v2/data/output/", save_output=False):

    top_path = args['top_path']
    conf_path = args['conf_path']
    n_steps = args['n_steps']
    key = random.PRNGKey(args['key'])
    init_method = args['params']
    save_every = args['save_every']

    use_ext_force = args['use_ext_force']
    ext_force_magnitude = args['ext_force_magnitude']
    ext_force_bps1 = args['ext_force_bps1']
    ext_force_bps2 = args['ext_force_bps2']

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
        run_name = f"langevin_{timestamp}_n{n_steps}_k{args['key']}"
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
    mass = RigidBody(center=jnp.array([nucleotide_mass]),
                     orientation=jnp.array([moment_of_inertia]))

    # params = get_params.get_default_params(t=T, no_smoothing=False)

    # params = [0.10450547, 0.5336675 , 1.2209406]
    # params = [2.0, 0.25, 0.7525]

    if init_method == "random":
        init_fene_params = [0.60, 0.75, 1.1]
        init_stacking_params = [
            0.25, 0.7, 2.0, 0.4, 1.2, 1.3, 0.2, # f1(dr_stack)
            0.5, 0.35, 0.6, # f4(theta_4)
            1.5, 1.1, 0.3, # f4(theta_5p)
            2.0, 0.2, 0.75, # f4(theta_6p)
            0.7, 2.0, # f5(-cos(phi1))
            1.3, 0.8 # f5(-cos(phi2))
        ]
    elif init_method == "oxdna":

        # starting with the correct parameters
        init_fene_params = [2.0, 0.25, 0.7525]
        init_stacking_params = [
            1.3448, 2.6568, 6.0, 0.4, 0.9, 0.32, 0.75, # f1(dr_stack)
            1.30, 0.0, 0.8, # f4(theta_4)
            0.90, 0.0, 0.95, # f4(theta_5p)
            0.90, 0.0, 0.95, # f4(theta_6p)
            2.0, -0.65, # f5(-cos(phi1))
            2.0, -0.65 # f5(-cos(phi2))
        ]
    else:
        raise RuntimeError(f"Invalid parameter initialization method: {init_method}")
    params = init_fene_params + init_stacking_params


    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    # displacement_fn, shift_fn = space.periodic(config_info.box_size)
    displacement_fn, shift_fn = space.free()

    loss_fn = geometry.get_backbone_distance_loss(top_info.bonded_nbrs, displacement_fn)
    loss_fn = jit(loss_fn)

    body = config_info.states[0]
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT

    # gamma = RigidBody(center=jnp.array([DEFAULT_TEMP/2.5]),
                      # orientation=jnp.array([DEFAULT_TEMP/7.5]))
    gamma = RigidBody(center=jnp.array([kT/2.5]),
                      orientation=jnp.array([kT/7.5]))

    energy_fn, compute_subterms = factory.energy_fn_factory(displacement_fn,
                                                            back_site, stack_site, base_site,
                                                            top_info.bonded_nbrs, top_info.unbonded_nbrs)
    energy_fn = jit(Partial(energy_fn, seq=seq, params=params))
    compute_subterms = jit(Partial(compute_subterms, seq=seq, params=params))

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

    state = init_fn(key, body, mass=mass, seq=seq, params=params)

    trajectory = [state.position]
    energies = [energy_fn(state.position)]
    # losses = [loss_fn(state.position)[1]]
    losses = [loss_fn(state.position)]
    # subterms = [compute_subterms(state.position)]
    # bb_distances = [loss_fn(state.position)[0]]
    print(bcolors.OKBLUE + f"Starting simulation..." + bcolors.ENDC)
    for i in tqdm(range(n_steps), colour="red"): # note: colour can be one of [hex (#00ff00), BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE]
        state = step_fn(state, seq=seq, params=params)

        if i % save_every == 0:
            energies.append(energy_fn(state.position))
            trajectory.append(state.position)
            # losses.append(loss_fn(state.position)[1])
            losses.append(loss_fn(state.position))
            # subterms.append(compute_subterms(state.position))
            # bb_distances.append(loss_fn(state.position)[0])

    final_traj = TrajectoryInfo(top_info, states=trajectory, box_size=config_info.box_size)
    if save_output:
        print(bcolors.OKBLUE + f"Writing trajectory to file..." + bcolors.ENDC)
        final_traj.write(run_dir / "output.dat", reverse=True, write_topology=False)
    return final_traj, energies



if __name__ == "__main__":
    import time
    import numpy as np
    from loss import geometry
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
    parser.add_argument('--params', type=str,
                        default="oxdna",
                        choices=["random", "oxdna"],
                        help='Method for initializing parameters')

    # External force
    # note: https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
    parser.add_argument('--ext-force-bps1', nargs='+', help='First list of bases for external force', required=False)
    parser.add_argument('--ext-force-bps2', nargs='+', help='Second list of bases for external force', required=False)
    parser.add_argument('--use-ext-force', action='store_true')
    parser.add_argument('--ext-force-magnitude', type=float, default=1.0,
                        help="Magnitude of the external force in the z-direction")

    args = vars(parser.parse_args())

    # top_path = "data/simple-helix/generated.top"
    # conf_path = "data/simple-helix/start.conf"

    # top_path = "data/persistence-length/init.top"
    # conf_path = "data/persistence-length/init.conf"
    # conf_path = "data/persistence-length/relaxed.dat"

    # top_path = "data/single-strand/generated.top"
    # conf_path = "data/single-strand/start.conf"

    # top_path = "data/elastic-mod/generated.top"
    # conf_path = "data/elastic-mod/generated.dat"

    # key = random.PRNGKey(0)

    start = time.time()

    traj, energies = run_single_langevin(args, save_output=True)
    end = time.time()
    total_time = end - start
    print(bcolors.OKGREEN + f"Finished simulation in {np.round(total_time, 2)} seconds" + bcolors.ENDC)
