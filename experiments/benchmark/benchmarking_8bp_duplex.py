import pdb
from pathlib import Path
import shutil
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp
import subprocess

import jax
import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax, lax, tree_util
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, checkpoint
from jax_dna.common.trajectory import TrajectoryInfo
from jax_dna.common.topology import TopologyInfo
from jax_dna.loss import geometry, pitch, propeller
from jax_dna.dna1 import model, oxdna_utils

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)



checkpoint_every = 50
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def run(args):


    run_name = args['run_name']
    n_steps = args['n_steps']
    oxdna_path = Path(args['oxdna_path'])
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"
    device = args['device']
    if device == "cpu":
        backend = "CPU"
    elif device == "gpu":
        backend = "CUDA"
    else:
        raise RuntimeError(f"Invalid device: {device}")
    n_threads = args['n_threads']
    oxdna_cuda_device = args['oxdna_cuda_device']
    oxdna_cuda_list = args['oxdna_cuda_list']

    if run_name is None:
        raise RuntimeError(f"Must set a run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    # Load the system
    sys_basedir = Path("data/sys-defs/simple-helix")
    top_path = sys_basedir / "sys.top"
    top_info = TopologyInfo(top_path, reverse_direction=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    conf_path = sys_basedir / "bound_relaxed.conf"
    conf_info = TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path, reverse_direction=True
    )

    # Setup utilities for simulation
    displacement_fn, shift_fn = space.free()

    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    gamma = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64))
    mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))

    params = deepcopy(model.EMPTY_BASE_PARAMS)
    params = tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.float64), params)
    key = random.PRNGKey(0)

    # JAX-MD Simulation
    init_body = conf_info.get_states()[0]

    em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
    init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
    init_state = init_fn(key, init_body, mass=mass, seq=seq_oh,
                         bonded_nbrs=top_info.bonded_nbrs,
                         unbonded_nbrs=top_info.unbonded_nbrs.T)
    step_fn = jit(step_fn)


    ## Option 1: For loop
    start = time.time()
    trajectory = list()
    timestep_times = list()
    state = init_state
    post_compilation_index = 2
    for i in tqdm(range(n_steps)):
        if i == post_compilation_index:
            post_comp_start = time.time()

        step_start = time.time()
        state = step_fn(state,
                        seq=seq_oh,
                        bonded_nbrs=top_info.bonded_nbrs,
                        unbonded_nbrs=top_info.unbonded_nbrs.T)
        step_end = time.time()
        timestep_times.append(step_end - step_start)

        trajectory.append(state.position)
    post_comp_end = time.time()
    end = time.time()
    traj = utils.tree_stack(trajectory)
    print(f"Post compilation took {post_comp_end - post_comp_start} seconds")

    pdb.set_trace()

    ## Option 2: Scan

    @jit
    def scan_fn(state, step):
        state = step_fn(state,
                        seq=seq_oh,
                        bonded_nbrs=top_info.bonded_nbrs,
                        unbonded_nbrs=top_info.unbonded_nbrs.T)
        return state, state.position

    sim_fn = lambda: scan(scan_fn, init_state, jnp.arange(n_steps))
    sim_fn = jit(sim_fn)

    start = time.time()
    # fin_state, traj = scan(scan_fn, init_state, jnp.arange(n_steps))
    fin_state, traj = sim_fn()
    end = time.time()
    print(f"Initial scan took {end - start} seconds")

    start = time.time()
    fin_state, traj = sim_fn()
    end = time.time()
    print(f"Second scan took {end - start} seconds")

    start = time.time()
    fin_state, traj = sim_fn()
    end = time.time()
    print(f"Third scan took {end - start} seconds")


    ## Option 2: Gradients of scan

    """
    def dummy_loss_fn(iter_params):
        em = model.EnergyModel(displacement_fn, iter_params, t_kelvin=t_kelvin)
        init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(key, init_body, mass=mass, seq=seq_oh,
                             bonded_nbrs=top_info.bonded_nbrs,
                             unbonded_nbrs=top_info.unbonded_nbrs.T)
        step_fn = jit(step_fn)

        @jit
        def scan_fn(state, step):
            state = step_fn(state,
                            seq=seq_oh,
                            bonded_nbrs=top_info.bonded_nbrs,
                            unbonded_nbrs=top_info.unbonded_nbrs.T)
            return state, state.position

        fin_state, traj = scan(scan_fn, init_state, jnp.arange(n_steps))

        return fin_state.position.center.sum()
    grad_fn = grad(dummy_loss_fn)

    params_grads = deepcopy(model.EMPTY_BASE_PARAMS)
    params_grads["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]

    start = time.time()
    grads = grad_fn(params_grads)
    end = time.time()
    print(f"Initial gradient calculation took {end - start} seconds")


    start = time.time()
    grads = grad_fn(params_grads)
    end = time.time()
    print(f"Second gradient calculation took {end - start} seconds")
    """






    # Run with oxDNA standalone code on CPU
    sys_basedir_template = Path("data/templates/simple-helix")
    input_template_path = sys_basedir_template / "input"

    box_size = conf_info.box_size

    sim_dir = run_dir / f"standalone"
    sim_dir.mkdir(parents=False, exist_ok=False)

    shutil.copy(top_path, sim_dir / "sys.top")
    init_conf_info = TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        # reverse_direction=True
        reverse_direction=False
    )
    init_conf_info.write(sim_dir / "init.conf", reverse=False, write_topology=False)

    oxdna_utils.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)

    seed = 0
    sample_every = 100

    oxdna_utils.rewrite_input_file(
        input_template_path, sim_dir,
        temp=f"{t_kelvin}K", steps=n_steps,
        init_conf_path=str(sim_dir / "init.conf"),
        top_path=str(sim_dir / "sys.top"),
        save_interval=sample_every, seed=seed,
        equilibration_steps=0, dt=dt,
        backend=backend,
        cuda_device=oxdna_cuda_device, cuda_list=oxdna_cuda_list,
        log_file=str(sim_dir / "sim.log"),
    )
    input_path = sim_dir / "input"

    start = time.time()
    oxdna_process = subprocess.run([oxdna_exec_path, input_path])
    rc = oxdna_process.returncode
    if rc != 0:
        raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")
    end = time.time()
    print(f"Standalone simulation tooke {end - start} seconds")




    return




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmarking simulation times")
    parser.add_argument('--device', type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--n-steps', type=int, default=10000, help='# steps')
    parser.add_argument('--oxdna-path', type=str,
                        default="/n/holylabs/LABS/brenner_lab/Users/rkrueger/oxdna-bin-cpu1/oxDNA",
                        help='oxDNA base directory')
    parser.add_argument('--oxdna-cuda-device', type=int, default=0,
                        help="CUDA device for running oxDNA simulations")
    parser.add_argument('--oxdna-cuda-list', type=str, default="verlet",
                        choices=["no", "verlet"],
                        help="CUDA neighbor lists")

    args = vars(parser.parse_args())

    run(args)
