import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time

import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax, tree_util
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.loss import geometry
from jax_dna.dna1 import model
from jax_dna import dna1, loss

from jax.config import config
config.update("jax_enable_x64", True)



checkpoint_every = 10
if checkpoint_every is None:
    scan = jax.lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def run():

    displacement_fn, shift_fn = space.free()

    sys_basedir = Path("data/sys-defs/simple-helix")
    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    # note: we don't include the end base pairs due to fraying
    _, helical_diam_loss_fn = geometry.get_helical_diameter_loss_fn(
        top_info.bonded_nbrs[1:-1], displacement_fn, model.com_to_backbone)


    conf_path = sys_basedir / "bound_relaxed.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path, reverse_direction=True
    )
    init_body = conf_info.get_states()[0]

    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    gamma_scale = 1000
    gamma = rigid_body.RigidBody(center=jnp.array([kT/2.5 * gamma_scale], dtype=jnp.float64),
                                 orientation=jnp.array([kT/7.5 * gamma_scale], dtype=jnp.float64))
    mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))

    def sim_fn(params, body, n_steps, key):
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(key, body, mass=mass, seq=seq_oh,
                             bonded_nbrs=top_info.bonded_nbrs,
                             unbonded_nbrs=top_info.unbonded_nbrs.T)

        @jit
        def scan_fn(state, step):
            state = step_fn(state,
                            seq=seq_oh,
                            bonded_nbrs=top_info.bonded_nbrs,
                            unbonded_nbrs=top_info.unbonded_nbrs.T)
            return state, state.position

        fin_state, traj = scan(scan_fn, init_state, jnp.arange(n_steps))
        return fin_state.position, traj

    num_eq_steps = 5000
    eq_fn = lambda params, key: sim_fn(params, init_body, num_eq_steps, key)
    eq_fn = jit(eq_fn)

    @jit
    def body_loss_fn(body):
        return helical_diam_loss_fn(body)

    # note: assumes trajectory begins in equilibrium
    sample_every = 500
    @jit
    def traj_loss_fn(traj):
        states_to_eval = traj[::sample_every]
        losses = vmap(body_loss_fn)(states_to_eval)
        return losses.mean()

    num_steps = 25000
    @jit
    def loss_fn(params, eq_body, key):
        fin_pos, traj = sim_fn(params, eq_body, num_steps, key)
        loss = traj_loss_fn(traj)
        return loss
    grad_fn = value_and_grad(loss_fn)
    batched_grad_fn = jit(vmap(grad_fn, (None, 0, 0)))


    params = deepcopy(model.EMPTY_BASE_PARAMS)
    params["fene"]["eps_backbone"] = model.DEFAULT_BASE_PARAMS["fene"]["eps_backbone"]w
    params["fene"]["delta_backbone"] = model.DEFAULT_BASE_PARAMS["fene"]["delta_backbone"]w
    params["fene"]["r0_backbone"] = model.DEFAULT_BASE_PARAMS["fene"]["r0_backbone"]w
    params["stacking"]["eps_stack_base"] = model.DEFAULT_BASE_PARAMS["stacking"]["eps_stack_base"]
    params["stacking"]["a_stack_4"] = model.DEFAULT_BASE_PARAMS["stacking"]["a_stack_4"]
    lr = 0.01
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    n_epochs = 100
    batch_size = 10
    key = random.PRNGKey(0)
    mapped_eq_fn = jit(vmap(eq_fn, (None, 0)))

    test_output_path = "test_output.txt"
    test_loss_path = "test_loss.txt"

    for i in tqdm(range(n_epochs)):
        key, iter_key = random.split(key)
        iter_key, eq_key = random.split(iter_key)
        eq_keys = random.split(eq_key, batch_size)
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        eq_bodies, _ = mapped_eq_fn(params, eq_keys)

        batch_keys = random.split(iter_key, batch_size)
        start = time.time()
        losses, grads = batched_grad_fn(params, eq_bodies, batch_keys)
        end = time.time()
        iter_time = end - start

        avg_grads = tree_util.tree_map(jnp.mean, grads)

        iter_str = f"----- Iteration {i} -----\n"
        iter_str += f"- Time: {iter_time}\n"
        iter_str += f"- Avg. Loss: {jnp.mean(losses)}\n"
        iter_str += f"- Params: {pprint.pformat(params, indent=4)}\n"
        iter_str += f"- Avg. Grads: {pprint.pformat(avg_grads, indent=4)}\n\n"
        with open(test_output_path, "a") as f:
            f.write(iter_str)

        with open(test_loss_path, "a") as f:
            f.write(f"{jnp.mean(losses)}\n")


        updates, opt_state = optimizer.update(avg_grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return

if __name__ == "__main__":
    run()
