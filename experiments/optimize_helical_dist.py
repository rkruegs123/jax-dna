import pdb
from pathlib import Path
from copy import deepcopy
from functools import partial

import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory
from jax_dna.loss import geometry
from jax_dna.dna1 import model
from jax_dna import dna1, loss



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
    gamma = rigid_body.RigidBody(center=jnp.array([kT/2.5], dtype=jnp.float64),
                                 orientation=jnp.array([kT/7.5], dtype=jnp.float64))
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

        fin_state, traj = lax.scan(scan_fn, init_state, jnp.arange(n_steps))
        return fin_state.position, traj

    num_eq_steps = 10
    eq_fn = lambda params, key: sim_fn(params, init_body, num_eq_steps, key)
    eq_fn = jit(eq_fn)

    @jit
    def body_loss_fn(body):
        return helical_diam_loss_fn(body)

    # note: assumes trajectory begins in equilibrium
    sample_every = 10
    @jit
    def traj_loss_fn(traj):
        states_to_eval = traj[::sample_every]
        losses = vmap(body_loss_fn)(states_to_eval)
        return losses.mean()

    num_steps = 100
    @jit
    def loss_fn(params, eq_body, key):
        fin_pos, traj = sim_fn(params, eq_body, num_steps, key)
        loss = traj_loss_fn(traj)
        return loss
    grad_fn = value_and_grad(loss_fn)
    batched_grad_fn = jit(vmap(grad_fn, (None, 0, 0)))


    params = deepcopy(model.EMPTY_BASE_PARAMS)
    params["stacking"]["eps_stack_base"] = 1.0
    params["stacking"]["a_stack_4"] = 1.0
    lr = 0.01
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    n_epochs = 3
    batch_size = 2
    key = random.PRNGKey(0)
    mapped_eq_fn = jit(vmap(eq_fn, (None, 0)))
    for _ in range(n_epochs):
        key, iter_key = random.split(key)
        iter_key, eq_key = random.split(iter_key)
        eq_keys = random.split(eq_key, batch_size)
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        eq_bodies, _ = mapped_eq_fn(params, eq_keys)

        batch_keys = random.split(iter_key, batch_size)
        losses, grads = batched_grad_fn(params, eq_bodies, batch_keys)
        pdb.set_trace()

        # FIXME: average the grads
        avg_grads = FIXME

        updates, opt_state = optimizer.update(avg_grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return

if __name__ == "__main__":
    run()
