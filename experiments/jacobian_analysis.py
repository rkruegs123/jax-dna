import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax, tree_util, jacfwd
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.loss import geometry, pitch, propeller
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

    key = random.PRNGKey(0)

    displacement_fn, shift_fn = space.free()

    sys_basedir = Path("data/sys-defs/simple-helix")
    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)


    conf_path = sys_basedir / "bound_relaxed.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path, reverse_direction=True
    )
    init_body = conf_info.get_states()[0]
    n = init_body.center.shape[0]
    dimension = 3

    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    gamma = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64))
    mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))


    params = deepcopy(model.EMPTY_BASE_PARAMS)
    em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)

    # Minimization
    init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
    state = init_fn(key, init_body, mass=mass, seq=seq_oh,
                    bonded_nbrs=top_info.bonded_nbrs,
                    unbonded_nbrs=top_info.unbonded_nbrs.T)

    minimize_steps = 1000
    log_every = 50
    energies = list()
    for i in tqdm(range(minimize_steps)):
        state = step_fn(state, seq=seq_oh,
                        bonded_nbrs=top_info.bonded_nbrs,
                        unbonded_nbrs=top_info.unbonded_nbrs.T)
        if i % log_every == 0:
            curr_energy = em.energy_fn(
                state.position, seq=seq_oh,
                bonded_nbrs=top_info.bonded_nbrs,
                unbonded_nbrs=top_info.unbonded_nbrs.T)
            energies.append(curr_energy)

    body_min = deepcopy(state.position)
    plt.plot(range(0, minimize_steps, log_every), energies)
    plt.xlabel("Minimization Iteration")
    plt.ylabel("Energy")
    plt.show()



    # Find the range of timesteps between which eigenvalues of the Jacobian are >1
    powers = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]
    blowup_power = None
    for p in powers:
        dt = 10**p # 1e(p)

        init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
        # init_fn, step_fn = simulate.brownian(energy_fn, shift_fn, dt, kt, gamma) # FIXME: won't work for rigid bodies

        def dynamical_fn(rb_flattened):
            center_flattened = rb_flattened[:n*dimension]
            center = center_flattened.reshape(n, dimension)

            quat_vec_flattened = rb_flattened[n*dimension:]
            quat_vec = quat_vec_flattened.reshape(n, 4)

            rb = rigid_body.RigidBody(center, rigid_body.Quaternion(quat_vec))

            dummy_state = init_fn(
                key, rb, mass=mass, seq=seq_oh,
                bonded_nbrs=top_info.bonded_nbrs,
                unbonded_nbrs=top_info.unbonded_nbrs.T)
            # dummy_state = init_fn(key, pos)
            stepped_rb = step_fn(dummy_state, seq=seq_oh,
                                 bonded_nbrs=top_info.bonded_nbrs,
                                 unbonded_nbrs=top_info.unbonded_nbrs.T).position
            stepped_rb_flattened = jnp.concatenate([stepped_rb.center.flatten(),
                                                    stepped_rb.orientation.vec.flatten()])
            return stepped_rb_flattened

        dynamical_jac = jit(jacfwd(dynamical_fn))
        body_min_flattened = jnp.concatenate([body_min.center.flatten(),
                                              body_min.orientation.vec.flatten()])
        jac = dynamical_jac(body_min_flattened)
        jac_evals, jac_evecs = jit(jnp.linalg.eig)(jac)
        abs_evals = jnp.abs(jac_evals)
        max_abs_eval = abs_evals.max()

        print(f"dt={dt}, max abs. eval={max_abs_eval}\n")

        # if max_abs_eval > 1.0 + 1e-1:
        if max_abs_eval > 1.0 and blowup_power is None:
            blowup_power = p
            # break

    if blowup_power is None:
        raise RuntimeError(f"Must increase the power")

    return

if __name__ == "__main__":
    run()
