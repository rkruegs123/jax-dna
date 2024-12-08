import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time

import jax
import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax, tree_util
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.loss import geometry, pitch, propeller
from jax_dna.dna1 import model
from jax_dna import dna1, loss

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)



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

    # Get the loss function terms

    ## note: we don't include the end base pairs due to fraying
    compute_helical_diameters, helical_diam_loss_fn = geometry.get_helical_diameter_loss_fn(
        top_info.bonded_nbrs[1:-1], displacement_fn, model.com_to_backbone)

    compute_bb_distances, bb_dist_loss_fn = geometry.get_backbone_distance_loss_fn(
        top_info.bonded_nbrs, displacement_fn, model.com_to_backbone)

    simple_helix_quartets = jnp.array([
        [1, 14, 2, 13], [2, 13, 3, 12],
        [3, 12, 4, 11], [4, 11, 5, 10],
        [5, 10, 6, 9]])
    compute_avg_pitch, pitch_loss_fn = pitch.get_pitch_loss_fn(
        simple_helix_quartets, displacement_fn, model.com_to_hb)

    simple_helix_bps = jnp.array([[1, 14], [2, 13], [3, 12],
                                  [4, 11], [5, 10], [6, 9]])
    compute_avg_p_twist, p_twist_loss_fn = propeller.get_propeller_loss_fn(simple_helix_bps)


    conf_path = sys_basedir / "bound_relaxed.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path, reverse_direction=True
    )
    init_body = conf_info.get_states()[0]

    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    gamma_eq = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64))
    # gamma_scale = 2500
    gamma_scale = 1
    gamma_opt = rigid_body.RigidBody(
        center=gamma_eq.center * gamma_scale,
        orientation=gamma_eq.orientation * gamma_scale)
    mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))

    def sim_fn(params, body, n_steps, key, gamma):
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

    num_eq_steps = 10000
    eq_fn = lambda params, key: sim_fn(params, init_body, num_eq_steps, key, gamma_eq)
    eq_fn = jit(eq_fn)

    def body_metadata_fn(body):
        helical_diams = compute_helical_diameters(body)
        mean_helical_diam = jnp.mean(helical_diams)

        bb_dists = compute_bb_distances(body)
        mean_bb_dist = jnp.mean(bb_dists)

        mean_pitch = compute_avg_pitch(body)

        mean_p_twist = compute_avg_p_twist(body)

        return (mean_helical_diam, mean_bb_dist, mean_pitch, mean_p_twist)


    @jit
    def body_loss_fn(body):
        loss = helical_diam_loss_fn(body) + bb_dist_loss_fn(body) \
               + pitch_loss_fn(body) + p_twist_loss_fn(body)
        return loss, body_metadata_fn(body)

    # note: assumes trajectory begins in equilibrium
    sample_every = 100
    @jit
    def traj_loss_fn(traj):
        states_to_eval = traj[::sample_every]
        losses, all_metadata = vmap(body_loss_fn)(states_to_eval)
        return losses.mean(), all_metadata

    # num_steps = 50000
    num_steps = 10000
    @jit
    def loss_fn(params, eq_body, key):
        fin_pos, traj = sim_fn(params, eq_body, num_steps, key, gamma_opt)
        loss, metadata = traj_loss_fn(traj)
        return loss, metadata

    params = deepcopy(model.EMPTY_BASE_PARAMS)

    batch_size = 10
    key = random.PRNGKey(0)
    mapped_eq_fn = jit(vmap(eq_fn, (None, 0)))


    key, iter_key = random.split(key)
    iter_key, eq_key = random.split(iter_key)
    eq_keys = random.split(eq_key, batch_size)
    em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
    eq_bodies, _ = mapped_eq_fn(params, eq_keys)

    batch_keys = random.split(iter_key, batch_size)
    start = time.time()
    losses, batched_metadata = vmap(loss_fn, (None, 0, 0))(params, eq_bodies, batch_keys)
    end = time.time()
    iter_time = end - start

    pdb.set_trace()


    import numpy as onp
    import matplotlib.pyplot as plt

    prop_twists = batched_metadata[-1].flatten()
    running_avg_ptwist = prop_twists.cumsum() / onp.arange(1, 1001)
    plt.plot(running_avg_ptwist)
    plt.show()

    return

if __name__ == "__main__":
    run()
