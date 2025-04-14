# ruff: noqa
import pdb
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

import jax.numpy as jnp
import numpy as onp
from jax import jit, random, vmap
from jax_md import energy, quantity, rigid_body, simulate, space, util
from tqdm import tqdm

Array = util.Array
f32 = util.f32
ShiftFn = space.ShiftFn
T = simulate.T
InitFn = simulate.InitFn
ApplyFn = simulate.ApplyFn
Simulator = simulate.Simulator


def center_stochastic_step(state: simulate.NVTLangevinState, dt: float, kT: float, gamma: float):
    """A single stochastic step (the `O` step)."""
    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt(kT * (1 - c1**2))
    momentum_dist = simulate.Normal(c1 * state.momentum, c2**2 * state.mass)
    key, split = random.split(state.rng)

    sampled_momentum = momentum_dist.sample(split)
    log_probs = momentum_dist.log_prob(sampled_momentum)

    return state.set(momentum=sampled_momentum, rng=key), jnp.mean(log_probs)


def stochastic_step(state: rigid_body.RigidBody, dt: float, kT: float, gamma: float):
    key, center_key, orientation_key = random.split(state.rng, 3)

    rest, center, orientation = rigid_body.split_center_and_orientation(state)

    # center = simulate.stochastic_step(
    center, center_log_prob = center_stochastic_step(center.set(rng=center_key), dt, kT, gamma.center)

    Pi = orientation.momentum.vec
    I = orientation.mass
    G = gamma.orientation

    M = 4 / jnp.sum(1 / I, axis=-1)
    Q = orientation.position.vec
    P = rigid_body.MOMENTUM_PERMUTATION

    # First evaluate PI term
    Pi_mean = 0
    for l in range(3):
        I_l = I[:, [l], None]
        M_l = M[:, None, None]
        PP = P[l](Q)[:, None, :] * P[l](Q)[:, :, None]
        Pi_mean += jnp.exp(-G * M_l * dt / (4 * I_l)) * PP
    Pi_mean = jnp.einsum("nij,nj->ni", Pi_mean, Pi)

    # Then evaluate Q term
    Pi_var = 0
    for l in range(3):
        scale = jnp.sqrt(4 * kT * I[:, l] * (1 - jnp.exp(-M * G * dt / (2 * I[:, l]))))
        Pi_var += (scale[:, None] * P[l](Q)) ** 2

    momentum_dist = simulate.Normal(Pi_mean, Pi_var)
    sampled_q_momentum = momentum_dist.sample(orientation_key)
    q_log_probs = momentum_dist.log_prob(sampled_q_momentum)
    new_momentum = rigid_body.Quaternion(sampled_q_momentum)
    orientation = orientation.set(momentum=new_momentum)

    avg_log_prob = center_log_prob + jnp.mean(q_log_probs)

    return rigid_body.merge_center_and_orientation(rest.set(rng=key), center, orientation), avg_log_prob


def nvt_langevin(
    energy_or_force_fn: Callable[..., Array],
    shift_fn: ShiftFn,
    dt: float,
    kT: float,
    gamma: float = 0.1,
    center_velocity: bool = True,
    **sim_kwargs,
) -> Simulator:
    force_fn = quantity.canonicalize_force(energy_or_force_fn)

    @jit
    def init_fn(key, R, mass=f32(1.0), **kwargs):
        _kT = kwargs.pop("kT", kT)
        key, split = random.split(key)
        force = force_fn(R, **kwargs)
        state = simulate.NVTLangevinState(R, None, force, mass, key)
        state = simulate.canonicalize_mass(state)
        return simulate.initialize_momenta(state, split, _kT)

    @jit
    def step_fn(state, **kwargs):
        _dt = kwargs.pop("dt", dt)
        _kT = kwargs.pop("kT", kT)
        dt_2 = _dt / 2

        state = simulate.momentum_step(state, dt_2)
        state = simulate.position_step(state, shift_fn, dt_2, **kwargs)
        # state = simulate.stochastic_step(state, _dt, _kT, gamma)
        state, log_prob = stochastic_step(state, _dt, _kT, gamma)
        state = simulate.position_step(state, shift_fn, dt_2, **kwargs)
        state = state.set(force=force_fn(state.position, **kwargs))
        state = simulate.momentum_step(state, dt_2)

        return state, log_prob

    return init_fn, step_fn


if __name__ == "__main__":

    @partial(vmap, in_axes=(0, None))
    def rand_quat(key, dtype):
        return rigid_body.random_quaternion(key, dtype)

    for kT in [1e-3, 5e-3, 1e-2, 1e-1]:
        PARTICLE_COUNT = 40
        dtype = f32
        DYNAMICS_STEPS = 100

        N = PARTICLE_COUNT
        box_size = quantity.box_size_at_number_density(N, 0.1, 3)

        displacement, shift = space.periodic(box_size)

        key = random.PRNGKey(0)

        key, pos_key, quat_key = random.split(key, 3)

        R = box_size * random.uniform(pos_key, (N, 3), dtype=dtype)
        quat_key = random.split(quat_key, N)
        quaternion = rand_quat(quat_key, dtype)

        body = rigid_body.RigidBody(R, quaternion)
        shape = rigid_body.point_union_shape(
            rigid_body.tetrahedron.points * jnp.array([[1.0, 2.0, 3.0]], dtype), rigid_body.tetrahedron.masses
        )

        energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement), shape)

        dt = 5e-4

        gamma = rigid_body.RigidBody(0.1, 0.1)
        init_fn, step_fn = nvt_langevin(energy_fn, shift, dt, kT, gamma)

        step_fn = jit(step_fn)

        state = init_fn(key, body, mass=shape.mass())

        total_log_prob = 0.0
        for i in tqdm(range(DYNAMICS_STEPS)):
            state, log_prob = step_fn(state)
            total_log_prob += log_prob

        kT_final = rigid_body.temperature(state.position, state.momentum, state.mass)

        tol = 5e-4 if kT < 2e-3 else kT / 10
        diff = onp.abs(kT_final - kT)
        rdiff = diff / kT
        print(diff)
        print(diff < tol)
        print(rdiff)
        print(rdiff < tol)
        # self.assertAllClose(kT_final, dtype(kT), rtol=tol, atol=tol)
