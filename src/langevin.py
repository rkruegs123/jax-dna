
from collections import namedtuple

from typing import Any, Callable, TypeVar, Union, Tuple, Dict, Optional

import functools

from jax import grad, vmap
from jax import jit
from jax import random
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map, tree_reduce, tree_flatten, tree_unflatten

from jax_md import quantity
from jax_md import util
from jax_md import space
from jax_md import dataclasses
from jax_md import partition
from jax_md import smap
from jax_md.rigid_body import RigidBody 

Array = util.Array
ShiftFn = space.ShiftFn
Simulator = simulate.Simulator

@dataclasses.dataclass
class NVTLangevinState:
  position: Array
  momentum: Array
  force: Array
  mass: Array
  rng: Array

  @property
  def velocity(self) -> Array:
    return self.momentum / self.mass


q_permutation = [
      vmap(lambda q: jnp.array([-q[1], q[0], q[3], -q[2]])),
      vmap(lambda q: jnp.array([-q[2], -q[3], q[0], q[1]])),
      vmap(lambda q: jnp.array([-q[3], q[2], -q[1], q[0]])),
]


@functools.singledispatch
def stochastic_step(R: Array, P: Array, M: Array, key: Array, dt:float, kT: float, gamma: float):
  c1 = jnp.exp(-gamma * dt)
  c2 = jnp.sqrt(kT * (1 - c1**2))
  G = random.normal(key, P.shape)
  return c1 * P + c2 * jnp.sqrt(M) * G


@stochastic_step.register
def _(R: RigidBody, P: RigidBody, mass: RigidBody, key: RigidBody, dt:float, kT: float, gamma: RigidBody):
  center_key, orientation_key = random.split(key)
  P_center = stochastic_step(R.center, P.center, mass.center, center_key, dt, kT, gamma.center)
  
  # COMMENT ME
  Pi = P.orientation.vec
  eta = random.normal(orientation_key, (3,) + Pi.shape[:1] + (1,))
  I = mass.orientation
  G = gamma.orientation

  # BE CAREFUL ABOUT MULTISPECIES CASE.
  M = 4 / jnp.sum(1 / I, axis=-1)
  Q = R.orientation.vec
  S = q_permutation

  # First evaluate PI term
  Pi_new = 0
  for l in range(3):
    Pi_new += jnp.exp(-G * M * dt / (4 * I[:, l])) * S[l](Q)[:, None, :] * S[l](Q)[:, :, None]
  Pi_new = jnp.einsum('nij,nj->ni', Pi_new, Pi)

  # Then evaluate Q term
  for l in range(3):
    scale = jnp.sqrt(4 * kT * I[:, l] * (1 - jnp.exp(-M * G * dt / (2 * I[:, l]))))
    Pi_new += scale * S[l](Q * eta[l])

  return RigidBody(P_center, rigid_body.Quaternion(Pi_new))


def nvt_langevin(energy_or_force_fn: Callable[..., Array],
                 shift_fn: ShiftFn,
                 dt: float,
                 kT: float,
                 gamma: float=0.1,
                 center_velocity: bool=True,
                 **sim_kwargs) -> Simulator:
  force_fn = quantity.canonicalize_force(energy_or_force_fn)

  @jit
  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kwargs.pop('kT', kT)
    key, split = random.split(key)
    mass = canonicalize_mass(mass)
    P = initialize_momenta(R, mass, split, _kT)
    force = force_fn(R, **kwargs)
    return NVTLangevinState(R, P, force, mass, key)

  @jit
  def step_fn(state, **kwargs):
    _dt = kwargs.pop('dt', dt)
    _kT = kwargs.pop('kT', kT)
    update_fn = inner_update_fn(state.position, shift_fn, **sim_kwargs)
    def langevin_update_fn(R, P, F, M, dt, **kwargs):
      key = kwargs.get('langevin_key')
      # A step
      R, P, F, M = update_fn(R, P, F, M, dt / 2)
      # O step
      P = stochastic_step(R, P, M, key, dt, _kT, gamma)
      # A step 
      R, P, F, M = update_fn(R, P, F, M, dt / 2)
      return R, P, F, M
    rng, kwargs['langevin_key'] = random.split(state.rng)
    state = dataclasses.replace(state, rng=rng)
    return velocity_verlet(force_fn, langevin_update_fn, _dt, state, **kwargs)
  return init_fn, step_fn

