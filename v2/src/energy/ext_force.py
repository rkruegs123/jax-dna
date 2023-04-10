import pdb

import sys
sys.path.append('v2/src/')

import jax.numpy as jnp
from jax import jit
from jax.tree_util import Partial

from jax_md import quantity, util, rigid_body

from energy import factory
from utils import base_site, stack_site, back_site

f64 = util.f64


def get_force_fn(base_energy_or_force_fn, n, displacement_fn, nuc_ids, center_force, orientation_force):

    center_force = jnp.array(center_force)
    assert(len(center_force.shape) == 1)
    assert(center_force.shape[0] == 3)
    orientation_force = jnp.array(orientation_force)
    assert(len(orientation_force.shape) == 1)
    assert(orientation_force.shape[0] == 4)
    zero_center_force = jnp.array([0., 0., 0.])
    zero_orientation_force = jnp.array([0., 0., 0., 0.])

    base_force_fn = quantity.canonicalize_force(base_energy_or_force_fn)

    nuc_ids = set(nuc_ids)
    center_ext_force = jnp.array([center_force if idx in nuc_ids else zero_center_force
                                  for idx in range(n)])
    orientation_ext_force = jnp.array([orientation_force if idx in nuc_ids else zero_orientation_force
                                       for idx in range(n)])

    @jit
    def force_fn(body, **kwargs):
        base_force = base_force_fn(body, **kwargs)
        center = base_force.center + center_ext_force
        orientation_vec = base_force.orientation.vec + orientation_ext_force
        orientation = rigid_body.Quaternion(orientation_vec)
        return rigid_body.RigidBody(center=center, orientation=orientation)

    return base_force_fn, force_fn


if __name__ == "__main__":
    from utils import get_one_hot
    from loader.trajectory import TrajectoryInfo
    from loader.topology import TopologyInfo
    from jax_md import space


    top_path = "data/simple-helix/generated.top"
    top_info = TopologyInfo(top_path, reverse_direction=True)
    displacement_fn, shift_fn = space.free()

    base_energy_fn, _ = factory.energy_fn_factory(displacement_fn,
                                                  back_site, stack_site, base_site,
                                                  top_info.bonded_nbrs, top_info.unbonded_nbrs)

    base_force_fn, force_fn = get_force_fn(base_energy_fn, top_info.n, displacement_fn,
                                           nuc_ids=[1, 2],
                                           center_force=[0, 0, 1.0],
                                           orientation_force=[0, 0, 0.0, 1.0])
    recanon_force_fn = quantity.canonicalize_force(force_fn)

    conf_path = "data/simple-helix/start.conf"
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    body = config_info.states[0]
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)


    init_fene_params = [2.0, 0.25, 0.7525]
    init_stacking_params = [
        1.3448, 2.6568, 6.0, 0.4, 0.9, 0.32, 0.75, # f1(dr_stack)
        1.30, 0.0, 0.8, # f4(theta_4)
        0.90, 0.0, 0.95, # f4(theta_5p)
        0.90, 0.0, 0.95, # f4(theta_6p)
        2.0, -0.65, # f5(-cos(phi1))
        2.0, -0.65 # f5(-cos(phi2))
    ]
    params = init_fene_params + init_stacking_params

    # energy_fn = jit(Partial(base_energy_fn, seq=seq, params=params))
    # force_fn = jit(Partial(force_fn, seq=seq, params=params))
    # energy_fn = Partial(base_energy_fn, seq=seq, params=params)
    force_fn = Partial(force_fn, seq=seq, params=params)
    base_force_fn = Partial(base_force_fn, seq=seq, params=params)
    recanon_force_fn = Partial(recanon_force_fn, seq=seq, params=params)

    # e_val = energy_fn(body)
    f_val = force_fn(body)
    base_f_val = base_force_fn(body)
    recanon_f_val = recanon_force_fn(body)

    pdb.set_trace()
