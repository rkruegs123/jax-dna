import numpy as onp
import pdb
import toml
from functools import partial

from jax import jit
from jax import vmap
from jax import random
from jax import lax

from jax.config import config as jax_config
import jax.numpy as jnp

from jax_md import quantity
from jax_md import simulate
from jax_md import space
from jax_md import smap
from jax_md import energy
from jax_md import test_util
from jax_md import partition
from jax_md import util
from jax_md import rigid_body
from jax_md.rigid_body import RigidBody, Quaternion

from potential import exc_vol_unbonded
from utils import read_config, jax_traj_to_oxdna_traj
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import nucleotide_mass, get_kt


FLAGS = jax_config.FLAGS
DYNAMICS_STEPS = 100


f32 = util.f32
f64 = util.f64

DTYPE = [f32]
if FLAGS.jax_enable_x64:
    DTYPE += [f64]

@partial(vmap, in_axes=(0, None))
def rand_quat(key, dtype):
    return rigid_body.random_quaternion(key, dtype)



def dynamic_energy_fn_factory_fixed(displacement_fn, back_site, stack_site, base_site, neighbors):

    d = space.map_bond(partial(displacement_fn))
    nbs_i = neighbors[:, 0]
    nbs_j = neighbors[:, 1]

    def energy_fn(body: RigidBody, **kwargs) -> float:
        Q = body.orientation

        back_sites = body.center + rigid_body.quaternion_rotate(Q, back_site) # (N, 3)
        stack_sites = body.center + rigid_body.quaternion_rotate(Q, stack_site)
        base_sites = body.center + rigid_body.quaternion_rotate(Q, base_site)

        # Excluded volume (unbonded)
        dr_base = d(base_sites[nbs_i], base_sites[nbs_j])
        dr_backbone = d(back_sites[nbs_i], back_sites[nbs_j])
        dr_back_base = d(back_sites[nbs_i], base_sites[nbs_j])
        dr_base_back = d(base_sites[nbs_i], back_sites[nbs_j])
        exc_vol = exc_vol_unbonded(dr_base, dr_backbone, dr_back_base, dr_base_back)

        # FIXME: add the others

        return jnp.sum(exc_vol)

    return energy_fn





# For variable neighbors
def dynamic_energy_fn_factory(displacement_fn, back_site, stack_site, base_site):

    d = space.map_bond(partial(displacement_fn))

    def energy_fn(body: RigidBody, neighbor: partition.NeighborList) -> float:
        Q = body.orientation
        back_sites = body.center + rigid_body.quaternion_rotate(Q, back_site) # (N, 3)
        stack_sites = body.center + rigid_body.quaternion_rotate(Q, stack_site)
        base_sites = body.center + rigid_body.quaternion_rotate(Q, base_site)

        pdb.set_trace()
        nbs_i = neighbor[:, 0]
        nbs_j = neighbor[:, 1]

        # FIXME: completey wrong potential
        # FENE
        dr_back = d(back_sites[nbs_i], back_sites[nbs_j]) # N x N x 3
        r_back = jnp.linalg.norm(dr_back, axis=1)
        fene = v_fene(r_back)

        return jnp.sum(fene)

    return energy_fn


if __name__ == "__main__":
    pass
