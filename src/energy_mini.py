import numpy as onp
import pdb
import toml
from functools import partial

from jax import jit
from jax import vmap
from jax import random
from jax import lax
from jax.tree_util import Partial

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

from utils import back_site, stack_site, base_site
from utils import nucleotide_mass, get_kt, moment_of_inertia
from utils import Q_to_back_base, Q_to_cross_prod, Q_to_base_normal
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import clamp
from trajectory import TrajectoryInfo
from topology import TopologyInfo
# from potential import v_fene, exc_vol_bonded, stacking
from potential_hard import v_fene, v_fene2, exc_vol_bonded, stacking

from jax.config import config
config.update("jax_enable_x64", True)


def energy_fn_factory(displacement_fn,
                      back_site, stack_site, base_site,
                      bonded_neighbors, unbonded_neighbors):

    d = space.map_bond(partial(displacement_fn))
    nn_i = bonded_neighbors[:, 0]
    nn_j = bonded_neighbors[:, 1]

    def energy_fn(body: RigidBody, seq: util.Array, params, **kwargs) -> float:

        params_v_fene = Partial(v_fene2, eps_backbone=params[0], delta_backbone=params[1],
                                r0_backbone=params[2])


        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q)
        back_sites = body.center + com_to_backbone * back_base_vectors
        stack_sites = body.center + com_to_stacking * back_base_vectors
        base_sites = body.center + com_to_hb * back_base_vectors

        # back_sites = body.center + rigid_body.quaternion_rotate(Q, back_site) # (N, 3)
        # stack_sites = body.center + rigid_body.quaternion_rotate(Q, stack_site)
        # base_sites = body.center + rigid_body.quaternion_rotate(Q, base_site)

        dr_back = d(back_sites[nn_i], back_sites[nn_j]) # N x N x 3
        r_back = jnp.linalg.norm(dr_back, axis=1)
        fene = params_v_fene(r_back)
        return jnp.sum(fene)

        # return params["fene"]["eps_backbone"]
        # return jnp.sum(params) * jnp.sum(r_back)

    return energy_fn


if __name__ == "__main__":
    pass
