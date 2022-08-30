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
from utils import clamp
from trajectory import TrajectoryInfo
from topology import TopologyInfo
from potential import v_fene, exc_vol_bonded, stacking

from jax.config import config
config.update("jax_enable_x64", True)

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


def nn_energy_fn_factory(displacement_fn,
                         back_site, stack_site, base_site,
                         neighbors):

    d = space.map_bond(partial(displacement_fn))
    nbs_i = neighbors[:, 0]
    nbs_j = neighbors[:, 1]

    def _compute_subterms(body: RigidBody, seq: util.Array, params):


        params_v_fene = Partial(v_fene, params=params)
        params_exc_vol_bonded = Partial(exc_vol_bonded, params=params)
        params_stacking = Partial(stacking, params=params)

        Q = body.orientation
        """
        # Conversion in Tom's thesis, Appendix A
        back_base_vectors = Q_to_back_base(Q)
        back_sites = body.center + com_to_backbone * back_base_vectors
        stack_sites = body.center + com_to_stacking * back_base_vectors
        base_sites = body.center + com_to_hb * back_base_vectors
        """

        back_sites = body.center + rigid_body.quaternion_rotate(Q, back_site) # (N, 3)
        stack_sites = body.center + rigid_body.quaternion_rotate(Q, stack_site)
        base_sites = body.center + rigid_body.quaternion_rotate(Q, base_site)

        # FIXME: flatten, make
        # Note: I believe we don't have to flatten. In Sam's original code, R_sites contained *all* N*3 interaction sites

        # FIXME: for neighbors, this will change
        # d = space.map_product(partial(displacement_fn, **kwargs))

        # FENE
        dr_back = d(back_sites[nbs_i], back_sites[nbs_j]) # N x N x 3
        r_back = jnp.linalg.norm(dr_back, axis=1)
        fene = params_v_fene(r_back)

        # Excluded volume (bonded)
        dr_base = d(base_sites[nbs_i], base_sites[nbs_j])
        dr_back_base = d(back_sites[nbs_i], base_sites[nbs_j])
        dr_base_back = d(base_sites[nbs_i], back_sites[nbs_j])
        exc_vol = params_exc_vol_bonded(dr_base, dr_back_base, dr_base_back)

        # Stacking
        dr_stack = d(stack_sites[nbs_i], stack_sites[nbs_j])
        base_normals = Q_to_base_normal(Q) # space frame, normalized
        cross_prods = Q_to_cross_prod(Q) # space frame, normalized
        theta4 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nbs_i], base_normals[nbs_j])))
        theta5 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', dr_stack, base_normals[nbs_j]) / jnp.linalg.norm(dr_stack, axis=1)))
        theta6 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nbs_i], dr_stack) / jnp.linalg.norm(dr_stack, axis=1)))
        cosphi1 = -jnp.einsum('ij, ij->i', cross_prods[nbs_i], dr_back) / jnp.linalg.norm(dr_back, axis=1)
        cosphi2 = -jnp.einsum('ij, ij->i', cross_prods[nbs_j], dr_back) / jnp.linalg.norm(dr_back, axis=1)
        stack = params_stacking(dr_stack, theta4, theta5, theta6, cosphi1, cosphi2)

        return jnp.sum(fene), jnp.sum(exc_vol), jnp.sum(stack)

    # Philosophy: use `functools.partial` to fix either `seq` or `potential_fns` depending on what is being optimized
    def energy_fn(body: RigidBody, seq: util.Array, params, **kwargs) -> float:
        fene, exc_vol, stack = _compute_subterms(body, seq, params)
        return fene + exc_vol + stack


    return energy_fn, _compute_subterms



if __name__ == "__main__":

    top_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/polyA_10bp/generated.top"
    config_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/polyA_10bp/equilibrated.dat"

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=config_path, reverse_direction=True)

    # FIXME: Petr changed to just 1.0 and 1.0
    mass = rigid_body.RigidBody(center=jnp.array([nucleotide_mass]), orientation=jnp.array([moment_of_inertia]))

    body = config_info.states[0]
    box_size = config_info.box_size

    displacement, shift = space.periodic(box_size)
    key = random.PRNGKey(0)
    key, pos_key, quat_key = random.split(key, 3)
    dtype = f64


    n = config_info.top_info.n
    bonded_neighbors = config_info.top_info.bonded_nbrs

    energy_fn = nn_energy_fn_factory(displacement,
                                     back_site=back_site,
                                     stack_site=stack_site,
                                     base_site=base_site,
                                     neighbors=bonded_neighbors)


    # Simulate with the energy function via Nose-Hoover

    # kT = 1e-3
    kT = get_kt(t=TEMP) # 300 Kelvin = 0.1 kT
    dt = 5e-3

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)

    step_fn = jit(step_fn)

    state = init_fn(key, body, mass=mass)
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    trajectory = [state.position]

    for i in range(DYNAMICS_STEPS):
        state = step_fn(state)
        trajectory.append(state.position)


    # E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)


    # FIXME: Add excluded volume and stacking
    pdb.set_trace()

    # FIXME: convert states to TrajInfo and write the new traj_info to file
    # jax_traj_to_oxdna_traj(trajectory, box_size, every_n=50)

    print("done")
