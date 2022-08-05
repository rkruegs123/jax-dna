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

from potential import v_fene, exc_vol_bonded, stacking, TEMP
from utils import read_config, jax_traj_to_oxdna_traj
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import nucleotide_mass, get_kt, moment_of_inertia
from utils import Q_to_back_base, Q_to_cross_prod, Q_to_base_normal


FLAGS = jax_config.FLAGS
DYNAMICS_STEPS = 3000


f32 = util.f32
f64 = util.f64

DTYPE = [f32]
if FLAGS.jax_enable_x64:
    DTYPE += [f64]

@partial(vmap, in_axes=(0, None))
def rand_quat(key, dtype):
    return rigid_body.random_quaternion(key, dtype)


def static_energy_fn_factory(displacement_fn, back_site, stack_site, base_site, neighbors):

    d = space.map_bond(partial(displacement_fn))
    nbs_i = neighbors[:, 0]
    nbs_j = neighbors[:, 1]

    def energy_fn(body: RigidBody, **kwargs) -> float:
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
        fene = v_fene(r_back)

        # Excluded volume (bonded)
        dr_base = d(base_sites[nbs_i], base_sites[nbs_j])
        dr_back_base = d(back_sites[nbs_i], base_sites[nbs_j])
        dr_base_back = d(base_sites[nbs_i], back_sites[nbs_j])
        exc_vol = exc_vol_bonded(dr_base, dr_back_base, dr_base_back)

        # Stacking
        dr_stack = d(stack_sites[nbs_i], stack_sites[nbs_j])
        base_normals = Q_to_base_normal(Q) # space frame, normalized
        cross_prods = Q_to_cross_prod(Q) # space frame, normalized
        theta4 = jnp.arccos(jnp.einsum('ij, ij->i', base_normals[nbs_i], base_normals[nbs_j])) # FIXME: understand `np.einsum`
        # FIXME: have to normalize the cosine here by the magnitude of dr_stack
        theta5 = jnp.pi - jnp.arccos(jnp.einsum('ij, ij->i', dr_stack, base_normals[nbs_j]) / jnp.linalg.norm(dr_stack, axis=1))
        theta6 = jnp.arccos(jnp.einsum('ij, ij->i', base_normals[nbs_i], dr_stack) / jnp.linalg.norm(dr_stack, axis=1))
        cosphi1 = jnp.einsum('ij, ij->i', cross_prods[nbs_i], dr_back) / jnp.linalg.norm(dr_back, axis=1) # FIXME: Ordering is probably wrong here. E.g. directionality of dr_back. Also, may or may not need a minus sign
        cosphi2 = jnp.einsum('ij, ij->i', cross_prods[nbs_j], dr_back) / jnp.linalg.norm(dr_back, axis=1) # FIXME: same as for cosphi1
        stack = stacking(dr_stack, theta4, theta5, theta6, cosphi1, cosphi2)


        # return jnp.sum(fene) + jnp.sum(exc_vol) + jnp.sum(stack)
        return jnp.sum(fene) + jnp.sum(exc_vol)

    return energy_fn


if __name__ == "__main__":


    # Bug in rigid body -- Nose-Hoover defaults to f32(1.0) rather than a RigidBody with this value

    """
    shape = rigid_body.point_union_shape(
      onp.array([[0.0, 0.0, 0.0]], f32),
      f32(nucleotide_mass)
    ) # just to get the mass from
    mass = shape.mass()
    """


    mass = rigid_body.RigidBody(center=jnp.array([nucleotide_mass]), orientation=jnp.array([moment_of_inertia]))

    body, box_size = read_config("data/polyA_10bp/equilibrated.dat")

    box_size = box_size[0]

    displacement, shift = space.periodic(box_size)
    key = random.PRNGKey(0)
    key, pos_key, quat_key = random.split(key, 3)
    dtype = DTYPE[0]

    N = body.center.shape[0]

    base_site = jnp.array(
        [com_to_hb, 0.0, 0.0]
    )
    stack_site = jnp.array(
        [com_to_stacking, 0.0, 0.0]
    )
    back_site = jnp.array(
        [com_to_backbone, 0.0, 0.0]
    )
    """
    bonded_neighbors = onp.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4]
    ])
    """

    n = 10 # FIXME: redundant. Use `N` from above
    bonded_neighbors = onp.array(
        [[i, i+1] for i in range(n - 1)]
    )

    energy_fn = static_energy_fn_factory(displacement,
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

    jax_traj_to_oxdna_traj(trajectory, box_size, every_n=50)

    print("done")
