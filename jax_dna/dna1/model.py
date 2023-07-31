import pdb
import unittest
from functools import partial
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

from jax import jit, random
import jax.numpy as jnp
from jax_md import space, simulate, rigid_body

from jax_dna.common.utils import DEFAULT_TEMP, clamp
from jax_dna.common.utils import Q_to_back_base, Q_to_base_normal, Q_to_cross_prod
from jax_dna.common.interactions import v_fene_smooth
from jax_dna.common import utils, topology, trajectory
from jax_dna.dna1.load_params import load, process


DEFAULT_BASE_PARAMS = load(process=False) # Note: only processing depends on temperature
EMPTY_BASE_PARAMS = {
    "fene": dict(),
    "excluded_volume": dict(),
    "stacking": dict(),
    "hydrogen_bonding": dict(),
    "cross_stacking": dict(),
    "coaxial_stacking": dict()
}
com_to_stacking = 0.34
com_to_hb = 0.4
com_to_backbone = -0.4


class EnergyModel:
    def __init__(self, displacement_fn, override_base_params=EMPTY_BASE_PARAMS, t_kelvin=DEFAULT_TEMP):
        self.displacement_fn = displacement_fn
        self.displacement_mapped = jit(space.map_bond(partial(displacement_fn)))
        self.t_kelvin = t_kelvin

        fene_params = DEFAULT_BASE_PARAMS["fene"] | override_base_params["fene"]
        exc_vol_params = DEFAULT_BASE_PARAMS["excluded_volume"] | override_base_params["excluded_volume"]
        stacking_params = DEFAULT_BASE_PARAMS["stacking"] | override_base_params["stacking"]
        hb_params = DEFAULT_BASE_PARAMS["hydrogen_bonding"] | override_base_params["hydrogen_bonding"]
        cr_params = DEFAULT_BASE_PARAMS["cross_stacking"] | override_base_params["cross_stacking"]
        cx_params = DEFAULT_BASE_PARAMS["coaxial_stacking"] | override_base_params["coaxial_stacking"]

        self.base_params = {
            "fene": fene_params,
            "excluded_volume": exc_vol_params,
            "stacking": stacking_params,
            "hydrogen_bonding": hb_params,
            "cross_stacking": cr_params,
            "coaxial_stacking": cx_params
        }
        self.params = process(self.base_params, self.t_kelvin)

    def compute_subterms(self, body, seq, bonded_nbrs, unbonded_nbrs):
        nn_i = bonded_nbrs[:, 0]
        nn_j = bonded_nbrs[:, 1]

        op_i = unbonded_nbrs[0]
        op_j = unbonded_nbrs[1]
        mask = jnp.array(op_i < body.center.shape[0], dtype=jnp.int32)

        # Compute relevant variables for our potential
        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q) # space frame, normalized
        base_normals = Q_to_base_normal(Q) # space frame, normalized
        cross_prods = Q_to_cross_prod(Q) # space frame, normalized

        back_sites = body.center + com_to_backbone * back_base_vectors
        stack_sites = body.center + com_to_stacking * back_base_vectors
        base_sites = body.center + com_to_hb * back_base_vectors

        ## Fene variables
        dr_back_nn = self.displacement_mapped(back_sites[nn_i], back_sites[nn_j]) # N x N x 3
        r_back_nn = jnp.linalg.norm(dr_back_nn, axis=1)

        ## Exc. vol bonded variables
        dr_base_nn = self.displacement_mapped(base_sites[nn_i], base_sites[nn_j])
        dr_back_base_nn = self.displacement_mapped(back_sites[nn_i], base_sites[nn_j])
        dr_base_back_nn = self.displacement_mapped(base_sites[nn_i], back_sites[nn_j])

        ## Stacking variables
        dr_stack_nn = self.displacement_mapped(stack_sites[nn_i], stack_sites[nn_j])
        r_stack_nn = jnp.linalg.norm(dr_stack_nn, axis=1)

        theta4 = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nn_i], base_normals[nn_j])))
        theta5 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', dr_stack_nn, base_normals[nn_j]) / r_stack_nn))
        theta6 = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[nn_i], dr_stack_nn) / r_stack_nn))
        cosphi1 = -jnp.einsum('ij, ij->i', cross_prods[nn_i], dr_back_nn) / r_back_nn
        cosphi2 = -jnp.einsum('ij, ij->i', cross_prods[nn_j], dr_back_nn) / r_back_nn

        ## Exc. vol unbonded variables -- FIXME

        ## Hydrogen bonding -- FIXME

        ## Cross stacking variables -- FIXME

        ## Coaxial stacking -- FIXME

        # Compute the contributions from each interaction
        fene_dg = v_fene_smooth(r_back_nn, **self.params["fene"]).sum()

        return fene_dg, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def energy_fn(self, body, seq, bonded_nbrs, unbonded_nbrs):
        dgs = self.compute_subterms(body, seq, bonded_nbrs, unbonded_nbrs)
        fene_dg, b_exc_dg, stack_dg, n_exc_dg, hb_dg, cr_stack, cx_stack = dgs
        return fene_dg + b_exc_dg + stack_dg + n_exc_dg + hb_dg + cr_stack + cx_stack


class TestDna1(unittest.TestCase):
    test_data_basedir = Path("data/test-data")

    def test_init(self):
        displacement_fn, shift_fn = space.free()
        model = EnergyModel(displacement_fn)

    def test_simulate(self):
        displacement_fn, shift_fn = space.free()
        dt = 5e-3
        t_kelvin = DEFAULT_TEMP
        kT = utils.get_kt(t_kelvin)
        gamma = rigid_body.RigidBody(center=jnp.array([kT/2.5], dtype=jnp.float64),
                                     orientation=jnp.array([kT/7.5], dtype=jnp.float64))
        mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                    orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))

        top_path = self.test_data_basedir / "simple-helix" / "generated.top"
        top_info = topology.TopologyInfo(top_path, reverse_direction=True)
        seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

        conf_path = self.test_data_basedir / "simple-helix" / "start.conf"

        conf_info = trajectory.TrajectoryInfo(
            top_info,
            read_from_file=True, traj_path=conf_path, reverse_direction=True
        )

        init_body = conf_info.get_states()[0]

        n_steps = 10
        key = random.PRNGKey(0)


        def sim_fn(param_dict):
            model = EnergyModel(displacement_fn, param_dict)

            energy_fn = partial(model.energy_fn, seq=seq_oh,
                                bonded_nbrs=top_info.bonded_nbrs,
                                unbonded_nbrs=top_info.unbonded_nbrs.T)
            init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)

            state = init_fn(key, init_body, mass=mass)

            for i in tqdm(range(n_steps)):
                state = step_fn(state)
            return state.position.center.sum() # note: dummy loss function

        test_param_dict = deepcopy(EMPTY_BASE_PARAMS)
        test_param_dict["fene"]["eps_backbone"] = 1.5

        pos_sum = sim_fn(test_param_dict)
        pdb.set_trace()





if __name__ == "__main__":
    unittest.main()
