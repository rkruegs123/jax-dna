import pdb
import unittest
from functools import partial
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import numpy as onp

from jax import jit, random, lax, grad, value_and_grad
import jax.numpy as jnp
from jax_md import space, simulate, rigid_body

from jax_dna.common.utils import DEFAULT_TEMP, clamp
from jax_dna.common.utils import Q_to_back_base, Q_to_base_normal, Q_to_cross_prod
from jax_dna.common.interactions import v_fene_smooth, stacking, exc_vol_bonded, \
    exc_vol_unbonded, cross_stacking, coaxial_stacking, hydrogen_bonding
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

        ## Exc. vol unbonded variables
        dr_base_op = self.displacement_mapped(base_sites[op_j], base_sites[op_i]) # Note the flip here
        dr_backbone_op = self.displacement_mapped(back_sites[op_j], back_sites[op_i]) # Note the flip here
        dr_back_base_op = self.displacement_mapped(back_sites[op_i], base_sites[op_j]) # Note: didn't flip this one (and others) because no need, but should look into at some point
        dr_base_back_op = self.displacement_mapped(base_sites[op_i], back_sites[op_j])

        ## Hydrogen bonding
        r_base_op = jnp.linalg.norm(dr_base_op, axis=1)
        theta1_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -back_base_vectors[op_i], back_base_vectors[op_j])))
        theta2_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -back_base_vectors[op_j], dr_base_op) / r_base_op))
        theta3_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', back_base_vectors[op_i], dr_base_op) / r_base_op))
        theta4_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[op_i], base_normals[op_j])))
        # note: are these swapped in Lorenzo's code?
        theta7_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -base_normals[op_j], dr_base_op) / r_base_op))
        theta8_op = jnp.pi - jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[op_i], dr_base_op) / r_base_op))

        ## Cross stacking variables -- all already computed

        ## Coaxial stacking
        dr_stack_op = self.displacement_mapped(stack_sites[op_j], stack_sites[op_i]) # note: reversed
        dr_stack_norm_op = dr_stack_op / jnp.linalg.norm(dr_stack_op, axis=1, keepdims=True)
        dr_backbone_norm_op = dr_backbone_op / jnp.linalg.norm(dr_backbone_op, axis=1, keepdims=True)
        theta5_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', base_normals[op_i], dr_stack_norm_op)))
        theta6_op = jnp.arccos(clamp(jnp.einsum('ij, ij->i', -base_normals[op_j], dr_stack_norm_op)))
        cosphi3_op = jnp.einsum('ij, ij->i', dr_stack_norm_op,
                                jnp.cross(dr_backbone_norm_op, back_base_vectors[op_j]))
        cosphi4_op = jnp.einsum('ij, ij->i', dr_stack_norm_op,
                                jnp.cross(dr_backbone_norm_op, back_base_vectors[op_i]))

        # Compute the contributions from each interaction
        fene_dg = v_fene_smooth(r_back_nn, **self.params["fene"]).sum()
        exc_vol_bonded_dg = exc_vol_bonded(dr_base_nn, dr_back_base_nn, dr_base_back_nn,
                                           **self.params["excluded_volume_bonded"]).sum()
        stack_dg = stacking(r_stack_nn, theta4, theta5, theta6, cosphi1, cosphi2, **self.params["stacking"]).sum()

        exc_vol_unbonded_dg = exc_vol_unbonded(
            dr_base_op, dr_backbone_op, dr_back_base_op, dr_base_back_op,
            **self.params["excluded_volume"]
        )
        exc_vol_unbonded_dg = jnp.where(mask, exc_vol_unbonded_dg, 0.0).sum() # Mask for neighbors

        v_hb = hydrogen_bonding(
            dr_base_op, theta1_op, theta2_op, theta3_op, theta4_op,
            theta7_op, theta8_op, **self.params["hydrogen_bonding"])
        v_hb = jnp.where(mask, v_hb, 0.0) # Mask for neighbors
        hb_probs = utils.get_hb_probs(seq, op_i, op_j) # get the probabilities of all possibile hydrogen bonds for all neighbors
        hb_weights = jnp.dot(hb_probs, utils.HB_WEIGHTS)
        hb_dg = jnp.dot(hb_weights, v_hb)

        cr_stack_dg = cross_stacking(
            r_base_op, theta1_op, theta2_op, theta3_op,
            theta4_op, theta7_op, theta8_op, **self.params["cross_stacking"])
        cr_stack_dg = jnp.where(mask, cr_stack_dg, 0.0).sum() # Mask for neighbors

        cx_stack_dg = coaxial_stacking(
            dr_stack_op, theta4_op, theta1_op, theta5_op,
            theta6_op, cosphi3_op, cosphi4_op, **self.params["coaxial_stacking"])
        cx_stack_dg = jnp.where(mask, cx_stack_dg, 0.0).sum() # Mask for neighbors

        return fene_dg, exc_vol_bonded_dg, stack_dg, \
            exc_vol_unbonded_dg, hb_dg, cr_stack_dg, cx_stack_dg

    def energy_fn(self, body, seq, bonded_nbrs, unbonded_nbrs):
        dgs = self.compute_subterms(body, seq, bonded_nbrs, unbonded_nbrs)
        fene_dg, b_exc_dg, stack_dg, n_exc_dg, hb_dg, cr_stack, cx_stack = dgs
        return fene_dg + b_exc_dg + stack_dg + n_exc_dg + hb_dg + cr_stack + cx_stack


class TestDna1(unittest.TestCase):
    test_data_basedir = Path("data/test-data")

    def test_init(self):
        displacement_fn, shift_fn = space.free()
        model = EnergyModel(displacement_fn)

    def test_grad_dummy(self):

        displacement_fn, shift_fn = space.free()

        def loss_fn(param_dict):
            model = EnergyModel(displacement_fn, param_dict)
            return model.params["fene"]["eps_backbone"]

        test_param_dict = deepcopy(EMPTY_BASE_PARAMS)
        test_param_dict["fene"]["eps_backbone"] = 1.5

        test_grad = grad(loss_fn)(test_param_dict)

        self.assertNotEqual(test_grad, 0.0)

    @unittest.skip("Disabled by default because compilation isn slow on CPU")
    def test_simulate_grad(self, write_traj=False):
        # Setup the system
        box_size = 20.0
        displacement_fn, shift_fn = space.periodic(box_size)
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

        n_steps = 1000
        key = random.PRNGKey(0)

        def sim_fn(param_dict):
            model = EnergyModel(displacement_fn, param_dict)

            energy_fn = partial(model.energy_fn, seq=seq_oh,
                                bonded_nbrs=top_info.bonded_nbrs,
                                unbonded_nbrs=top_info.unbonded_nbrs.T)

            init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
            step_fn = jit(step_fn)
            init_state = init_fn(key, init_body, mass=mass)

            @jit
            def scan_fn(state, step):
                state = step_fn(state)
                return state, state.position

            fin_state, traj = lax.scan(scan_fn, init_state, jnp.arange(n_steps))
            # return fin_state.position.center.sum(), traj # note: grad w.r.t. this loss is 0.0
            return fin_state.position.center[-1][0], traj # note: dummy loss function

        # Define a dummy set of parameters to override
        test_param_dict = deepcopy(EMPTY_BASE_PARAMS)
        test_param_dict["fene"]["eps_backbone"] = 2.5
        test_param_dict["stacking"]["eps_stack_base"] = 0.5
        test_param_dict["excluded_volume"]["eps_exc"] = 5.0

        # Run a simulation to confirm that this doesn't yield any errors
        pos_sum, traj = sim_fn(test_param_dict)

        if write_traj:
            traj_to_write = traj[::100]
            traj_info = trajectory.TrajectoryInfo(
                top_info, read_from_states=True, states=traj_to_write, box_size=box_size)
            traj_info.write("dna1_sanity.conf", reverse=True)

        (pos_sum, traj), pos_sum_grad = jit(value_and_grad(sim_fn, has_aux=True))(test_param_dict)

        self.assertNotEqual(pos_sum_grad, 0.0)

    def check_energy_subterms(self, basedir, top_fname, traj_fname, t_kelvin):

        print(f"---- Checking energy breakdown agreement for base directory: {basedir} ----")

        basedir = Path(basedir)
        if not basedir.exists():
            raise RuntimeError(f"No directory exists at location: {basedir}")

        # First, load the oxDNA subterms
        split_energy_fname = basedir / "split_energy.dat"
        if not split_energy_fname.exists():
            raise RuntimeError(f"No energy subterm file exists at location: {split_energy_fname}")
        split_energy_df = pd.read_csv(
            split_energy_fname,
            names=["t", "fene", "b_exc", "stack", "n_exc", "hb",
               "cr_stack", "cx_stack"],
            delim_whitespace=True)
        oxdna_subterms = split_energy_df.iloc[1:, :]

        # Then, compute subterms via our energy model
        top_path = basedir / top_fname
        if not top_path.exists():
            raise RuntimeError(f"No topology file at location: {top_path}")
        traj_path = basedir / traj_fname
        if not traj_path.exists():
            raise RuntimeError(f"No trajectory file at location: {traj_path}")

        ## note: we don't reverse direction to keep ordering the same
        top_info = topology.TopologyInfo(top_path, reverse_direction=False)
        seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True, traj_path=traj_path, reverse_direction=False)
        traj_states = traj_info.get_states()

        displacement_fn, shift_fn = space.periodic(traj_info.box_size)
        model = EnergyModel(displacement_fn, t_kelvin=t_kelvin)

        trajectory_subterms = list()
        for state in tqdm(traj_states):
            dgs = model.compute_subterms(
                state, seq_oh, top_info.bonded_nbrs, top_info.unbonded_nbrs.T)
            avg_subterms = onp.array(dgs) / top_info.n # average per nucleotide
            trajectory_subterms.append(avg_subterms)
        trajectory_subterms = onp.array(trajectory_subterms)

        # Check for equality

        # FIXME: put oxdna_subterms as a numpy array?
        pdb.set_trace()

    def test_subterms(self):
        subterm_tests = [
            (self.test_data_basedir / "simple-helix", "generated.top", "output.dat", 296.15)
        ]

        for basedir, top_fname, traj_fname, t_kelvin in subterm_tests:
            self.check_energy_subterms(basedir, top_fname, traj_fname, t_kelvin)


if __name__ == "__main__":
    unittest.main()
