import pdb
import numpy as onp
from pathlib import Path
from copy import deepcopy

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax_md import space

from jax_dna.common import utils, topology, trajectory
from jax_dna.rna2 import model, load_params


def run():
    basedir = Path("data") / "templates" / "5ht-tc-rmse-rna"
    assert(basedir.exists())
    top_path = basedir / "sys.top"
    # top_path = "/home/ryan/Downloads/output.top"
    # conf_path = basedir / "init.conf"
    conf_path = basedir / "target.conf"
    # conf_path = "/home/ryan/Downloads/output.dat"

    top_info = topology.TopologyInfo(top_path, reverse_direction=False, is_rna=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq, is_rna=True), dtype=jnp.float64)
    conf_info = trajectory.TrajectoryInfo(
        top_info, read_from_file=True, traj_path=conf_path, reverse_direction=False)
    body = conf_info.get_states()[0]

    ss_hb_weights, ss_stack_weights, ss_cross_weights = load_params.read_seq_specific(load_params.DEFAULT_BASE_PARAMS)

    displacement_fn, shift_fn = space.free()

    salt_conc = 1.0
    t_kelvin = utils.DEFAULT_TEMP

    params = deepcopy(load_params.EMPTY_BASE_PARAMS)
    em = model.EnergyModel(
        displacement_fn, params, t_kelvin=t_kelvin, salt_conc=salt_conc,
        ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights)

    em_symm = model.EnergyModel(
        displacement_fn, params, t_kelvin=t_kelvin, salt_conc=salt_conc,
        ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights, use_symm_coax=True)

    order = ["fene", "exc_vol_bonded", "stack", "exc_vol_unbonded", "hb", "cr_stack", "cx_stack", "db"]
    def print_subterms(subterms):
        for term, term_name in zip(subterms, order):
            print(f"- {term_name}: {term}")

    # First, compute the full energy
    """
    bonded_nbrs = top_info.bonded_nbrs
    unbonded_nbrs = top_info.unbonded_nbrs.T
    dg = em.energy_fn(body, seq_oh, bonded_nbrs, unbonded_nbrs)
    dg_symm = em_symm.energy_fn(body, seq_oh, bonded_nbrs, unbonded_nbrs)

    dg_subterms = em.compute_subterms(body, seq_oh, bonded_nbrs, unbonded_nbrs)

    print(f"Initial")

    print("\nWithout symm:")
    print_subterms(dg_subterms)
    """

    # Check hydrogen bonding
    """
    bonded_nbrs = jnp.array([[0, 1]])
    unbonded_nbrs = jnp.array([[134, 146]]).T # hydrogen bonded

    print(f"\nChecking HB")

    subterms = em.compute_subterms(body, seq_oh, bonded_nbrs, unbonded_nbrs)
    print("\nWithout symm:")
    print_subterms(subterms)

    subterms_symm = em_symm.compute_subterms(body, seq_oh, bonded_nbrs, unbonded_nbrs)
    print("\nWith symm:")
    print_subterms(subterms_symm)
    """

    # Check coaxial stacking pairs
    print(f"\nChecking coax.")

    bonded_nbrs = jnp.array([[0, 1]])
    unbonded_nbrs = jnp.array([[137, 408], [172, 375], [241, 308], [200, 335]]).T
    unbonded_nbrs = body.center.shape[0] - unbonded_nbrs - 1 # Checked this conversion by checking distances in oxview


    subterms = em.compute_subterms(body, seq_oh, bonded_nbrs, unbonded_nbrs)
    print("\nWithout symm:")
    print_subterms(subterms)

    subterms_symm = em_symm.compute_subterms(body, seq_oh, bonded_nbrs, unbonded_nbrs)
    print("\nWith symm:")
    print_subterms(subterms_symm)

    pdb.set_trace()

if __name__ == "__main__":
    run()
