import pdb
import toml
from pathlib import Path

import jax.numpy as jnp

from jax_dna.common.smoothing import get_f4_smoothing_params
from jax_dna.common import utils
from jax_dna.dna1 import load_params as load_params_dna1


def _process(params, t_kelvin, salt_conc):
    geometry_params = params["geometry"]
    proc_params = {
        "fene": params["fene"],
        "excluded_volume": params["excluded_volume"],
        "stacking": params["stacking"],
        "hydrogen_bonding": params["hydrogen_bonding"],
        "cross_stacking": params["cross_stacking"],
        "coaxial_stacking": params["coaxial_stacking"]
    }

    # Add dummy parameters...

    ## ... for stacking f4(theta_4) term
    proc_params["stacking"]["a_stack_4"] = 0.90
    proc_params["stacking"]["theta0_stack_4"] = 0.0
    proc_params["stacking"]["delta_theta_star_stack_4"] = 0.95

    ## ... for cross_stacking f4(theta_4) + f4(pi - theta_4) term
    proc_params["cross_stacking"]["a_cross_4"] = 1.50
    proc_params["cross_stacking"]["theta0_cross_4"] = 0.0
    proc_params["cross_stacking"]["delta_theta_star_cross_4"] = 0.65


    proc_params = load_params_dna1._process(proc_params, t_kelvin)

    # Remove stacking parameters for f4(theta_4)
    to_remove = set(["a_stack_4", "theta0_stack_4", "delta_theta_star_stack_4",
                     "b_stack_4", "delta_theta_stack_4_c"])
    proc_params = load_params_dna1.clean_stacking_keys(
        proc_params, keys_to_remove=to_remove)

    # Remove cross stacking parameters for f4(theta_4) + f4(pi - theta_4)
    to_remove = set(["a_cross_4", "theta0_cross_4", "delta_theta_star_cross_4",
                     "b_cross_4", "delta_theta_cross_4_c"])
    proc_params['cross_stacking'] = {
        k: proc_params['cross_stacking'][k] for k in params['cross_stacking'].keys() if k not in to_remove
    }

    # Add additional smoothing terms for stacking potential
    b_stack_9, delta_theta_stack_9_c = get_f4_smoothing_params(
        proc_params['stacking']['a_stack_9'],
        proc_params['stacking']['theta0_stack_9'],
        proc_params['stacking']['delta_theta_star_stack_9'])
    proc_params['stacking']['b_stack_9'] = b_stack_9
    proc_params['stacking']['delta_theta_stack_9_c'] = delta_theta_stack_9_c

    b_stack_10, delta_theta_stack_10_c = get_f4_smoothing_params(
        proc_params['stacking']['a_stack_10'],
        proc_params['stacking']['theta0_stack_10'],
        proc_params['stacking']['delta_theta_star_stack_10'])
    proc_params['stacking']['b_stack_10'] = b_stack_10
    proc_params['stacking']['delta_theta_stack_10_c'] = delta_theta_stack_10_c

    # Process the debye-huckel parameters
    kT = utils.get_kt(t_kelvin)

    db_lambda_factor = 0.3616455075438555
    db_lambda = db_lambda_factor * jnp.sqrt(kT / 0.1) / jnp.sqrt(salt_conc)
    proc_params['debye'] = dict()
    proc_params['debye']['minus_kappa'] = -1.0 / db_lambda
    proc_params['debye']['r_high'] = 3.0 * db_lambda

    db_prefactor =  0.08173808693529228*(params['debye']['q_eff']**2)
    proc_params['debye']['prefactor'] = db_prefactor

    x = proc_params['debye']['r_high']
    q = db_prefactor
    l = db_lambda
    db_B = -(jnp.exp(-x / l) * q * q * (x + l) * (x + l)) \
           / (-4. * x * x * x * l * l * q)
    proc_params['debye']['B'] = db_B
    db_rcut = x * (q * x + 3. * q * l) / (q * (x + l))
    proc_params['debye']['rcut'] = db_rcut

    proc_params["geometry"] = geometry_params

    return proc_params

def load(seq_avg, params_path="data/thermo-params/rna2.toml",
         t_kelvin=utils.DEFAULT_TEMP, salt_conc=0.5, q_eff=0.815,
         process=True):

    if not seq_avg:
        raise NotImplementedError(f"Sequence dependence not implemented for RNA model")

    params = toml.load(params_path)
    params['debye'] = dict()
    params['debye']['q_eff'] = q_eff
    if process:
        params = _process(params, t_kelvin, salt_conc)
    return params

if __name__ == "__main__":
    params = load(seq_avg=True)

    pdb.set_trace()
    print("done")
