import pdb
from pathlib import Path
import numpy as onp
import toml

import jax.numpy as jnp

from jax_dna.common import utils
from jax_dna.dna1 import load_params as load_params_dna1


def _process(params, t_kelvin, salt_conc):

    proc_params = {
        "fene": params["fene"],
        "excluded_volume": params["excluded_volume"],
        "stacking": params["stacking"],
        "hydrogen_bonding": params["hydrogen_bonding"],
        "cross_stacking": params["cross_stacking"],
        "coaxial_stacking": params["coaxial_stacking"]
    }

    proc_params = load_params_dna1._process(proc_params, t_kelvin)

    # Remove unwanted coaxial stacking terms
    del proc_params['coaxial_stacking']['a_coax_3p']
    del proc_params['coaxial_stacking']['cos_phi3_star_coax']
    del proc_params['coaxial_stacking']['b_cos_phi3_coax']
    del proc_params['coaxial_stacking']['cos_phi3_c_coax']
    del proc_params['coaxial_stacking']['a_coax_4p']
    del proc_params['coaxial_stacking']['cos_phi4_star_coax']
    del proc_params['coaxial_stacking']['b_cos_phi4_coax']
    del proc_params['coaxial_stacking']['cos_phi4_c_coax']

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

    return proc_params

def load(seq_avg, dna1_params_path="data/thermo-params/dna1.toml",
         t_kelvin=utils.DEFAULT_TEMP, salt_conc=0.5, q_eff=0.815,
         process=True):

    # Load the DNA1 base params
    if not Path(dna1_params_path).exists():
        raise RuntimeError(f"No file at location: {dna1_params_path}")
    params = toml.load(dna1_params_path)

    # Override various values that change in DNA2
    params['fene']['r0_backbone'] = 0.7564
    params['coaxial_stacking']['k_coax'] = 58.5
    params['coaxial_stacking']['theta0_coax_1'] = onp.pi - 0.25
    params['coaxial_stacking']['A_coax_1_f6'] = 40.0
    params['coaxial_stacking']['B_coax_1_f6'] = onp.pi - 0.025

    if seq_avg:
        params['stacking']['eps_stack_base'] = 1.3523
        params['stacking']['eps_stack_kt_coeff'] = 2.6717
        params['hydrogen_bonding']['eps_hb'] = 1.0678

    # Add additional DNA2 parameters for coaxial stacking
    params['coaxial_stacking']['A_coax_1_f6'] = 40.0
    params['coaxial_stacking']['B_coax_1_f6'] = onp.pi - 0.025

    # Add DNA2 parameters for Debye-Huckel interaction
    params['debye'] = dict()
    params['debye']['q_eff'] = q_eff

    if process:
        params = _process(params, t_kelvin, salt_conc)
    return params


if __name__ == "__main__":
    pass
