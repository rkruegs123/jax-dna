import pdb
from copy import deepcopy

import jax.numpy as jnp

from jax_dna.dna1 import oxdna_utils as dna1_utils
from jax_dna.dna2 import model
from jax_dna.common.utils import DEFAULT_TEMP


def recompile_oxdna(override_base_params, oxdna_path, t_kelvin, num_threads=4, seq_avg=True):
    variable_mapper = deepcopy(dna1_utils.DEFAULT_VARIABLE_MAPPER)
    if seq_avg:
        variable_mapper['stacking']["eps_stack_base"] = "STCK_BASE_EPS_OXDNA2"
        variable_mapper['stacking']["eps_stack_kt_coeff"] = "STCK_FACT_EPS_OXDNA2"
        variable_mapper["hydrogen_bonding"]["eps_hb"] = "HYDR_EPS_OXDNA2"

    return dna1_utils.recompile_oxdna(override_base_params, oxdna_path, t_kelvin, num_threads=num_threads, variable_mapper=variable_mapper)

if __name__ == "__main__":

    oxdna_path = Path("/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/")
    t_kelvin = DEFAULT_TEMP

    seq_avg = True

    override_base_params = deepcopy(model.EMPTY_BASE_PARAMS)
    if seq_avg:
        default_base_params = model.default_base_params_seq_avg
    else:
        default_base_params = model.default_base_params_seq_dep

    override_base_params["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]

    override_base_params['stacking']['a_stack_5'] = 0.3
    override_base_params['stacking']['a_stack_6'] = 0.2

    recompile_oxdna(override_base_params, oxdna_path, t_kelvin, num_threads=4)
