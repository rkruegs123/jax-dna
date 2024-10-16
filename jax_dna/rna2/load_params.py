import pdb
import toml
from pathlib import Path
import numpy as onp
from copy import deepcopy

import jax.numpy as jnp

from jax_dna.common.smoothing import get_f4_smoothing_params
from jax_dna.common import utils
from jax_dna.common.utils import RNA_ALPHA
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

    # db_lambda_factor = 0.3616455075438555
    db_lambda_factor = 0.3667258
    db_lambda = db_lambda_factor * jnp.sqrt(kT / 0.1) / jnp.sqrt(salt_conc)
    proc_params['debye'] = dict()
    proc_params['debye']['minus_kappa'] = -1.0 / db_lambda
    proc_params['debye']['r_high'] = 3.0 * db_lambda

    db_prefactor =  0.05404383975812547*(params['debye']['q_eff']**2)
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

def load(params_path="data/thermo-params/rna2.toml",
         t_kelvin=utils.DEFAULT_TEMP, salt_conc=0.5, q_eff=1.26,
         process=True):

    params = toml.load(params_path)
    params['debye'] = dict()
    params['debye']['q_eff'] = q_eff
    if process:
        params = _process(params, t_kelvin, salt_conc)
    return params


DEFAULT_BASE_PARAMS = load(process=False)

EMPTY_BASE_PARAMS = {
    "geometry": dict(),
    "fene": dict(),
    "excluded_volume": dict(),
    "stacking": dict(),
    "hydrogen_bonding": dict(),
    "cross_stacking": dict(),
    "coaxial_stacking": dict(),
    "debye": dict()
}


def add_coupling(base_params):
    # Stacking
    base_params["stacking"]["a_stack_6"] = base_params["stacking"]["a_stack_5"]
    base_params["stacking"]["theta0_stack_6"] = base_params["stacking"]["theta0_stack_5"]
    base_params["stacking"]["delta_theta_star_stack_6"] = base_params["stacking"]["delta_theta_star_stack_5"]

    # Hydrogen Bonding
    base_params["hydrogen_bonding"]["a_hb_3"] = base_params["hydrogen_bonding"]["a_hb_2"]
    base_params["hydrogen_bonding"]["theta0_hb_3"] = base_params["hydrogen_bonding"]["theta0_hb_2"]
    base_params["hydrogen_bonding"]["delta_theta_star_hb_3"] = base_params["hydrogen_bonding"]["delta_theta_star_hb_2"]

    base_params["hydrogen_bonding"]["a_hb_8"] = base_params["hydrogen_bonding"]["a_hb_7"]
    base_params["hydrogen_bonding"]["theta0_hb_8"] = base_params["hydrogen_bonding"]["theta0_hb_7"]
    base_params["hydrogen_bonding"]["delta_theta_star_hb_8"] = base_params["hydrogen_bonding"]["delta_theta_star_hb_7"]

    # Coaxial Stacking
    base_params["coaxial_stacking"]["a_coax_6"] = base_params["coaxial_stacking"]["a_coax_5"]
    base_params["coaxial_stacking"]["theta0_coax_6"] = base_params["coaxial_stacking"]["theta0_coax_5"]
    base_params["coaxial_stacking"]["delta_theta_star_coax_6"] = base_params["coaxial_stacking"]["delta_theta_star_coax_5"]

    # Cross stacking
    base_params["cross_stacking"]["a_cross_3"] = base_params["cross_stacking"]["a_cross_2"]
    base_params["cross_stacking"]["theta0_cross_3"] = base_params["cross_stacking"]["theta0_cross_2"]
    base_params["cross_stacking"]["delta_theta_star_cross_3"] = base_params["cross_stacking"]["delta_theta_star_cross_2"]

    base_params["cross_stacking"]["a_cross_8"] = base_params["cross_stacking"]["a_cross_7"]
    base_params["cross_stacking"]["theta0_cross_8"] = base_params["cross_stacking"]["theta0_cross_7"]
    base_params["cross_stacking"]["delta_theta_star_cross_8"] = base_params["cross_stacking"]["delta_theta_star_cross_7"]



def get_full_base_params(override_base_params):

    default_base_params = deepcopy(DEFAULT_BASE_PARAMS)

    geometry_params = default_base_params["geometry"] | override_base_params["geometry"]
    fene_params = default_base_params["fene"] | override_base_params["fene"]
    exc_vol_params = default_base_params["excluded_volume"] | override_base_params["excluded_volume"]
    stacking_params = default_base_params["stacking"] | override_base_params["stacking"]
    hb_params = default_base_params["hydrogen_bonding"] | override_base_params["hydrogen_bonding"]
    cr_params = default_base_params["cross_stacking"] | override_base_params["cross_stacking"]
    cx_params = default_base_params["coaxial_stacking"] | override_base_params["coaxial_stacking"]
    debye_params = default_base_params["debye"] | override_base_params["debye"]

    base_params = {
        "geometry": geometry_params,
        "fene": fene_params,
        "excluded_volume": exc_vol_params,
        "stacking": stacking_params,
        "hydrogen_bonding": hb_params,
        "cross_stacking": cr_params,
        "coaxial_stacking": cx_params,
        "debye": debye_params
    }
    add_coupling(base_params)
    return base_params


DEFAULT_SS_PATH = "data/seq-specific/seq_rna.txt"
def read_seq_specific(base_params, ss_path=DEFAULT_SS_PATH,
                      enforce_symmetry=True):
    ss_path = Path(ss_path)
    assert(ss_path.exists())

    with open(ss_path, "r") as f:
        lines = f.readlines()

    vals = dict()
    for unstripped_line in lines:
        line = unstripped_line.strip()
        if not line:
            continue
        if lines[0] == "#":
            continue

        tokens = line.split()
        assert(len(tokens) == 3)
        assert(tokens[1] == "=")
        key = tokens[0]
        val = tokens[2]
        if val[-1] == "f":
            val = val[:-1]

        try:
            val = float(val)
        except:
            raise ValueError(f"Invalid float: {val}")

        vals[key] = val

    # Parse HB table
    hb_mult = onp.zeros((4, 4))
    default_f1_eps_hb = base_params["hydrogen_bonding"]["eps_hb"]

    hb_au_val = None
    if "HYDR_A_T" in vals:
        hb_au_val = vals["HYDR_A_T"]
    if "HYDR_T_A" in vals:
        if hb_au_val is not None and enforce_symmetry:
            assert(hb_au_val == vals["HYDR_T_A"])
        else:
            hb_au_val = vals["HYDR_T_A"]
    assert(hb_au_val is not None)
    hb_au_mult = hb_au_val / default_f1_eps_hb

    hb_gc_val = None
    if "HYDR_G_C" in vals:
        hb_gc_val = vals["HYDR_G_C"]
    if "HYDR_C_G" in vals:
        if hb_gc_val is not None and enforce_symmetry:
            assert(hb_gc_val == vals["HYDR_C_G"])
        else:
            hb_gc_val = vals["HYDR_C_G"]
    assert(hb_gc_val is not None)
    hb_gc_mult = hb_gc_val / default_f1_eps_hb

    hb_gu_val = None
    if "HYDR_G_T" in vals:
        hb_gu_val = vals["HYDR_G_T"]
    if "HYDR_T_G" in vals:
        if hb_gu_val is not None and enforce_symmetry:
            assert(hb_gu_val == vals["HYDR_U_G"])
        else:
            hb_gu_val = vals["HYDR_U_G"]
    assert(hb_gu_val is not None)
    hb_gu_mult = hb_gu_val / default_f1_eps_hb

    hb_mult[RNA_ALPHA.index("A"), RNA_ALPHA.index("U")] = hb_au_mult
    hb_mult[RNA_ALPHA.index("U"), RNA_ALPHA.index("A")] = hb_au_mult
    hb_mult[RNA_ALPHA.index("G"), RNA_ALPHA.index("C")] = hb_gc_mult
    hb_mult[RNA_ALPHA.index("C"), RNA_ALPHA.index("G")] = hb_gc_mult
    hb_mult[RNA_ALPHA.index("G"), RNA_ALPHA.index("U")] = hb_gu_mult
    hb_mult[RNA_ALPHA.index("U"), RNA_ALPHA.index("G")] = hb_gu_mult


    # Parse cross stacking table
    cross_mult = onp.zeros((4, 4))
    default_cross_k = base_params["cross_stacking"]["k_cross"]

    for nuc1 in utils.RNA_ALPHA:
        for nuc2 in utils.RNA_ALPHA:
            nuc1_repr = nuc1 if nuc1 != "U" else "T"
            nuc2_repr = nuc2 if nuc2 != "U" else "T"
            key = f"CROSS_{nuc1_repr}_{nuc2_repr}"
            assert(key in vals)
            ss_cross_k = vals[key]
            prefactor = ss_cross_k / default_cross_k
            cross_mult[RNA_ALPHA.index(nuc1), RNA_ALPHA.index(nuc2)] = prefactor

    # Stacking
    stack_mult = onp.zeros((4, 4))
    eps_stack_base = base_params["stacking"]["eps_stack_base"]
    eps_stack_kt_coeff = base_params["stacking"]["eps_stack_kt_coeff"]

    tol_places = 3
    eps_stack_prime = vals["ST_T_DEP"]
    diff = onp.abs(eps_stack_prime - eps_stack_kt_coeff/eps_stack_base)
    assert(diff < 1*10**(-tol_places))

    for nuc1 in utils.RNA_ALPHA:
        for nuc2 in utils.RNA_ALPHA:
            nuc1_repr = nuc1 if nuc1 != "U" else "T"
            nuc2_repr = nuc2 if nuc2 != "U" else "T"
            key = f"STCK_{nuc1_repr}_{nuc2_repr}"

            assert(key in vals)

            ss_eps_stack_base = vals[key]
            prefactor = ss_eps_stack_base / eps_stack_base
            stack_mult[RNA_ALPHA.index(nuc1), RNA_ALPHA.index(nuc2)] = prefactor

    return hb_mult, stack_mult, cross_mult


if __name__ == "__main__":
    params = load(seq_avg=True)

    pdb.set_trace()
    print("done")
