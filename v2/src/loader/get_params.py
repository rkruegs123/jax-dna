import pdb
import toml
from pathlib import Path

from loader.smoothing import get_f1_smoothing_params, get_f2_smoothing_params, \
    get_f3_smoothing_params, get_f4_smoothing_params, get_f5_smoothing_params
from utils import get_kt, DEFAULT_TEMP


def add_misc(params, t):
    kt = get_kt(t)
    params['stacking']['eps_stack'] = params['stacking']['eps_stack_base'] \
                                      + params['stacking']['eps_stack_kt_coeff'] * kt
    return params

def add_smoothing(params):

    # Excluded Volume
    exc_vol = params['excluded_volume']

    ## f3(dr_backbone)
    b_backbone, dr_c_backbone = get_f3_smoothing_params(exc_vol['dr_star_backbone'],
                                                        exc_vol['eps_exc'],
                                                        exc_vol['sigma_backbone'])
    params['excluded_volume']['b_backbone'] = b_backbone
    params['excluded_volume']['dr_c_backbone'] = dr_c_backbone

    ## f3(dr_base)
    b_base, dr_c_base = get_f3_smoothing_params(exc_vol['dr_star_base'],
                                                exc_vol['eps_exc'],
                                                exc_vol['sigma_base'])
    params['excluded_volume']['b_base'] = b_base
    params['excluded_volume']['dr_c_base'] = dr_c_base

    ## f3(dr_back_base)
    b_back_base, dr_c_back_base = get_f3_smoothing_params(exc_vol['dr_star_back_base'],
                                                          exc_vol['eps_exc'],
                                                          exc_vol['sigma_back_base'])
    params['excluded_volume']['b_back_base'] = b_back_base
    params['excluded_volume']['dr_c_back_base'] = dr_c_back_base

    ## f3(dr_base_back)
    b_base_back, dr_c_base_back = get_f3_smoothing_params(exc_vol['dr_star_base_back'],
                                                          exc_vol['eps_exc'],
                                                          exc_vol['sigma_base_back'])
    params['excluded_volume']['b_base_back'] = b_base_back
    params['excluded_volume']['dr_c_base_back'] = dr_c_base_back


    # Stacking
    stacking = params['stacking']

    ## f1(dr_stack)
    b_low_stack, dr_c_low_stack, b_high_stack, dr_c_high_stack = get_f1_smoothing_params(
        stacking['eps_stack'], stacking['dr0_stack'], stacking['a_stack'], stacking['dr_c_stack'],
        stacking['dr_low_stack'], stacking['dr_high_stack'])
    params['stacking']['b_low_stack'] = b_low_stack
    params['stacking']['dr_c_low_stack'] = dr_c_low_stack
    params['stacking']['b_high_stack'] = b_high_stack
    params['stacking']['dr_c_high_stack'] = dr_c_high_stack

    ## f4(theta_4)
    b_stack_4, delta_theta_stack_4_c = get_f4_smoothing_params(stacking['a_stack_4'],
                                                               stacking['theta0_stack_4'],
                                                               stacking['delta_theta_star_stack_4'])
    params['stacking']['b_stack_4'] = b_stack_4
    params['stacking']['delta_theta_stack_4_c'] = delta_theta_stack_4_c

    ## f4(theta_5p)
    b_stack_5, delta_theta_stack_5_c = get_f4_smoothing_params(stacking['a_stack_5'],
                                                               stacking['theta0_stack_5'],
                                                               stacking['delta_theta_star_stack_5'])
    params['stacking']['b_stack_5'] = b_stack_5
    params['stacking']['delta_theta_stack_5_c'] = delta_theta_stack_5_c

    ## f4(theta_6p)
    b_stack_6, delta_theta_stack_6_c = get_f4_smoothing_params(stacking['a_stack_6'],
                                                               stacking['theta0_stack_6'],
                                                               stacking['delta_theta_star_stack_6'])
    params['stacking']['b_stack_6'] = b_stack_6
    params['stacking']['delta_theta_stack_6_c'] = delta_theta_stack_6_c

    ## f5(-cos(phi1))
    b_neg_cos_phi1_stack, neg_cos_phi1_c_stack = get_f5_smoothing_params(stacking['a_stack_1'],
                                                                         stacking['neg_cos_phi1_star_stack'])
    params['stacking']['b_neg_cos_phi1_stack'] = b_neg_cos_phi1_stack
    params['stacking']['neg_cos_phi1_c_stack'] = neg_cos_phi1_c_stack

    ## f5(-cos(phi2))
    b_neg_cos_phi2_stack, neg_cos_phi2_c_stack = get_f5_smoothing_params(stacking['a_stack_2'],
                                                                         stacking['neg_cos_phi2_star_stack'])
    params['stacking']['b_neg_cos_phi2_stack'] = b_neg_cos_phi2_stack
    params['stacking']['neg_cos_phi2_c_stack'] = neg_cos_phi2_c_stack


    # Hydrogen Bonding
    hb = params['hydrogen_bonding']

    ## f1(dr_hb)
    b_low_hb, dr_c_low_hb, b_high_hb, dr_c_high_hb = get_f1_smoothing_params(
        hb['eps_hb'], hb['dr0_hb'], hb['a_hb'], hb['dr_c_hb'],
        hb['dr_low_hb'], hb['dr_high_hb'])
    params['hydrogen_bonding']['b_low_hb'] = b_low_hb
    params['hydrogen_bonding']['dr_c_low_hb'] = dr_c_low_hb
    params['hydrogen_bonding']['b_high_hb'] = b_high_hb
    params['hydrogen_bonding']['dr_c_high_hb'] = dr_c_high_hb

    ## f4(theta_1)
    b_hb_1, delta_theta_hb_1_c = get_f4_smoothing_params(hb['a_hb_1'],
                                                         hb['theta0_hb_1'],
                                                         hb['delta_theta_star_hb_1'])
    params['hydrogen_bonding']['b_hb_1'] = b_hb_1
    params['hydrogen_bonding']['delta_theta_hb_1_c'] = delta_theta_hb_1_c

    ## f4(theta_2)
    b_hb_2, delta_theta_hb_2_c = get_f4_smoothing_params(hb['a_hb_2'],
                                                         hb['theta0_hb_2'],
                                                         hb['delta_theta_star_hb_2'])
    params['hydrogen_bonding']['b_hb_2'] = b_hb_2
    params['hydrogen_bonding']['delta_theta_hb_2_c'] = delta_theta_hb_2_c

    ## f4(theta_3)
    b_hb_3, delta_theta_hb_3_c = get_f4_smoothing_params(hb['a_hb_3'],
                                                         hb['theta0_hb_3'],
                                                         hb['delta_theta_star_hb_3'])
    params['hydrogen_bonding']['b_hb_3'] = b_hb_3
    params['hydrogen_bonding']['delta_theta_hb_3_c'] = delta_theta_hb_3_c

    ## f4(theta_4)
    b_hb_4, delta_theta_hb_4_c = get_f4_smoothing_params(hb['a_hb_4'],
                                                         hb['theta0_hb_4'],
                                                         hb['delta_theta_star_hb_4'])
    params['hydrogen_bonding']['b_hb_4'] = b_hb_4
    params['hydrogen_bonding']['delta_theta_hb_4_c'] = delta_theta_hb_4_c

    ## f4(theta_7)
    b_hb_7, delta_theta_hb_7_c = get_f4_smoothing_params(hb['a_hb_7'],
                                                         hb['theta0_hb_7'],
                                                         hb['delta_theta_star_hb_7'])
    params['hydrogen_bonding']['b_hb_7'] = b_hb_7
    params['hydrogen_bonding']['delta_theta_hb_7_c'] = delta_theta_hb_7_c

    ## f4(theta_8)
    b_hb_8, delta_theta_hb_8_c = get_f4_smoothing_params(hb['a_hb_8'],
                                                         hb['theta0_hb_8'],
                                                         hb['delta_theta_star_hb_8'])
    params['hydrogen_bonding']['b_hb_8'] = b_hb_8
    params['hydrogen_bonding']['delta_theta_hb_8_c'] = delta_theta_hb_8_c


    # Cross Stacking
    cross = params['cross_stacking']

    ## f2(dr_hb) (FIXME: is the _hb a typo? Should be _cross_stack, no?)
    b_low_cross, dr_c_low_cross, b_high_cross, dr_c_high_cross = get_f2_smoothing_params(
        cross['k_cross'],
        cross['r0_cross'],
        cross['dr_c_cross'],
        cross['dr_low_cross'],
        cross['dr_high_cross'])
    params['cross_stacking']['b_low_cross'] = b_low_cross
    params['cross_stacking']['dr_c_low_cross'] = dr_c_low_cross
    params['cross_stacking']['b_high_cross'] = b_high_cross
    params['cross_stacking']['dr_c_high_cross'] = dr_c_high_cross

    ## f4(theta_1)
    b_cross_1, delta_theta_cross_1_c = get_f4_smoothing_params(cross['a_cross_1'],
                                                               cross['theta0_cross_1'],
                                                               cross['delta_theta_star_cross_1'])
    params['cross_stacking']['b_cross_1'] = b_cross_1
    params['cross_stacking']['delta_theta_cross_1_c'] = delta_theta_cross_1_c

    ## f4(theta_2)
    b_cross_2, delta_theta_cross_2_c = get_f4_smoothing_params(cross['a_cross_2'],
                                                               cross['theta0_cross_2'],
                                                               cross['delta_theta_star_cross_2'])
    params['cross_stacking']['b_cross_2'] = b_cross_2
    params['cross_stacking']['delta_theta_cross_2_c'] = delta_theta_cross_2_c

    ## f4(theta_3)
    b_cross_3, delta_theta_cross_3_c = get_f4_smoothing_params(cross['a_cross_3'],
                                                               cross['theta0_cross_3'],
                                                               cross['delta_theta_star_cross_3'])
    params['cross_stacking']['b_cross_3'] = b_cross_3
    params['cross_stacking']['delta_theta_cross_3_c'] = delta_theta_cross_3_c

    ## f4(theta_4) + f4(pi - theta_4)
    b_cross_4, delta_theta_cross_4_c = get_f4_smoothing_params(cross['a_cross_4'],
                                                               cross['theta0_cross_4'],
                                                               cross['delta_theta_star_cross_4'])
    params['cross_stacking']['b_cross_4'] = b_cross_4
    params['cross_stacking']['delta_theta_cross_4_c'] = delta_theta_cross_4_c

    ## f4(theta_7) + f4(pi - theta_7)
    b_cross_7, delta_theta_cross_7_c = get_f4_smoothing_params(cross['a_cross_7'],
                                                               cross['theta0_cross_7'],
                                                               cross['delta_theta_star_cross_7'])
    params['cross_stacking']['b_cross_7'] = b_cross_7
    params['cross_stacking']['delta_theta_cross_7_c'] = delta_theta_cross_7_c

    ## f4(theta_8) + f4(pi - theta_8)
    b_cross_8, delta_theta_cross_8_c = get_f4_smoothing_params(cross['a_cross_8'],
                                                               cross['theta0_cross_8'],
                                                               cross['delta_theta_star_cross_8'])
    params['cross_stacking']['b_cross_8'] = b_cross_8
    params['cross_stacking']['delta_theta_cross_8_c'] = delta_theta_cross_8_c


    # Coaxial Stacking
    coax = params['coaxial_stacking']

    ## f2(dr_coax)
    b_low_coax, dr_c_low_coax, b_high_coax, dr_c_high_coax = get_f2_smoothing_params(
        coax['k_coax'],
        coax['dr0_coax'],
        coax['dr_c_coax'],
        coax['dr_low_coax'],
        coax['dr_high_coax'])
    params['coaxial_stacking']['b_low_coax'] = b_low_coax
    params['coaxial_stacking']['dr_c_low_coax'] = dr_c_low_coax
    params['coaxial_stacking']['b_high_coax'] = b_high_coax
    params['coaxial_stacking']['dr_c_high_coax'] = dr_c_high_coax

    ## f4(theta_1) + f4(2*pi - theta_1)
    b_coax_1, delta_theta_coax_1_c = get_f4_smoothing_params(coax['a_coax_1'],
                                                             coax['theta0_coax_1'],
                                                             coax['delta_theta_star_coax_1'])
    params['coaxial_stacking']['b_coax_1'] = b_coax_1
    params['coaxial_stacking']['delta_theta_coax_1_c'] = delta_theta_coax_1_c

    ## f4(theta_4)
    b_coax_4, delta_theta_coax_4_c = get_f4_smoothing_params(coax['a_coax_4'],
                                                             coax['theta0_coax_4'],
                                                             coax['delta_theta_star_coax_4'])
    params['coaxial_stacking']['b_coax_4'] = b_coax_4
    params['coaxial_stacking']['delta_theta_coax_4_c'] = delta_theta_coax_4_c

    ## f4(theta_5) + f4(pi - theta_5)
    b_coax_5, delta_theta_coax_5_c = get_f4_smoothing_params(coax['a_coax_5'],
                                                             coax['theta0_coax_5'],
                                                             coax['delta_theta_star_coax_5'])
    params['coaxial_stacking']['b_coax_5'] = b_coax_5
    params['coaxial_stacking']['delta_theta_coax_5_c'] = delta_theta_coax_5_c

    ## f4(theta_6) + f4(pi - theta_6)
    b_coax_6, delta_theta_coax_6_c = get_f4_smoothing_params(coax['a_coax_6'],
                                                             coax['theta0_coax_6'],
                                                             coax['delta_theta_star_coax_6'])
    params['coaxial_stacking']['b_coax_6'] = b_coax_6
    params['coaxial_stacking']['delta_theta_coax_6_c'] = delta_theta_coax_6_c

    ## f5(cos(phi3))
    b_cos_phi3_coax, cos_phi3_c_coax = get_f5_smoothing_params(coax['a_coax_3p'],
                                                               coax['cos_phi3_star_coax'])
    params['coaxial_stacking']['b_cos_phi3_coax'] = b_cos_phi3_coax
    params['coaxial_stacking']['cos_phi3_c_coax'] = cos_phi3_c_coax

    ## f5(cos(phi4))
    b_cos_phi4_coax, cos_phi4_c_coax = get_f5_smoothing_params(coax['a_coax_4p'],
                                                               coax['cos_phi4_star_coax'])
    params['coaxial_stacking']['b_cos_phi4_coax'] = b_cos_phi4_coax
    params['coaxial_stacking']['cos_phi4_c_coax'] = cos_phi4_c_coax

    return params

def remove_keys(params):
    del params['stacking']['eps_stack_kt_coeff']
    del params['stacking']['eps_stack_base']
    return params

# Temperature (t) in Kelvin
def get_params(params, t, no_smoothing=False):
    params = add_misc(params, t)
    if not no_smoothing:
        params = add_smoothing(params)
    params = remove_keys(params)
    return params


def get_default_params(params_path="v2/params/tom.toml", t=DEFAULT_TEMP, no_smoothing=False):
    if not Path(params_path).exists():
        raise RuntimeError(f"No file at location: {params_path}")
    params = toml.load(params_path)
    return get_params(params, t=t, no_smoothing=no_smoothing)


def get_init_optimize_params(method="oxdna"):
    if method == "oxdna":
        # starting with the correct parameters
        init_fene_params = [2.0, 0.25, 0.7525]
        init_stacking_params = [
            1.3448, 2.6568, 6.0, 0.4, 0.9, 0.32, 0.75, # f1(dr_stack)
            1.30, 0.0, 0.8, # f4(theta_4)
            0.90, 0.0, 0.95, # f4(theta_5p)
            0.90, 0.0, 0.95, # f4(theta_6p)
            2.0, -0.65, # f5(-cos(phi1))
            2.0, -0.65 # f5(-cos(phi2))
        ]
    elif method == "random":
        init_fene_params = [0.60, 0.75, 1.1]
        init_stacking_params = [
            0.25, 0.7, 2.0, 0.4, 1.2, 1.3, 0.2, # f1(dr_stack)
            0.5, 0.35, 0.6, # f4(theta_4)
            1.5, 1.1, 0.3, # f4(theta_5p)
            2.0, 0.2, 0.75, # f4(theta_6p)
            0.7, 2.0, # f5(-cos(phi1))
            1.3, 0.8 # f5(-cos(phi2))
        ]
    else:
        raise RuntimeError(f"Invalid method: {method}")
    return init_fene_params + init_stacking_params

def get_init_optimize_params_hb_seq_dependent(method="oxdna"):
    assert(method == "oxdna")

    # starting with the correct parameters
    init_f4_a_vals = [1.50, 1.50, 1.50, 0.46, 4.0, 4.0]

    return [init_f4_a_vals, init_f4_a_vals]

# FIXME: Used for processing arrays in energy function, but really duplicate logic from when we actually read in the parameters...
def process_stacking_params(unprocessed_params, kt):
    a_stack = unprocessed_params['a_stack']
    dr0_stack = unprocessed_params['dr0_stack']
    dr_c_stack = unprocessed_params['dr_c_stack']
    dr_low_stack = unprocessed_params['dr_low_stack']
    dr_high_stack = unprocessed_params['dr_high_stack']
    eps_stack_base = unprocessed_params['eps_stack_base']
    eps_stack_kt_coeff = unprocessed_params['eps_stack_kt_coeff']
    eps_stack = eps_stack_base + eps_stack_kt_coeff * kt

    b_low_stack, dr_c_low_stack, b_high_stack, dr_c_high_stack = get_f1_smoothing_params(
        eps_stack, dr0_stack, a_stack, dr_c_stack,
        dr_low_stack, dr_high_stack)

    a_stack_4 = unprocessed_params['a_stack_4']
    theta0_stack_4 = unprocessed_params['theta0_stack_4']
    delta_theta_star_stack_4 = unprocessed_params['delta_theta_star_stack_4']
    b_stack_4, delta_theta_stack_4_c = get_f4_smoothing_params(
        a_stack_4,
        theta0_stack_4,
        delta_theta_star_stack_4)

    a_stack_5 = unprocessed_params['a_stack_5']
    theta0_stack_5 = unprocessed_params['theta0_stack_5']
    delta_theta_star_stack_5 = unprocessed_params['delta_theta_star_stack_5']
    b_stack_5, delta_theta_stack_5_c = get_f4_smoothing_params(
        a_stack_5,
        theta0_stack_5,
        delta_theta_star_stack_5)

    a_stack_6 = unprocessed_params['a_stack_6']
    theta0_stack_6 = unprocessed_params['theta0_stack_6']
    delta_theta_star_stack_6 = unprocessed_params['delta_theta_star_stack_6']
    b_stack_6, delta_theta_stack_6_c = get_f4_smoothing_params(
        a_stack_6,
        theta0_stack_6,
        delta_theta_star_stack_6)

    a_stack_1 = unprocessed_params['a_stack_1']
    neg_cos_phi1_star_stack = unprocessed_params['neg_cos_phi1_star_stack']
    b_neg_cos_phi1_stack, neg_cos_phi1_c_stack = get_f5_smoothing_params(
        a_stack_1,
        neg_cos_phi1_star_stack)

    a_stack_2 = unprocessed_params['a_stack_2']
    neg_cos_phi2_star_stack = unprocessed_params['neg_cos_phi2_star_stack']
    b_neg_cos_phi2_stack, neg_cos_phi2_c_stack = get_f5_smoothing_params(
        a_stack_2,
        neg_cos_phi2_star_stack)

    processed_params = {
        # f1(dr_stack)
        "eps_stack": eps_stack,
        "dr0_stack": dr0_stack,
        "a_stack": a_stack,
        "dr_c_stack": dr_c_stack,
        "dr_low_stack": dr_low_stack,
        "dr_high_stack": dr_high_stack,
        "b_low_stack": b_low_stack,
        "dr_c_low_stack": dr_c_low_stack,
        "b_high_stack": b_high_stack,
        "dr_c_high_stack": dr_c_high_stack,

        # f4(theta_4)
        "a_stack_4": a_stack_4,
        "theta0_stack_4": theta0_stack_4,
        "delta_theta_star_stack_4": delta_theta_star_stack_4,
        "b_stack_4": b_stack_4,
        "delta_theta_stack_4_c": delta_theta_stack_4_c,

        # f4(theta_5p)
        "a_stack_5": a_stack_5,
        "theta0_stack_5": theta0_stack_5,
        "delta_theta_star_stack_5": delta_theta_star_stack_5,
        "b_stack_5": b_stack_5,
        "delta_theta_stack_5_c": delta_theta_stack_5_c,

        # f4(theta_6p)
        "a_stack_6": a_stack_6,
        "theta0_stack_6": theta0_stack_6,
        "delta_theta_star_stack_6": delta_theta_star_stack_6,
        "b_stack_6": b_stack_6,
        "delta_theta_stack_6_c": delta_theta_stack_6_c,

        ## f5(-cos(phi1))
        "a_stack_1": a_stack_1,
        "neg_cos_phi1_star_stack": neg_cos_phi1_star_stack,
        "b_neg_cos_phi1_stack": b_neg_cos_phi1_stack,
        "neg_cos_phi1_c_stack": neg_cos_phi1_c_stack,

        ## f5(-cos(phi2))
        "a_stack_2": a_stack_2,
        "neg_cos_phi2_star_stack": neg_cos_phi2_star_stack,
        "b_neg_cos_phi2_stack": b_neg_cos_phi2_stack,
        "neg_cos_phi2_c_stack": neg_cos_phi2_c_stack

    }
    return processed_params



if __name__ == "__main__":
    get_default_params()
