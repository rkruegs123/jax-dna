import pdb
import toml
from pathlib import Path

from smoothing_fast import get_f1_smoothing_params, get_f2_smoothing_params, get_f3_smoothing_params, \
    get_f4_smoothing_params, get_f5_smoothing_params
from utils import get_kt, DEFAULT_TEMP


# Temperature (t) in Kelvin
def get_params(params, t):
    kt = get_kt(t)

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
    params['stacking']['eps_stack'] = params['stacking']['eps_stack_base'] \
                                      + params['stacking']['eps_stack_kt_coeff'] * kt # Do this quickly so that it's included in `stacking`
    stacking = params['stacking']

    ## f1(dr_stack)
    b_backbone, dr_c_backbone = get_f3_smoothing_params(exc_vol['dr_star_backbone'],
                                                        exc_vol['eps_exc'],
                                                        exc_vol['sigma_backbone'])
    b_low_stack, dr_c_low_stack, b_high_stack, dr_c_high_stack = get_f1_smoothing_params(
        stacking['eps_stack'], stacking['dr0_stack'], stacking['a_stack'], stacking['dr_c_stack'],
        stacking['dr_low_stack'], stacking['dr_high_stack'])
    params['stacking']['b_low_stack'] = b_low_stack
    params['stacking']['dr_c_low_stack'] = dr_c_low_stack
    params['stacking']['b_high_stack'] = b_high_stack
    params['stacking']['dr_c_high_stack'] = dr_c_high_stack

    ## f4(theta_4)
    b_theta_4, delta_theta_4_c = get_f4_smoothing_params(stacking['a_stack_4'],
                                                         stacking['theta0_stack_4'],
                                                         stacking['delta_theta_star_stack_4'])
    params['stacking']['b_theta_4'] = b_theta_4
    params['stacking']['delta_theta_4_c'] = delta_theta_4_c

    ## f4(theta_5p)
    b_theta_5, delta_theta_5_c = get_f4_smoothing_params(stacking['a_stack_5'],
                                                         stacking['theta0_stack_5'],
                                                         stacking['delta_theta_star_stack_5'])
    params['stacking']['b_theta_5'] = b_theta_5
    params['stacking']['delta_theta_5_c'] = delta_theta_5_c

    ## f4(theta_6p)
    b_theta_6, delta_theta_6_c = get_f4_smoothing_params(stacking['a_stack_6'],
                                                         stacking['theta0_stack_6'],
                                                         stacking['delta_theta_star_stack_6'])
    params['stacking']['b_theta_6'] = b_theta_6
    params['stacking']['delta_theta_6_c'] = delta_theta_6_c

    ## f5(-cos(phi1))
    b_neg_cos_phi1, neg_cos_phi1_c = get_f5_smoothing_params(stacking['a_stack_1'],
                                                             stacking['neg_cos_phi1_star_stack'])
    params['stacking']['b_neg_cos_phi1'] = b_neg_cos_phi1
    params['stacking']['neg_cos_phi1_c'] = neg_cos_phi1_c

    ## f5(-cos(phi2))
    b_neg_cos_phi2, neg_cos_phi2_c = get_f5_smoothing_params(stacking['a_stack_2'],
                                                             stacking['neg_cos_phi2_star_stack'])
    params['stacking']['b_neg_cos_phi2'] = b_neg_cos_phi2
    params['stacking']['neg_cos_phi2_c'] = neg_cos_phi2_c


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
    b_theta_1, delta_theta_1_c = get_f4_smoothing_params(hb['a_hb_1'],
                                                         hb['theta0_hb_1'],
                                                         hb['delta_theta_star_hb_1'])
    params['hydrogen_bonding']['b_theta_1'] = b_theta_1
    params['hydrogen_bonding']['delta_theta_1_c'] = delta_theta_1_c

    ## f4(theta_2)
    b_theta_2, delta_theta_2_c = get_f4_smoothing_params(hb['a_hb_2'],
                                                         hb['theta0_hb_2'],
                                                         hb['delta_theta_star_hb_2'])
    params['hydrogen_bonding']['b_theta_2'] = b_theta_2
    params['hydrogen_bonding']['delta_theta_2_c'] = delta_theta_2_c

    ## f4(theta_3)
    b_theta_3, delta_theta_3_c = get_f4_smoothing_params(hb['a_hb_3'],
                                                         hb['theta0_hb_3'],
                                                         hb['delta_theta_star_hb_3'])
    params['hydrogen_bonding']['b_theta_3'] = b_theta_3
    params['hydrogen_bonding']['delta_theta_3_c'] = delta_theta_3_c

    ## f4(theta_4)
    b_theta_4, delta_theta_4_c = get_f4_smoothing_params(hb['a_hb_4'],
                                                         hb['theta0_hb_4'],
                                                         hb['delta_theta_star_hb_4'])
    params['hydrogen_bonding']['b_theta_4'] = b_theta_4
    params['hydrogen_bonding']['delta_theta_4_c'] = delta_theta_4_c

    ## f4(theta_7)
    b_theta_7, delta_theta_7_c = get_f4_smoothing_params(hb['a_hb_7'],
                                                         hb['theta0_hb_7'],
                                                         hb['delta_theta_star_hb_7'])
    params['hydrogen_bonding']['b_theta_7'] = b_theta_7
    params['hydrogen_bonding']['delta_theta_7_c'] = delta_theta_7_c

    ## f4(theta_8)
    b_theta_8, delta_theta_8_c = get_f4_smoothing_params(hb['a_hb_8'],
                                                         hb['theta0_hb_8'],
                                                         hb['delta_theta_star_hb_8'])
    params['hydrogen_bonding']['b_theta_8'] = b_theta_8
    params['hydrogen_bonding']['delta_theta_8_c'] = delta_theta_8_c


    # Cross Stacking
    cross = params['cross_stacking']

    ## f2(dr_hb) (FIXME: is the _hb a typo? Should be _cross_stack, no?)
    b_low_cross, dr_c_low_cross, b_high_cross, dr_c_high_cross = get_f2_smoothing_params(
        cross['k'],
        cross['r0_cross'],
        cross['dr_c_cross'],
        cross['dr_low_cross'],
        cross['dr_high_cross'])
    params['cross_stacking']['b_low_cross'] = b_low_cross
    params['cross_stacking']['dr_c_low_cross'] = dr_c_low_cross
    params['cross_stacking']['b_high_cross'] = b_high_cross
    params['cross_stacking']['dr_c_high_cross'] = dr_c_high_cross

    ## f4(theta_1)
    b_theta_1, delta_theta_1_c = get_f4_smoothing_params(cross['a_cross_1'],
                                                         cross['theta0_cross_1'],
                                                         cross['delta_theta_star_cross_1'])
    params['cross_stacking']['b_theta_1'] = b_theta_1
    params['cross_stacking']['delta_theta_1_c'] = delta_theta_1_c

    ## f4(theta_2)
    b_theta_2, delta_theta_2_c = get_f4_smoothing_params(cross['a_cross_2'],
                                                         cross['theta0_cross_2'],
                                                         cross['delta_theta_star_cross_2'])
    params['cross_stacking']['b_theta_2'] = b_theta_2
    params['cross_stacking']['delta_theta_2_c'] = delta_theta_2_c

    ## f4(theta_3)
    b_theta_3, delta_theta_3_c = get_f4_smoothing_params(cross['a_cross_3'],
                                                         cross['theta0_cross_3'],
                                                         cross['delta_theta_star_cross_3'])
    params['cross_stacking']['b_theta_3'] = b_theta_3
    params['cross_stacking']['delta_theta_3_c'] = delta_theta_3_c

    ## f4(theta_4) + f4(pi - theta_4)
    b_theta_4, delta_theta_4_c = get_f4_smoothing_params(cross['a_cross_4'],
                                                         cross['theta0_cross_4'],
                                                         cross['delta_theta_star_cross_4'])
    params['cross_stacking']['b_theta_4'] = b_theta_4
    params['cross_stacking']['delta_theta_4_c'] = delta_theta_4_c

    ## f4(theta_7) + f4(pi - theta_7)
    b_theta_7, delta_theta_7_c = get_f4_smoothing_params(cross['a_cross_7'],
                                                         cross['theta0_cross_7'],
                                                         cross['delta_theta_star_cross_7'])
    params['cross_stacking']['b_theta_7'] = b_theta_7
    params['cross_stacking']['delta_theta_7_c'] = delta_theta_7_c

    ## f4(theta_8) + f4(pi - theta_8)
    b_theta_8, delta_theta_8_c = get_f4_smoothing_params(cross['a_cross_8'],
                                                         cross['theta0_cross_8'],
                                                         cross['delta_theta_star_cross_8'])
    params['cross_stacking']['b_theta_8'] = b_theta_8
    params['cross_stacking']['delta_theta_8_c'] = delta_theta_8_c


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
    b_theta_1, delta_theta_1_c = get_f4_smoothing_params(coax['a_coax_1'],
                                                         coax['theta0_coax_1'],
                                                         coax['delta_theta_star_coax_1'])
    params['coaxial_stacking']['b_theta_1'] = b_theta_1
    params['coaxial_stacking']['delta_theta_1_c'] = delta_theta_1_c

    ## f4(theta_4)
    b_theta_4, delta_theta_4_c = get_f4_smoothing_params(coax['a_coax_4'],
                                                         coax['theta0_coax_4'],
                                                         coax['delta_theta_star_coax_4'])
    params['coaxial_stacking']['b_theta_4'] = b_theta_4
    params['coaxial_stacking']['delta_theta_4_c'] = delta_theta_4_c

    ## f4(theta_5) + f4(pi - theta_5)
    b_theta_5, delta_theta_5_c = get_f4_smoothing_params(coax['a_coax_5'],
                                                         coax['theta0_coax_5'],
                                                         coax['delta_theta_star_coax_5'])
    params['coaxial_stacking']['b_theta_5'] = b_theta_5
    params['coaxial_stacking']['delta_theta_5_c'] = delta_theta_5_c

    ## f4(theta_6) + f4(pi - theta_6)
    b_theta_6, delta_theta_6_c = get_f4_smoothing_params(coax['a_coax_6'],
                                                         coax['theta0_coax_6'],
                                                         coax['delta_theta_star_coax_6'])
    params['coaxial_stacking']['b_theta_6'] = b_theta_6
    params['coaxial_stacking']['delta_theta_6_c'] = delta_theta_6_c

    ## f5(cos(phi3))
    b_cos_phi3, cos_phi3_c = get_f5_smoothing_params(coax['a_coax_3p'],
                                                     coax['cos_phi3_star_coax'])
    params['coaxial_stacking']['b_cos_phi3'] = b_cos_phi3
    params['coaxial_stacking']['cos_phi3_c'] = cos_phi3_c

    ## f5(cos(phi4))
    b_cos_phi4, cos_phi4_c = get_f5_smoothing_params(coax['a_coax_4p'],
                                                     coax['cos_phi4_star_coax'])
    params['coaxial_stacking']['b_cos_phi4'] = b_cos_phi4
    params['coaxial_stacking']['cos_phi4_c'] = cos_phi4_c

    return params


def get_default_params(params_path="tom.toml"):
    if not Path(params_path).exists():
        raise RuntimeError(f"No file at location: {params_path}")
    params = toml.load(params_path)
    return get_params(params, t=DEFAULT_TEMP)


if __name__ == "__main__":
    get_default_params()
