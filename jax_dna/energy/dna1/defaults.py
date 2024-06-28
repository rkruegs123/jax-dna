import jax.numpy as jnp
import numpy as np

import jax_dna.energy.dna1.base_smoothing_functions as bsf

T_KELVIN = 296.15

STRUCTURE = {
    "com_to_stacking": 0.34,
    "com_to_hb": 0.4,
    "com_to_backbone": -0.4,
    # Flattened HB/stack weights yield the correct Kron. product
    # Kron: AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT
    "hb_weights_sa": jnp.array([
                        [0.0, 0.0, 0.0, 1.0], # AX
                        [0.0, 0.0, 1.0, 0.0], # CX
                        [0.0, 1.0, 0.0, 0.0], # GX
                        [1.0, 0.0, 0.0, 0.0]  # TX
                    ]),
    "stack_weights_sa": jnp.array([
                            [1.0, 1.0, 1.0, 1.0], # AX
                            [1.0, 1.0, 1.0, 1.0], # CX
                            [1.0, 1.0, 1.0, 1.0], # GX
                            [1.0, 1.0, 1.0, 1.0]  # TX
                        ]),
}

HB_WEIGHTS_SA = jnp.array([
    [0.0, 0.0, 0.0, 1.0], # AX
    [0.0, 0.0, 1.0, 0.0], # CX
    [0.0, 1.0, 0.0, 0.0], # GX
    [1.0, 0.0, 0.0, 0.0]  # TX
])

STACK_WEIGHTS_SA = jnp.array([
    [1.0, 1.0, 1.0, 1.0], # AX
    [1.0, 1.0, 1.0, 1.0], # CX
    [1.0, 1.0, 1.0, 1.0], # GX
    [1.0, 1.0, 1.0, 1.0]  # TX
])

V_FENE = {
    "eps_backbone": 2.0,
    "r0_backbone": 0.7525,
    "delta_backbone": 0.25,
    "fmax": 500,
    "finf": 4.0,
}

def _init_fene(params: dict[str, float]) -> dict[str, float]:
    return params


V_FENE = _init_fene(V_FENE)

BONDED_DG = {
    "eps_exc": 2.0,
    "dr_star_base": 0.32,
    "sigma_base": 0.33,
    "dr_star_back_base": 0.50,
    "sigma_back_base": 0.515,
    "sigma_base_back": 0.515,
    "dr_star_base_back": 0.50,
}


def _init_excluded_volume_dg(params: dict[str, float], unbonded: bool) -> dict[str, float]:
    calculated_params = {}
    ## f3(dr_base)
    b_base, dr_c_base = bsf.get_f3_smoothing_params(params["dr_star_base"], params["eps_exc"], params["sigma_base"])
    calculated_params["b_base"] = b_base
    calculated_params["dr_c_base"] = dr_c_base

    ## f3(dr_back_base)
    b_back_base, dr_c_back_base = bsf.get_f3_smoothing_params(
        params["dr_star_back_base"], params["eps_exc"], params["sigma_back_base"]
    )
    calculated_params["b_back_base"] = b_back_base
    calculated_params["dr_c_back_base"] = dr_c_back_base

    ## f3(dr_base_back)
    b_base_back, dr_c_base_back = bsf.get_f3_smoothing_params(
        params["dr_star_base_back"],
        params["eps_exc"],
        params["sigma_base_back"],
    )
    calculated_params["b_base_back"] = b_base_back
    calculated_params["dr_c_base_back"] = dr_c_base_back

    if unbonded:
        ## f3(dr_backbone)
        b_backbone, dr_c_backbone = bsf.get_f3_smoothing_params(
            params["dr_star_backbone"],
            params["eps_exc"],
            params["sigma_backbone"],
        )
        calculated_params["b_backbone"] = b_backbone
        calculated_params["dr_c_backbone"] = dr_c_backbone

    return calculated_params | params


BONDED_DG = _init_excluded_volume_dg(BONDED_DG, False)

UNBONDED_DG = {
    "eps_exc": 2.0,
    "dr_star_base": 0.32,
    "sigma_base": 0.33,
    "dr_star_back_base": 0.50,
    "sigma_back_base": 0.515,
    "dr_star_base_back": 0.50,
    "sigma_base_back": 0.515,
    "dr_star_backbone": 0.675,
    "sigma_backbone": 0.70,
}

UNBONDED_DG = _init_excluded_volume_dg(UNBONDED_DG, True)

STACKING_DG = {
    "dr_low_stack": 0.32,
    "dr_high_stack": 0.75,
    "eps_stack": 1.3448,  # esp_stack_base?
    "a_stack": 6.0,
    "dr0_stack": 0.4,
    "dr_c_stack": 0.9,
    "theta0_stack_4": 0.0,
    "delta_theta_star_stack_4": 0.8,
    "a_stack_4": 1.30,
    "theta0_stack_5": 0.0,
    "delta_theta_star_stack_5": 0.95,
    "a_stack_5": 0.90,
    "theta0_stack_6": 0.0,
    "delta_theta_star_stack_6": 0.95,
    "a_stack_6": 0.90,
    "neg_cos_phi1_star_stack": -0.65,
    "a_stack_1": 2.00,
    "neg_cos_phi2_star_stack": -0.65,
    "a_stack_2": 2.00,
}


def _init_stacking(params: dict[str, float]) -> dict[str, float]:
    calculated_params = {}
    ## f1(dr_stack)
    (b_low_stack, dr_c_low_stack, b_high_stack, dr_c_high_stack) = bsf.get_f1_smoothing_params(
        params["eps_stack"],
        params["dr0_stack"],
        params["a_stack"],
        params["dr_c_stack"],
        params["dr_low_stack"],
        params["dr_high_stack"],
    )

    calculated_params["b_low_stack"] = b_low_stack
    calculated_params["dr_c_low_stack"] = dr_c_low_stack  # (ryanhausen) does this get used?
    calculated_params["b_high_stack"] = b_high_stack
    calculated_params["dr_c_high_stack"] = dr_c_high_stack  # (ryanhausen) does this get used?

    ## f4(theta_4)
    b_stack_4, delta_theta_stack_4_c = bsf.get_f4_smoothing_params(
        params["a_stack_4"],
        params["theta0_stack_4"],
        params["delta_theta_star_stack_4"],
    )
    calculated_params["delta_theta_stack_4_c"] = delta_theta_stack_4_c
    calculated_params["b_stack_4"] = b_stack_4

    ## f4(theta_5p)
    b_stack_5, delta_theta_stack_5_c = bsf.get_f4_smoothing_params(
        params["a_stack_5"], params["theta0_stack_5"], params["delta_theta_star_stack_5"]
    )
    calculated_params["delta_theta_stack_5_c"] = delta_theta_stack_5_c
    calculated_params["b_stack_5"] = b_stack_5

    ## f4(theta_6p)
    b_stack_6, delta_theta_stack_6_c = bsf.get_f4_smoothing_params(
        params["a_stack_6"],
        params["theta0_stack_6"],
        params["delta_theta_star_stack_6"],
    )
    params["delta_theta_stack_6_c"] = delta_theta_stack_6_c
    params["b_stack_6"] = b_stack_6

    ## f5(-cos(phi1))
    b_neg_cos_phi1_stack, neg_cos_phi1_c_stack = bsf.get_f5_smoothing_params(
        params["a_stack_1"],
        params["neg_cos_phi1_star_stack"],
    )
    calculated_params["neg_cos_phi1_c_stack"] = neg_cos_phi1_c_stack
    calculated_params["b_neg_cos_phi1_stack"] = b_neg_cos_phi1_stack

    ## f5(-cos(phi2))
    b_neg_cos_phi2_stack, neg_cos_phi2_c_stack = bsf.get_f5_smoothing_params(
        params["a_stack_2"], params["neg_cos_phi2_star_stack"]
    )
    calculated_params["neg_cos_phi2_c_stack"] = neg_cos_phi2_c_stack
    calculated_params["b_neg_cos_phi2_stack"] = b_neg_cos_phi2_stack

    return calculated_params | params


STACKING_DG = _init_stacking(STACKING_DG)

CROSS_STACKING_DG = {
    "dr_low_cross": 0.495,
    "dr_high_cross": 0.655,
    "k_cross": 47.5,
    "r0_cross": 0.575,
    "dr_c_cross": 0.675,
    "theta0_cross_1": np.pi - 2.35,
    "delta_theta_star_cross_1": 0.58,
    "a_cross_1": 2.25,
    "theta0_cross_2": 1.00,
    "delta_theta_star_cross_2": 0.68,
    "a_cross_2": 1.70,
    "theta0_cross_3": 1.00,
    "delta_theta_star_cross_3": 0.68,
    "a_cross_3": 1.70,
    "theta0_cross_4": 0.0,
    "delta_theta_star_cross_4": 0.65,
    "a_cross_4": 1.50,
    "theta0_cross_7": 0.875,
    "delta_theta_star_cross_7": 0.68,
    "a_cross_7": 1.70,
    "theta0_cross_8": 0.875,
    "delta_theta_star_cross_8": 0.68,
    "a_cross_8": 1.70,
}


def _init_cross_stacking(params: dict[str, float]) -> dict[str, float]:
    calculated_params = {}
    ## f2(dr_hb)
    (
        b_low_cross,
        dr_c_low_cross,
        b_high_cross,
        dr_c_high_cross,
    ) = bsf.get_f2_smoothing_params(
        params["k_cross"], params["r0_cross"], params["dr_c_cross"], params["dr_low_cross"], params["dr_high_cross"]
    )
    calculated_params["b_low_cross"] = b_low_cross
    calculated_params["dr_c_low_cross"] = dr_c_low_cross
    calculated_params["b_high_cross"] = b_high_cross
    calculated_params["dr_c_high_cross"] = dr_c_high_cross

    ## f4(theta_1)
    b_cross_1, delta_theta_cross_1_c = bsf.get_f4_smoothing_params(
        params["a_cross_1"],
        params["theta0_cross_1"],
        params["delta_theta_star_cross_1"],
    )
    calculated_params["b_cross_1"] = b_cross_1
    calculated_params["delta_theta_cross_1_c"] = delta_theta_cross_1_c

    ## f4(theta_2)
    b_cross_2, delta_theta_cross_2_c = bsf.get_f4_smoothing_params(
        params["a_cross_2"],
        params["theta0_cross_2"],
        params["delta_theta_star_cross_2"],
    )
    calculated_params["b_cross_2"] = b_cross_2
    calculated_params["delta_theta_cross_2_c"] = delta_theta_cross_2_c

    ## f4(theta_3)
    b_cross_3, delta_theta_cross_3_c = bsf.get_f4_smoothing_params(
        params["a_cross_3"], params["theta0_cross_3"], params["delta_theta_star_cross_3"]
    )

    calculated_params["b_cross_3"] = b_cross_3
    calculated_params["delta_theta_cross_3_c"] = delta_theta_cross_3_c

    ## f4(theta_4) + f4(pi - theta_4)
    b_cross_4, delta_theta_cross_4_c = bsf.get_f4_smoothing_params(
        params["a_cross_4"],
        params["theta0_cross_4"],
        params["delta_theta_star_cross_4"],
    )
    calculated_params["b_cross_4"] = b_cross_4
    calculated_params["delta_theta_cross_4_c"] = delta_theta_cross_4_c

    ## f4(theta_7) + f4(pi - theta_7)
    b_cross_7, delta_theta_cross_7_c = bsf.get_f4_smoothing_params(
        params["a_cross_7"],
        params["theta0_cross_7"],
        params["delta_theta_star_cross_7"],
    )
    calculated_params["b_cross_7"] = b_cross_7
    calculated_params["delta_theta_cross_7_c"] = delta_theta_cross_7_c

    ## f4(theta_8) + f4(pi - theta_8)
    b_cross_8, delta_theta_cross_8_c = bsf.get_f4_smoothing_params(
        params["a_cross_8"],
        params["theta0_cross_8"],
        params["delta_theta_star_cross_8"],
    )
    calculated_params["b_cross_8"] = b_cross_8
    calculated_params["delta_theta_cross_8_c"] = delta_theta_cross_8_c

    return calculated_params | params


CROSS_STACKING_DG = _init_cross_stacking(CROSS_STACKING_DG)


COAXIAL_STACKING_DG = {
    "dr_low_coax": 0.22,
    "dr_high_coax": 0.58,
    "k_coax": 46.0,
    "dr0_coax": 0.4,
    "dr_c_coax": 0.6,
    "theta0_coax_4": 0.0,
    "delta_theta_star_coax_4": 0.8,
    "a_coax_4": 1.30,
    "theta0_coax_1": np.pi - 0.60,
    "delta_theta_star_coax_1": 0.65,
    "a_coax_1": 2.00,
    "theta0_coax_5": 0.0,
    "delta_theta_star_coax_5": 0.95,
    "a_coax_5": 0.90,
    "theta0_coax_6": 0.0,
    "delta_theta_star_coax_6": 0.95,
    "a_coax_6": 0.90,
    "cos_phi3_star_coax": -0.65,
    "a_coax_3p": 2.0,
    "cos_phi4_star_coax": -0.65,
    "a_coax_4p": 2.0,
}


def _init_coaxial_stacking(params: dict[str, float]) -> dict[str, float]:
    calculated_params = {}

    ## f2(dr_coax)
    b_low_coax, dr_c_low_coax, b_high_coax, dr_c_high_coax = bsf.get_f2_smoothing_params(
        params["k_coax"], params["dr0_coax"], params["dr_c_coax"], params["dr_low_coax"], params["dr_high_coax"]
    )
    calculated_params["b_low_coax"] = b_low_coax
    calculated_params["b_high_coax"] = b_high_coax
    calculated_params["dr_c_low_coax"] = dr_c_low_coax
    calculated_params["dr_c_high_coax"] = dr_c_high_coax

    ## f4(theta_4)
    b_coax_4, delta_theta_coax_4_c = bsf.get_f4_smoothing_params(
        params["a_coax_4"], params["theta0_coax_4"], params["delta_theta_star_coax_4"]
    )
    calculated_params["b_coax_4"] = b_coax_4
    calculated_params["delta_theta_coax_4_c"] = delta_theta_coax_4_c

    ## f4(theta_1) + f4(2*pi - theta_1)
    b_coax_1, delta_theta_coax_1_c = bsf.get_f4_smoothing_params(
        params["a_coax_1"], params["theta0_coax_1"], params["delta_theta_star_coax_1"]
    )
    calculated_params["b_coax_1"] = b_coax_1
    calculated_params["delta_theta_coax_1_c"] = delta_theta_coax_1_c

    ## f4(theta_5) + f4(pi - theta_5)
    b_coax_5, delta_theta_coax_5_c = bsf.get_f4_smoothing_params(
        params["a_coax_5"],
        params["theta0_coax_5"],
        params["delta_theta_star_coax_5"],
    )
    calculated_params["b_coax_5"] = b_coax_5
    calculated_params["delta_theta_coax_5_c"] = delta_theta_coax_5_c

    ## f4(theta_6) + f4(pi - theta_6)
    b_coax_6, delta_theta_coax_6_c = bsf.get_f4_smoothing_params(
        params["a_coax_6"],
        params["theta0_coax_6"],
        params["delta_theta_star_coax_6"],
    )
    calculated_params["b_coax_6"] = b_coax_6
    calculated_params["delta_theta_coax_6_c"] = delta_theta_coax_6_c

    ## f5(cos(phi3))
    b_cos_phi3_coax, cos_phi3_c_coax = bsf.get_f5_smoothing_params(
        params["a_coax_3p"],
        params["cos_phi3_star_coax"],
    )
    calculated_params["b_cos_phi3_coax"] = b_cos_phi3_coax
    calculated_params["cos_phi3_c_coax"] = cos_phi3_c_coax

    ## f5(cos(phi4))
    b_cos_phi4_coax, cos_phi4_c_coax = bsf.get_f5_smoothing_params(params["a_coax_4p"], params["cos_phi4_star_coax"])
    calculated_params["b_cos_phi4_coax"] = b_cos_phi4_coax
    calculated_params["cos_phi4_c_coax"] = cos_phi4_c_coax

    return calculated_params | params


COAXIAL_STACKING_DG = _init_coaxial_stacking(COAXIAL_STACKING_DG)


HB_DG = {
    "dr_low_hb": 0.34,
    "dr_high_hb": 0.70,
    "eps_hb": 1.077,
    "a_hb": 8.0,
    "dr0_hb": 0.4,
    "dr_c_hb": 0.75,
    "theta0_hb_1": 0.0,
    "delta_theta_star_hb_1": 0.70,
    "a_hb_1": 1.50,
    "theta0_hb_2": 0.0,
    "delta_theta_star_hb_2": 0.70,
    "a_hb_2": 1.50,
    "theta0_hb_3": 0.0,
    "delta_theta_star_hb_3": 0.70,
    "a_hb_3": 1.50,
    "theta0_hb_4": np.pi,
    "delta_theta_star_hb_4": 0.70,
    "a_hb_4": 0.46,
    "theta0_hb_7": np.pi / 2,
    "delta_theta_star_hb_7": 0.45,
    "a_hb_7": 4.0,
    "theta0_hb_8": np.pi / 2,
    "delta_theta_star_hb_8": 0.45,
    "a_hb_8": 4.0,
}


def _init_hb(params: dict[str, float]) -> dict[str, float]:
    calculated_params = {}

    ## f1(dr_hb)
    b_low_hb, dr_c_low_hb, b_high_hb, dr_c_high_hb = bsf.get_f1_smoothing_params(
        params["eps_hb"],
        params["dr0_hb"],
        params["a_hb"],
        params["dr_c_hb"],
        params["dr_low_hb"],
        params["dr_high_hb"],
    )
    calculated_params["b_low_hb"] = b_low_hb
    calculated_params["b_high_hb"] = b_high_hb
    calculated_params["dr_c_low_hb"] = dr_c_low_hb
    calculated_params["dr_c_high_hb"] = dr_c_high_hb

    ## f4(theta_1)
    b_hb_1, delta_theta_hb_1_c = bsf.get_f4_smoothing_params(
        params["a_hb_1"],
        params["theta0_hb_1"],
        params["delta_theta_star_hb_1"],
    )

    calculated_params["b_hb_1"] = b_hb_1
    calculated_params["delta_theta_hb_1_c"] = delta_theta_hb_1_c

    ## f4(theta_2)
    b_hb_2, delta_theta_hb_2_c = bsf.get_f4_smoothing_params(
        params["a_hb_2"],
        params["theta0_hb_2"],
        params["delta_theta_star_hb_2"],
    )
    calculated_params["b_hb_2"] = b_hb_2
    calculated_params["delta_theta_hb_2_c"] = delta_theta_hb_2_c

    ## f4(theta_3)
    b_hb_3, delta_theta_hb_3_c = bsf.get_f4_smoothing_params(
        params["a_hb_3"],
        params["theta0_hb_3"],
        params["delta_theta_star_hb_3"],
    )
    calculated_params["b_hb_3"] = b_hb_3
    calculated_params["delta_theta_hb_3_c"] = delta_theta_hb_3_c

    ## f4(theta_4)
    b_hb_4, delta_theta_hb_4_c = bsf.get_f4_smoothing_params(
        params["a_hb_4"],
        params["theta0_hb_4"],
        params["delta_theta_star_hb_4"],
    )
    calculated_params["b_hb_4"] = b_hb_4
    calculated_params["delta_theta_hb_4_c"] = delta_theta_hb_4_c

    ## f4(theta_7)
    b_hb_7, delta_theta_hb_7_c = bsf.get_f4_smoothing_params(
        params["a_hb_7"],
        params["theta0_hb_7"],
        params["delta_theta_star_hb_7"],
    )
    calculated_params["b_hb_7"] = b_hb_7
    calculated_params["delta_theta_hb_7_c"] = delta_theta_hb_7_c

    ## f4(theta_8)
    b_hb_8, delta_theta_hb_8_c = bsf.get_f4_smoothing_params(
        params["a_hb_8"], params["theta0_hb_8"], params["delta_theta_star_hb_8"]
    )
    calculated_params["b_hb_8"] = b_hb_8
    calculated_params["delta_theta_hb_8_c"] = delta_theta_hb_8_c

    return calculated_params | params


HB_DG = _init_hb(HB_DG)
