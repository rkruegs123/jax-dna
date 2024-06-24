from copy import deepcopy
import pdb
import numpy as onp
from pathlib import Path

from jax_dna.rna2 import load_params, model
from jax_dna.common import utils


DEFAULT_VARIABLE_MAPPER = {
    "geometry": {
        "pos_back": "RNA_POS_BACK",
        "pos_stack": "RNA_POS_STACK",
        "pos_base": "RNA_POS_BASE",

        "pos_stack_3_a1": "RNA_POS_STACK_3_a1",
        "pos_stack_3_a2": "RNA_POS_STACK_3_a2",
        "pos_stack_5_a1": "RNA_POS_STACK_5_a1",
        "pos_stack_5_a2": "RNA_POS_STACK_5_a2",

        "pos_back_a1": "RNA_POS_BACK_a1",
        "pos_back_a3": "RNA_POS_BACK_a3",
        "pos_back_a2": "RNA_POS_BACK_a2",

        "p5_x": "p5_x",
        "p5_y": "p5_y",
        "p5_z": "p5_z",
        "p3_x": "p3_x",
        "p3_y": "p3_y",
        "p3_z": "p3_z"
    ],
    "fene": {
        "eps_backbone": "RNA_FENE_EPS",
        "delta_backbone": "RNA_FENE_DELTA",
        "r0_backbone": "RNA_FENE_R0"
    },
    "excluded_volume": {
        "eps_exc": "RNA_EXCL_EPS",

        "sigma_backbone": "RNA_EXCL_S1",
        "sigma_base": "RNA_EXCL_S2",
        "sigma_back_base": "RNA_EXCL_S3",
        "sigma_base_back": "RNA_EXCL_S4",

        "dr_star_backbone": "RNA_EXCL_R1",
        "dr_star_base": "RNA_EXCL_R2",
        "dr_star_back_base": "RNA_EXCL_R3",
        "dr_star_base_back": "RNA_EXCL_R4",

        "b_backbone": "RNA_EXCL_B1",
        "b_base": "RNA_EXCL_B2",
        "b_back_base": "RNA_EXCL_B3",
        "b_base_back": "RNA_EXCL_B4",

        "dr_c_backbone": "RNA_EXCL_RC1",
        "dr_c_base": "RNA_EXCL_RC2",
        "dr_c_back_base": "RNA_EXCL_RC3",
        "dr_c_base_back": "RNA_EXCL_RC4"
    },
    "stacking": {
        # f1(dr_stack)
        "eps_stack_base": "RNA_STCK_BASE_EPS",
        "eps_stack_kt_coeff": "RNA_STCK_FACT_EPS",
        "a_stack": "RNA_STCK_A",
        "dr0_stack": "RNA_STCK_R0",
        "dr_c_stack": "RNA_STCK_RC",
        "dr_low_stack": "RNA_STCK_RLOW",
        "dr_high_stack": "RNA_STCK_RHIGH",
        "b_low_stack": "RNA_STCK_BLOW",
        "b_high_stack": "RNA_STCK_BHIGH",
        "dr_c_low_stack": "RNA_STCK_RCLOW",
        "dr_c_high_stack": "RNA_STCK_RCHIGH",

        # f4(theta_5p)
        "a_stack_5": "RNA_STCK_THETA5_A",
        "theta0_stack_5": "RNA_STCK_THETA5_T0",
        "delta_theta_star_stack_5": "RNA_STCK_THETA5_TS",
        "b_stack_5": "RNA_STCK_THETA5_B",
        "delta_theta_stack_5_c": "RNA_STCK_THETA5_TC",

        # f4(theta_6p)
        "a_stack_6": "RNA_STCK_THETA6_A",
        "theta0_stack_6": "RNA_STCK_THETA6_T0",
        "delta_theta_star_stack_6": "RNA_STCK_THETA6_TS",
        "b_stack_6": "RNA_STCK_THETA6_B",
        "delta_theta_stack_6_c": "RNA_STCK_THETA6_TC",

        # f4(theta_9)
        "a_stack_9": "STCK_THETAB1_A",
        "theta0_stack_9": "STCK_THETAB1_T0",
        "delta_theta_star_stack_9": "STCK_THETAB1_TS",
        "b_stack_9": "STCK_THETAB1_B",
        "delta_theta_stack_9_c": "STCK_THETAB1_TC",

        # f4(theta_10)
        "a_stack_10": "STCK_THETAB2_A",
        "theta0_stack_10": "STCK_THETAB2_T0",
        "delta_theta_star_stack_10": "STCK_THETAB2_TS",
        "b_stack_10": "STCK_THETAB2_B",
        "delta_theta_stack_10_c": "STCK_THETAB2_TC",

        # f5(-cos(phi1))
        "a_stack_1": "RNA_STCK_PHI1_A",
        "neg_cos_phi1_star_stack": "RNA_STCK_PHI1_XS",
        "b_neg_cos_phi1_stack": "RNA_STCK_PHI1_B",
        "neg_cos_phi1_c_stack": "RNA_STCK_PHI1_XC",

        # f5(-cos(phi2))
        "a_stack_2": "RNA_STCK_PHI2_A",
        "neg_cos_phi2_star_stack": "RNA_STCK_PHI2_XS",
        "b_neg_cos_phi2_stack": "RNA_STCK_PHI2_B",
        "neg_cos_phi2_c_stack": "RNA_STCK_PHI2_XC"
    },
    "hydrogen_bonding": {

        # f1(dr_hb)
        "eps_hb": "RNA_HYDR_EPS",
        "a_hb": "RNA_HYDR_A",
        "dr0_hb": "RNA_HYDR_R0",
        "dr_c_hb": "RNA_HYDR_RC",
        "dr_low_hb": "RNA_HYDR_RLOW",
        "dr_high_hb": "RNA_HYDR_RHIGH",
        "b_low_hb": "RNA_HYDR_BLOW",
        "dr_c_low_hb": "RNA_HYDR_RCLOW",
        "b_high_hb": "RNA_HYDR_BHIGH",
        "dr_c_high_hb": "RNA_HYDR_RCHIGH",

        # f4(theta_1)
        "a_hb_1": "RNA_HYDR_THETA1_A",
        "theta0_hb_1": "RNA_HYDR_THETA1_T0",
        "delta_theta_star_hb_1": "RNA_HYDR_THETA1_TS",
        "b_hb_1": "RNA_HYDR_THETA1_B",
        "delta_theta_hb_1_c": "RNA_HYDR_THETA1_TC",

        # f4(theta_2)
        "a_hb_2": "RNA_HYDR_THETA2_A",
        "theta0_hb_2": "RNA_HYDR_THETA2_T0",
        "delta_theta_star_hb_2": "RNA_HYDR_THETA2_TS",
        "b_hb_2": "RNA_HYDR_THETA2_B",
        "delta_theta_hb_2_c": "RNA_HYDR_THETA2_TC",

        # f4(theta_3)
        "a_hb_3": "RNA_HYDR_THETA3_A",
        "theta0_hb_3": "RNA_HYDR_THETA3_T0",
        "delta_theta_star_hb_3": "RNA_HYDR_THETA3_TS",
        "b_hb_3": "RNA_HYDR_THETA3_B",
        "delta_theta_hb_3_c": "RNA_HYDR_THETA3_TC",

        # f4(theta_4)
        "a_hb_4": "RNA_HYDR_THETA4_A",
        "theta0_hb_4": "RNA_HYDR_THETA4_T0",
        "delta_theta_star_hb_4": "RNA_HYDR_THETA4_TS",
        "b_hb_4": "RNA_HYDR_THETA4_B",
        "delta_theta_hb_4_c": "RNA_HYDR_THETA4_TC",

        # f4(theta_7)
        "a_hb_7": "RNA_HYDR_THETA7_A",
        "theta0_hb_7": "RNA_HYDR_THETA7_T0",
        "delta_theta_star_hb_7": "RNA_HYDR_THETA7_TS",
        "b_hb_7": "RNA_HYDR_THETA7_B",
        "delta_theta_hb_7_c": "RNA_HYDR_THETA7_TC",

        # f4(theta_8)
        "a_hb_8": "RNA_HYDR_THETA8_A",
        "theta0_hb_8": "RNA_HYDR_THETA8_T0",
        "delta_theta_star_hb_8": "RNA_HYDR_THETA8_TS",
        "b_hb_8": "RNA_HYDR_THETA8_B",
        "delta_theta_hb_8_c": "RNA_HYDR_THETA8_TC"
    },
    "cross_stacking": {
        # f2(dr_cross)
        "k_cross": "RNA_CRST_K",
        "r0_cross": "RNA_CRST_R0",
        "dr_c_cross": "RNA_CRST_RC",
        "dr_low_cross": "RNA_CRST_RLOW",
        "dr_high_cross": "RNA_CRST_RHIGH",
        "b_low_cross": "RNA_CRST_BLOW",
        "dr_c_low_cross": "RNA_CRST_RCLOW",
        "b_high_cross": "RNA_CRST_BHIGH",
        "dr_c_high_cross": "RNA_CRST_RCHIGH",

        # f4(theta_1)
        "a_cross_1": "RNA_CRST_THETA1_A",
        "theta0_cross_1": "RNA_CRST_THETA1_T0",
        "delta_theta_star_cross_1": "RNA_CRST_THETA1_TS",
        "b_cross_1": "RNA_THETA1_B",
        "delta_theta_cross_1_c": "RNA_CRST_THETA1_TC",

        # f4(theta_2)
        "a_cross_2": "RNA_CRST_THETA2_A",
        "theta0_cross_2": "RNA_CRST_THETA2_T0",
        "delta_theta_star_cross_2": "RNA_CRST_THETA2_TS",
        "b_cross_2": "RNA_THETA2_B",
        "delta_theta_cross_2_c": "RNA_CRST_THETA2_TC",

        # f4(theta_3)
        "a_cross_3": "RNA_CRST_THETA3_A",
        "theta0_cross_3": "RNA_CRST_THETA3_T0",
        "delta_theta_star_cross_3": "RNA_CRST_THETA3_TS",
        "b_cross_3": "RNA_THETA3_B",
        "delta_theta_cross_3_c": "RNA_CRST_THETA3_TC",

        # f4(theta_7) + f4(pi - theta_7)
        "a_cross_7": "RNA_CRST_THETA7_A",
        "theta0_cross_7": "RNA_CRST_THETA7_T0",
        "delta_theta_star_cross_7": "RNA_CRST_THETA7_TS",
        "b_cross_7": "RNA_THETA7_B",
        "delta_theta_cross_7_c": "RNA_CRST_THETA7_TC",

        # f4(theta_8) + f4(pi - theta_8)
        "a_cross_8": "RNA_CRST_THETA8_A",
        "theta0_cross_8": "RNA_CRST_THETA8_T0",
        "delta_theta_star_cross_8": "RNA_CRST_THETA8_TS",
        "b_cross_8": "RNA_THETA8_B",
        "delta_theta_cross_8_c": "RNA_CRST_THETA8_TC"
    },
    "coaxial_stacking": {
        # f2(dr_coax)
        "k_coax": "RNA_CXST_K_OXDNA",
        "dr0_coax": "RNA_CXST_R0",
        "dr_c_coax": "RNA_CXST_RC",
        "dr_low_coax": "RNA_CXST_RLOW",
        "dr_high_coax": "RNA_CXST_RHIGH",
        "b_low_coax": "RNA_CXST_BLOW",
        "dr_c_low_coax": "RNA_CXST_RCLOW",
        "b_high_coax": "RNA_CXST_BHIGH",
        "dr_c_high_coax": "RNA_CXST_RCHIGH",

        # f4(theta_1) + f4(2*pi - theta_1)
        "a_coax_1": "RNA_CXST_THETA1_A",
        "theta0_coax_1": "RNA_CXST_THETA1_T0_OXDNA",
        "delta_theta_star_coax_1": "RNA_CXST_THETA1_TS",
        "b_coax_1": "RNA_CXST_THETA1_B",
        "delta_theta_coax_1_c": "RNA_CXST_THETA1_TC",

        # f4(theta_4)
        "a_coax_4": "RNA_CXST_THETA4_A",
        "theta0_coax_4": "RNA_CXST_THETA4_T0",
        "delta_theta_star_coax_4": "RNA_CXST_THETA4_TS",
        "b_coax_4": "RNA_CXST_THETA4_B",
        "delta_theta_coax_4_c": "RNA_CXST_THETA4_TC",

        # f4(theta_5) + f4(pi - theta_5)
        "a_coax_5": "RNA_CXST_THETA5_A",
        "theta0_coax_5": "RNA_CXST_THETA5_T0",
        "delta_theta_star_coax_5": "RNA_CXST_THETA5_TS",
        "b_coax_5": "RNA_CXST_THETA5_B",
        "delta_theta_coax_5_c": "RNA_CXST_THETA5_TC",

        # f4(theta_6) + f4(pi - theta_6)
        "a_coax_6": "RNA_CXST_THETA6_A",
        "theta0_coax_6": "RNA_CXST_THETA6_T0",
        "delta_theta_star_coax_6": "RNA_CXST_THETA6_TS",
        "b_coax_6": "RNA_CXST_THETA6_B",
        "delta_theta_coax_6_c": "RNA_CXST_THETA6_TC",

        # f5(cos(phi3))
        "a_coax_3p": "RNA_CXST_PHI3_A",
        "cos_phi3_star_coax": "RNA_CXST_PHI3_XS",
        "b_cos_phi3_coax": "RNA_CXST_PHI3_B",
        "cos_phi3_c_coax": "RNA_CXST_PHI3_XC",

        # f5(cos(phi4))
        "a_coax_4p": "RNA_CXST_PHI4_A",
        "cos_phi4_star_coax": "RNA_CXST_PHI4_XS",
        "b_cos_phi4_coax": "RNA_CXST_PHI4_B",
        "cos_phi4_c_coax": "RNA_CXST_PHI4_XC"
    }
}


def write_external_model(override_base_params, t_kelvin, fpath, variable_mapper=DEFAULT_VARIABLE_MAPPER):
    base_params = model.get_full_base_params(override_base_params)
    base_params_to_process = deepcopy(base_params) # they are processed in-place
    params = load_params._process(base_params_to_process, t_kelvin)

    ## Add back in eps_stack_base and eps_stack_kt_coeff
    del params['stacking']['eps_stack']
    params['stacking']['eps_stack_base'] = base_params['stacking']['eps_stack_base']
    params['stacking']['eps_stack_kt_coeff'] = base_params['stacking']['eps_stack_kt_coeff']

    ## Remove excluded_volume_bonded
    del params['excluded_volume_bonded']


    # Construct dictionary that maps oxRNA variable name to our internal naming
    new_oxrna_vars = dict()
    for t, t_params in params.items():
        mapper = variable_mapper[t]

        for k, v in t_params.items():
            oxrna_var_name = mapper[k]
            new_oxrna_vars[oxrna_var_name] = v
            if k == "delta_backbone":
                new_oxrna_vars["RNA_FENE_DELTA2"] = v**2

    # Construct a new set of lines for external_model.txt
    external_lines = list()
    for key, val in new_oxdna_vars.items():
        line = f"{key} = {val}"
        external_lines.append(line)

    with open(fpath, "w") as of:
        of.writelines(external_lines)

    return


def write_seq_specific(fpath, base_params, hb_mult, stack_mult, cross_mult):
    lines = list()

    # Write HB lines

    ## note: we assume HB symmetry for now
    default_f1_eps_hb = base_params["hydrogen_bonding"]["eps_hb"]

    au_mult = hb_mult[utils.RNA_ALPHA.index("A")][utils.RNA_ALPHA.index("U")]
    ua_mult = hb_mult[utils.RNA_ALPHA.index("U")][utils.RNA_ALPHA.index("A")]
    assert(au_mult == ua_mult)

    gc_mult = hb_mult[utils.RNA_ALPHA.index("G")][utils.RNA_ALPHA.index("C")]
    cg_mult = hb_mult[utils.RNA_ALPHA.index("C")][utils.RNA_ALPHA.index("G")]
    assert(gc_mult == cg_mult)

    gu_mult = hb_mult[utils.RNA_ALPHA.index("G")][utils.RNA_ALPHA.index("U")]
    ug_mult = hb_mult[utils.RNA_ALPHA.index("U")][utils.RNA_ALPHA.index("G")]
    assert(gu_mult == ug_mult)

    lines.append(f"HYDR_A_T = {au_mult * default_f1_eps_hb}")
    lines.append(f"HYDR_C_G = {cg_mult * default_f1_eps_hb}")
    lines.append(f"HYDR_G_T = {gu_mult * default_f1_eps_hb}")


    # Write cross stacking lines
    default_cross_k = base_params["cross_stacking"]["k_cross"]

    for nuc1 in utils.RNA_ALPHA:
        for nuc2 in utils.RNA_ALPHA:
            nuc1_repr = nuc1 if nuc1 != "U" else "T"
            nuc2_repr = nuc2 if nuc2 != "U" else "T"

            prefactor = cross_mult[RNA_ALPHA.index(nuc1), RNA_ALPHA.index(nuc2)]
            ss_cross_k = prefactor * default_cross_k

            lines.append(f"CROSS_{nuc1_repr}_{nuc2_repr} = {ss_cross_k}")

    # Write stacking lines
    eps_stack_base = base_params["stacking"]["eps_stack_base"]
    eps_stack_kt_coeff = base_params["stacking"]["eps_stack_kt_coeff"]

    for nuc1 in utils.RNA_ALPHA:
    for nuc2 in utils.RNA_ALPHA:
        nuc1_repr = nuc1 if nuc1 != "U" else "T"
        nuc2_repr = nuc2 if nuc2 != "U" else "T"

        prefactor = stack_mult[RNA_ALPHA.index(nuc1), RNA_ALPHA.index(nuc2)]
        ss_eps_stack_base = prefactor * eps_stack_base

        lines.append(f"STCK_{nuc1_repr}_{nuc2_repr} = {ss_eps_stack_base}")

    eps_stack_prime = eps_stack_kt_coeff / eps_stack_base
    lines.append(f"ST_T_DEP = {eps_stack_prime}")

    # Write to path
    fpath = Path(fpath)
    assert(not fpath.exists())
    with open(fpath, "w") as f:
        for line in lines:
            f.write(f"{line}\n")

    return



if __name__ == "__main__":
    pass
