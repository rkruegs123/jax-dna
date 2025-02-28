# ruff: noqa
# fmt: off
from copy import deepcopy
import pdb
from pathlib import Path
import subprocess
import shutil
import pandas as pd
import numpy as onp


from jax import jit, vmap
from jax_md import space
import jax.numpy as jnp

from jax_dna.common.utils import DEFAULT_TEMP, get_kt, get_one_hot, tree_stack
from jax_dna.dna1 import model, load_params
from jax_dna.common import topology, trajectory


MODEL_TEMPLATE_PATH = Path("data/templates/model_template.h")

DEFAULT_VARIABLE_MAPPER = {
    "fene": {
        "eps_backbone": "FENE_EPS",
        "delta_backbone": "FENE_DELTA",
        "r0_backbone": "FENE_R0_OXDNA"
    },
    # FIXME: check that 1-4 are as we expect
    "excluded_volume": {
        "eps_exc": "EXCL_EPS",

        "sigma_backbone": "EXCL_S1",
        "sigma_base": "EXCL_S2",
        "sigma_back_base": "EXCL_S3",
        "sigma_base_back": "EXCL_S4",

        "dr_star_backbone": "EXCL_R1",
        "dr_star_base": "EXCL_R2",
        "dr_star_back_base": "EXCL_R3",
        "dr_star_base_back": "EXCL_R4",

        "b_backbone": "EXCL_B1",
        "b_base": "EXCL_B2",
        "b_back_base": "EXCL_B3",
        "b_base_back": "EXCL_B4",

        "dr_c_backbone": "EXCL_RC1",
        "dr_c_base": "EXCL_RC2",
        "dr_c_back_base": "EXCL_RC3",
        "dr_c_base_back": "EXCL_RC4"
    },
    "stacking": {
        # f1(dr_stack)
        # FIXME: what to do with STCK_F1?
        "eps_stack_base": "STCK_BASE_EPS_OXDNA",
        "eps_stack_kt_coeff": "STCK_FACT_EPS_OXDNA",
        "a_stack": "STCK_A",
        "dr0_stack": "STCK_R0",
        "dr_c_stack": "STCK_RC",
        "dr_low_stack": "STCK_RLOW",
        "dr_high_stack": "STCK_RHIGH",
        "b_low_stack": "STCK_BLOW",
        "b_high_stack": "STCK_BHIGH",
        "dr_c_low_stack": "STCK_RCLOW",
        "dr_c_high_stack": "STCK_RCHIGH",

        # f4(theta_4)
        # FIXME: what to do about STCK_F4_THETA4
        "a_stack_4": "STCK_THETA4_A",
        "theta0_stack_4": "STCK_THETA4_T0",
        "delta_theta_star_stack_4": "STCK_THETA4_TS",
        "b_stack_4": "STCK_THETA4_B",
        "delta_theta_stack_4_c": "STCK_THETA4_TC",

        # f4(theta_5p)
        # FIXME: what to do about STCK_F5_THETA5
        "a_stack_5": "STCK_THETA5_A",
        "theta0_stack_5": "STCK_THETA5_T0",
        "delta_theta_star_stack_5": "STCK_THETA5_TS",
        "b_stack_5": "STCK_THETA5_B",
        "delta_theta_stack_5_c": "STCK_THETA5_TC",

        # f4(theta_6p)
        # FIXME: what to do about STCK_F6_THETA6
        "a_stack_6": "STCK_THETA6_A",
        "theta0_stack_6": "STCK_THETA6_T0",
        "delta_theta_star_stack_6": "STCK_THETA6_TS",
        "b_stack_6": "STCK_THETA6_B",
        "delta_theta_stack_6_c": "STCK_THETA6_TC",

        # f5(-cos(phi1))
        # FIXME: what to do about STCK_F5_PHI1
        "a_stack_1": "STCK_PHI1_A",
        "neg_cos_phi1_star_stack": "STCK_PHI1_XS",
        "b_neg_cos_phi1_stack": "STCK_PHI1_B",
        "neg_cos_phi1_c_stack": "STCK_PHI1_XC",

        # f5(-cos(phi2))
        # FIXME: what to do about STCK_F5_PHI2
        "a_stack_2": "STCK_PHI2_A",
        "neg_cos_phi2_star_stack": "STCK_PHI2_XS",
        "b_neg_cos_phi2_stack": "STCK_PHI2_B",
        "neg_cos_phi2_c_stack": "STCK_PHI2_XC"
    },
    "hydrogen_bonding": {

        # f1(dr_hb)
        # FIXME: what is HYDR_F1
        "eps_hb": "HYDR_EPS_OXDNA",
        "a_hb": "HYDR_A",
        "dr0_hb": "HYDR_R0",
        "dr_c_hb": "HYDR_RC",
        "dr_low_hb": "HYDR_RLOW",
        "dr_high_hb": "HYDR_RHIGH",
        "b_low_hb": "HYDR_BLOW",
        "dr_c_low_hb": "HYDR_RCLOW",
        "b_high_hb": "HYDR_BHIGH",
        "dr_c_high_hb": "HYDR_RCHIGH",

        # f4(theta_1)
        # FIXME: what is HYDR_F4_THETA1
        "a_hb_1": "HYDR_THETA1_A",
        "theta0_hb_1": "HYDR_THETA1_T0",
        "delta_theta_star_hb_1": "HYDR_THETA1_TS",
        "b_hb_1": "HYDR_THETA1_B",
        "delta_theta_hb_1_c": "HYDR_THETA1_TC",

        # f4(theta_2)
        # FIXME: what is HYDR_F4_THETA2
        "a_hb_2": "HYDR_THETA2_A",
        "theta0_hb_2": "HYDR_THETA2_T0",
        "delta_theta_star_hb_2": "HYDR_THETA2_TS",
        "b_hb_2": "HYDR_THETA2_B",
        "delta_theta_hb_2_c": "HYDR_THETA2_TC",

        # f4(theta_3)
        # FIXME: what is HYDR_F4_THETA3
        "a_hb_3": "HYDR_THETA3_A",
        "theta0_hb_3": "HYDR_THETA3_T0",
        "delta_theta_star_hb_3": "HYDR_THETA3_TS",
        "b_hb_3": "HYDR_THETA3_B",
        "delta_theta_hb_3_c": "HYDR_THETA3_TC",

        # f4(theta_4)
        # FIXME: what is HYDR_F4_THETA4
        "a_hb_4": "HYDR_THETA4_A",
        "theta0_hb_4": "HYDR_THETA4_T0",
        "delta_theta_star_hb_4": "HYDR_THETA4_TS",
        "b_hb_4": "HYDR_THETA4_B",
        "delta_theta_hb_4_c": "HYDR_THETA4_TC",

        # f4(theta_7)
        # FIXME: what is HYDR_F4_THETA7
        "a_hb_7": "HYDR_THETA7_A",
        "theta0_hb_7": "HYDR_THETA7_T0",
        "delta_theta_star_hb_7": "HYDR_THETA7_TS",
        "b_hb_7": "HYDR_THETA7_B",
        "delta_theta_hb_7_c": "HYDR_THETA7_TC",

        # f4(theta_8)
        # FIXME: what is HYDR_F4_THETA8
        "a_hb_8": "HYDR_THETA8_A",
        "theta0_hb_8": "HYDR_THETA8_T0",
        "delta_theta_star_hb_8": "HYDR_THETA8_TS",
        "b_hb_8": "HYDR_THETA8_B",
        "delta_theta_hb_8_c": "HYDR_THETA8_TC"
    },
    "cross_stacking": {
        # f2(dr_cross)
        # FIXME: what is CRST_F2
        "k_cross": "CRST_K",
        "r0_cross": "CRST_R0",
        "dr_c_cross": "CRST_RC",
        "dr_low_cross": "CRST_RLOW",
        "dr_high_cross": "CRST_RHIGH",
        "b_low_cross": "CRST_BLOW",
        "dr_c_low_cross": "CRST_RCLOW",
        "b_high_cross": "CRST_BHIGH",
        "dr_c_high_cross": "CRST_RCHIGH",

        # f4(theta_1)
        # FIXME: CRST_F4_THETA1
        "a_cross_1": "CRST_THETA1_A",
        "theta0_cross_1": "CRST_THETA1_T0", # FIXME: pi - 2.35
        "delta_theta_star_cross_1": "CRST_THETA1_TS",
        "b_cross_1": "CRST_THETA1_B",
        "delta_theta_cross_1_c": "CRST_THETA1_TC",

        # f4(theta_2)
        # FIXME: CRST_F4_THETA2
        "a_cross_2": "CRST_THETA2_A",
        "theta0_cross_2": "CRST_THETA2_T0", # FIXME: pi - 2.35
        "delta_theta_star_cross_2": "CRST_THETA2_TS",
        "b_cross_2": "CRST_THETA2_B",
        "delta_theta_cross_2_c": "CRST_THETA2_TC",

        # f4(theta_3)
        # FIXME: CRST_F4_THETA3
        "a_cross_3": "CRST_THETA3_A",
        "theta0_cross_3": "CRST_THETA3_T0", # FIXME: pi - 2.35
        "delta_theta_star_cross_3": "CRST_THETA3_TS",
        "b_cross_3": "CRST_THETA3_B",
        "delta_theta_cross_3_c": "CRST_THETA3_TC",

        # f4(theta_4) + f4(pi - theta_4)
        # FIXME: CRST_F4_THETA4
        "a_cross_4": "CRST_THETA4_A",
        "theta0_cross_4": "CRST_THETA4_T0", # FIXME: pi - 2.35
        "delta_theta_star_cross_4": "CRST_THETA4_TS",
        "b_cross_4": "CRST_THETA4_B",
        "delta_theta_cross_4_c": "CRST_THETA4_TC",

        # f4(theta_7) + f4(pi - theta_7)
        # FIXME: CRST_F4_THETA7
        "a_cross_7": "CRST_THETA7_A",
        "theta0_cross_7": "CRST_THETA7_T0", # FIXME: pi - 2.35
        "delta_theta_star_cross_7": "CRST_THETA7_TS",
        "b_cross_7": "CRST_THETA7_B",
        "delta_theta_cross_7_c": "CRST_THETA7_TC",

        # f4(theta_8) + f4(pi - theta_8)
        # FIXME: CRST_F4_THETA8
        "a_cross_8": "CRST_THETA8_A",
        "theta0_cross_8": "CRST_THETA8_T0", # FIXME: pi - 2.35
        "delta_theta_star_cross_8": "CRST_THETA8_TS",
        "b_cross_8": "CRST_THETA8_B",
        "delta_theta_cross_8_c": "CRST_THETA8_TC"
    },
    "coaxial_stacking": {
        # f2(dr_coax)
        # FIXME: what is CXST_F2
        "k_coax": "CXST_K_OXDNA",
        "dr0_coax": "CXST_R0",
        "dr_c_coax": "CXST_RC",
        "dr_low_coax": "CXST_RLOW",
        "dr_high_coax": "CXST_RHIGH",
        "b_low_coax": "CXST_BLOW",
        "dr_c_low_coax": "CXST_RCLOW",
        "b_high_coax": "CXST_BHIGH",
        "dr_c_high_coax": "CXST_RCHIGH",

        # f4(theta_1) + f4(2*pi - theta_1)
        # FIXME: what is CXST_F4_THETA1
        # FIXME what are CXST_THETA1_SA and CXST_THETA1_SB
        "a_coax_1": "CXST_THETA1_A",
        "theta0_coax_1": "CXST_THETA1_T0_OXDNA",
        "delta_theta_star_coax_1": "CXST_THETA1_TS",
        "b_coax_1": "CXST_THETA1_B",
        "delta_theta_coax_1_c": "CXST_THETA1_TC",

        # f4(theta_4)
        # FIXME: what is CXST_F4_THETA4
        "a_coax_4": "CXST_THETA4_A",
        "theta0_coax_4": "CXST_THETA4_T0",
        "delta_theta_star_coax_4": "CXST_THETA4_TS",
        "b_coax_4": "CXST_THETA4_B",
        "delta_theta_coax_4_c": "CXST_THETA4_TC",

        # f4(theta_5) + f4(pi - theta_5)
        # FIXME: what is CXST_F4_THETA5
        "a_coax_5": "CXST_THETA5_A",
        "theta0_coax_5": "CXST_THETA5_T0",
        "delta_theta_star_coax_5": "CXST_THETA5_TS",
        "b_coax_5": "CXST_THETA5_B",
        "delta_theta_coax_5_c": "CXST_THETA5_TC",

        # f4(theta_6) + f4(pi - theta_6)
        # FIXME: what is CXST_F4_THETA6
        "a_coax_6": "CXST_THETA6_A",
        "theta0_coax_6": "CXST_THETA6_T0",
        "delta_theta_star_coax_6": "CXST_THETA6_TS",
        "b_coax_6": "CXST_THETA6_B",
        "delta_theta_coax_6_c": "CXST_THETA6_TC",

        # f5(cos(phi3))
        # FIXME: what is CXST_F5_PHI3
        "a_coax_3p": "CXST_PHI3_A",
        "cos_phi3_star_coax": "CXST_PHI3_XS",
        "b_cos_phi3_coax": "CXST_PHI3_B",
        "cos_phi3_c_coax": "CXST_PHI3_XC",

        # f5(cos(phi4))
        # FIXME: what is CXST_F5_PHI4
        "a_coax_4p": "CXST_PHI4_A",
        "cos_phi4_star_coax": "CXST_PHI4_XS",
        "b_cos_phi4_coax": "CXST_PHI4_B",
        "cos_phi4_c_coax": "CXST_PHI4_XC"
    }
}




def recompile_oxdna(override_base_params, oxdna_path, t_kelvin, num_threads=4, variable_mapper=DEFAULT_VARIABLE_MAPPER):
    if not oxdna_path.exists():
        raise RuntimeError(f"No oxDNA package at path: {oxdna_path}")

    # Setup the oxDNA build directory if it doesn't exist
    build_dir = oxdna_path / "build"
    if not build_dir.exists():
        build_dir.mkdir(parents=False, exist_ok=False)

        process = subprocess.Popen(["cmake", ".."], cwd=build_dir)
        p_results = process.communicate()
        rc = process.returncode
        if rc != 0:
            raise RuntimeError(f"Build setup failed with error code: {rc}")

    # Process the parameters and add/remove necessary keys
    base_params = model.get_full_base_params(override_base_params)
    kt = get_kt(t_kelvin)
    base_params_to_process = deepcopy(base_params) # they are processed in-place
    params = load_params._process(base_params_to_process, t_kelvin)

    ## Add back in eps_stack_base and eps_stack_kt_coeff
    del params['stacking']['eps_stack']
    params['stacking']['eps_stack_base'] = base_params['stacking']['eps_stack_base']
    params['stacking']['eps_stack_kt_coeff'] = base_params['stacking']['eps_stack_kt_coeff']

    ## Remove excluded_volume_bonded
    del params['excluded_volume_bonded']


    # Construct dictionary that maps oxDNA variable name to our internal naming

    ## FIXME: should we round?
    new_oxdna_vars = dict()
    for t, t_params in params.items():
        mapper = variable_mapper[t]

        for k, v in t_params.items():
            oxdna_var_name = mapper[k]
            new_oxdna_vars[oxdna_var_name] = v
            if k == "delta_backbone":
                new_oxdna_vars["FENE_DELTA2"] = v**2

    # Construct a new set of lines for model.h
    with open(MODEL_TEMPLATE_PATH, "r") as f:
        model_template_lines = f.readlines()

    model_lines = list()
    not_written = list()
    for mt_line in model_template_lines:
        if mt_line[:7] != "#define":
            model_lines.append(mt_line)
            continue

        var_name = mt_line.split()[1]
        if var_name in new_oxdna_vars:
            new_var = new_oxdna_vars[var_name]
            new_line = f"#define {var_name} {new_var}f\n"
            model_lines.append(new_line)
        else:
            not_written.append(var_name)
            model_lines.append(mt_line)


    # Write the new model.h
    model_path = oxdna_path / "src/model.h"
    with open(model_path, "w") as of:
        of.writelines(model_lines)

    # make oxDNA
    process = subprocess.Popen(["make", f"-j{num_threads}"], cwd=build_dir)
    make_results = process.communicate()
    rc = process.returncode
    if rc != 0:
        raise RuntimeError(f"Make failed with error code: {rc}")

    return



def rewrite_input_file(template_path, output_dir,
                       temp=None, steps=None,
                       init_conf_path=None, top_path=None,
                       save_interval=None, seed=None,
                       equilibration_steps=None, dt=None,
                       no_stdout_energy=None, backend=None,
                       cuda_device=None, cuda_list=None,
                       log_file=None,
                       extrapolate_hist=None, # Pre-formatted string
                       weights_file=None, op_file=None,
                       external_forces_file=None,
                       restart_step_counter=None,
                       interaction_type=None,
                       external_model=None,
                       seq_dep_file=None,
                       seq_dep_file_RNA=None,
                       observables_str=None,
                       salt_concentration=None,
                       list_type=None
):
    with open(template_path, "r") as f:
        input_template_lines = f.readlines()

    input_lines = list()

    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise RuntimeError(f"Invalid path: {output_dir}")

    traj_path = str(output_dir / "output.dat")
    energy_path = str(output_dir / "energy.dat")
    last_conf_path = str(output_dir / "last_conf.dat")
    last_hist_path = str(output_dir / "last_hist.dat")
    traj_hist_path = str(output_dir / "traj_hist.dat")
    output_path = str(output_dir / "input")

    def gen_new_line(tokens, new_val, new_val_type):
        assert(len(tokens) == 3)
        assert(isinstance(new_val, new_val_type))
        tokens[2] = f"{new_val}"
        new_line = ' '.join(tokens)
        return new_line


    for it_line in input_template_lines:
        tokens = it_line.split()
        if not tokens:
            input_lines.append(it_line)
        elif tokens[0] == "trajectory_file":
            new_line = gen_new_line(tokens, traj_path, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "energy_file":
            new_line = gen_new_line(tokens, energy_path, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "lastconf_file":
            new_line = gen_new_line(tokens, last_conf_path, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "T" and temp is not None:
            new_line = gen_new_line(tokens, temp, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "steps" and steps is not None:
            new_line = gen_new_line(tokens, steps, int)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "conf_file" and init_conf_path is not None:
            new_line = gen_new_line(tokens, init_conf_path, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "topology" and top_path is not None:
            new_line = gen_new_line(tokens, top_path, str)
            input_lines.append(f"{new_line}\n")
        elif (tokens[0] == "print_conf_interval" or tokens[0] == "print_energy_every") \
             and save_interval is not None:
            new_line = gen_new_line(tokens, save_interval, int)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "seed" and seed is not None:
            new_line = gen_new_line(tokens, seed, int)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "equilibration_steps" and equilibration_steps is not None:
            new_line = gen_new_line(tokens, equilibration_steps, int)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "dt" and dt is not None:
            new_line = gen_new_line(tokens, dt, float)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "no_stdout_energy" and no_stdout_energy is not None:
            new_line = gen_new_line(tokens, no_stdout_energy, int)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "backend" and backend is not None:
            new_line = gen_new_line(tokens, backend, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "CUDA_device" and cuda_device is not None:
            new_line = gen_new_line(tokens, cuda_device, int)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "CUDA_list" and cuda_list is not None:
            new_line = gen_new_line(tokens, cuda_list, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "log_file" and log_file is not None:
            new_line = gen_new_line(tokens, log_file, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "extrapolate_hist" and extrapolate_hist is not None:
            new_tokens = [tokens[0], tokens[1], extrapolate_hist]
            new_line = ' '.join(new_tokens)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "weights_file" and weights_file is not None:
            new_line = gen_new_line(tokens, weights_file, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "op_file" and op_file is not None:
            new_line = gen_new_line(tokens, op_file, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "last_hist_file":
            new_line = gen_new_line(tokens, last_hist_path, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "traj_hist_file":
            new_line = gen_new_line(tokens, traj_hist_path, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "external_forces_file":
            new_line = gen_new_line(tokens, external_forces_file, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "restart_step_counter" and restart_step_counter is not None:
            new_line = gen_new_line(tokens, restart_step_counter, int)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "interaction_type" and interaction_type is not None:
            new_line = gen_new_line(tokens, interaction_type, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "external_model" and external_model is not None:
            new_line = gen_new_line(tokens, external_model, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "seq_dep_file" and seq_dep_file is not None:
            new_line = gen_new_line(tokens, seq_dep_file, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "seq_dep_file_RNA" and seq_dep_file_RNA is not None:
            new_line = gen_new_line(tokens, seq_dep_file_RNA, str)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "salt_concentration" and salt_concentration is not None:
            new_line = gen_new_line(tokens, salt_concentration, float)
            input_lines.append(f"{new_line}\n")
        elif tokens[0] == "list_type" and list_type is not None:
            new_line = gen_new_line(tokens, list_type, str)
            input_lines.append(f"{new_line}\n")
        else:
            input_lines.append(it_line)

    with open(output_path, "w") as of:
        of.writelines(input_lines)

    if observables_str is not None:
        with open(output_path, "a") as of:
            of.writelines(observables_str)

    return

if __name__ == "__main__":

    oxdna_path = Path("/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/")
    t_kelvin = DEFAULT_TEMP

    override_base_params = deepcopy(model.EMPTY_BASE_PARAMS)
    # override_base_params["fene"] = model.DEFAULT_BASE_PARAMS["fene"]
    override_base_params["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]


    override_base_params['stacking']['a_stack_5'] = 0.3
    override_base_params['stacking']['a_stack_6'] = 0.2

    """
    override_base_params['stacking']['a_stack'] = 5.99766143
    override_base_params['stacking']['a_stack_1'] = 2.01059752
    override_base_params['stacking']['a_stack_2'] = 2.00055296
    override_base_params['stacking']['a_stack_4'] = 1.29545318
    override_base_params['stacking']['a_stack_5'] = 0.89833459 # THIS CAUSES A PROBLEM. TRY CHANGNING THE VALUE MORE!
    override_base_params['stacking']['a_stack_6'] = 0.89947816 # THIS CAUSES A PROBLEM. TRY CHANGING THE VALUE MORE!
    override_base_params['stacking']['delta_theta_star_stack_4'] = 0.8
    override_base_params['stacking']['delta_theta_star_stack_5'] = 0.95
    override_base_params['stacking']['delta_theta_star_stack_6'] = 0.93864677
    override_base_params['stacking']['dr0_stack'] = 0.39436836
    override_base_params['stacking']['dr_c_stack'] = 0.89587676

    override_base_params['stacking']['dr_high_stack'] = 0.76012637
    override_base_params['stacking']['dr_low_stack'] = 0.32
    override_base_params['stacking']['eps_stack_base'] = 1.34052305
    override_base_params['stacking']['eps_stack_kt_coeff'] = 2.65252305
    override_base_params['stacking']['neg_cos_phi1_star_stack'] = -0.65
    override_base_params['stacking']['neg_cos_phi2_star_stack'] = -0.65

    override_base_params['stacking']['theta0_stack_4'] = 0.00160911
    override_base_params['stacking']['theta0_stack_5'] = 0.00156078 # THIS CAUSES A PROBLEM. TRY CHANGING THE VALUE MORE!
    override_base_params['stacking']['theta0_stack_6'] = -0.00034034 # THIS CAUSES A PROBLEM. TRY CHANGING THE VALUE MORE!
    """


    recompile_oxdna(override_base_params, oxdna_path, t_kelvin, num_threads=4)


    sys_dir = Path("/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/templates/simple-helix")
    template_path = sys_dir / "input"
    top_path = sys_dir / "sys.top"
    init_conf_path = sys_dir / "init.conf"

    output_dir = Path("tmp-dir")

    output_dir.mkdir(parents=False, exist_ok=False)
    shutil.copy(top_path, output_dir / "sys.top")
    shutil.copy(init_conf_path, output_dir / "init.conf")

    rewrite_input_file(template_path, output_dir,
                       temp=f"{t_kelvin}K", steps=50000,
                       init_conf_path=str(output_dir / "init.conf"),
                       top_path=str(output_dir / "sys.top"),
                       save_interval=10000, seed=3,
                       equilibration_steps=100, dt=5e-3
    )



    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"
    input_path = output_dir / "input"


    oxdna_process = subprocess.run([oxdna_exec_path, input_path])
    rc = oxdna_process.returncode
    if rc != 0:
        raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")


    # Load the trajectory and check the energies

    output_dir = Path("tmp-dir")
    energy_path = output_dir / "energy.dat"
    energy_df = pd.read_csv(
            energy_path,
            names=["time", "potential_energy", "kinetic_energy", "total_energy"],
            delim_whitespace=True)

    displacement_fn, shift_fn = space.free()
    em = model.EnergyModel(displacement_fn, override_base_params, t_kelvin=t_kelvin)

    top_info = topology.TopologyInfo(output_dir / "sys.top", reverse_direction=False)
    seq_oh = jnp.array(get_one_hot(top_info.seq), dtype=jnp.float64)


    energy_fn = lambda body: em.energy_fn(body,
                                          seq=seq_oh,
                                          bonded_nbrs=top_info.bonded_nbrs,
                                          unbonded_nbrs=top_info.unbonded_nbrs.T)

    traj_info = trajectory.TrajectoryInfo(top_info, read_from_file=True,
                                          traj_path=output_dir / "output.dat",
                                          reverse_direction=True)

    traj_states = traj_info.get_states()
    traj_states = tree_stack(traj_states)

    calc_energies = vmap(energy_fn)(traj_states)
    gt_energies = energy_df.iloc[1:, :].potential_energy.to_numpy() * seq_oh.shape[0]

    atol_places = 3

    for i, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
        print(f"State {i}:")
        print(f"\t- Ours: {calc}")
        print(f"\t- Reference: {gt}")
        diff = onp.abs(calc - gt)
        print(f"\t- Difference: {diff}")
        if diff >= 10**(-atol_places):
            print(f"\t- WARNING: difference greater than tolerance")
        # assert(diff < 10**(-atol_places))
