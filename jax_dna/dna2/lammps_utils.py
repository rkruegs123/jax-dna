import pdb
import numpy as onp
from textwrap import dedent
import pandas as pd
from pathlib import Path
from io import StringIO
from copy import deepcopy

from jax_dna.dna1.load_params import load
from jax_dna.common import utils



def read_log(log_path):
    with open(log_path, "r") as f:
        log_lines = f.readlines()

    start_line_idx = -1
    end_line_idx = -1
    check_start_str = "v_tns Temp"
    check_end_str = "Loop time of"
    n_check_end_str = 0
    for idx, line in enumerate(log_lines):
        if line[:len(check_start_str)] == check_start_str:
            assert(start_line_idx == -1)
            start_line_idx = idx

        if line[:len(check_end_str)] == check_end_str:
            n_check_end_str += 1
            end_line_idx = idx
    assert(n_check_end_str <= 2)


    log_df_lines = log_lines[start_line_idx:end_line_idx]
    log_df = pd.read_csv(StringIO(''.join(log_df_lines)), delim_whitespace=True)

    return log_df


def get_excv_cmd(params):
    excv_p = params['excluded_volume']
    return f"""
    pair_coeff * * oxdna2/excv {excv_p['eps_exc']} {excv_p['sigma_backbone']} {excv_p['dr_star_backbone']} {excv_p['eps_exc']} {excv_p['sigma_back_base']} {excv_p['dr_star_back_base']} {excv_p['eps_exc']} {excv_p['sigma_base']} {excv_p['dr_star_base']}
    """

def get_stk_cmd(params, seq_avg, kT):
    stk_p = params['stacking']
    stk_suffix = f"{kT} {stk_p['eps_stack_base']} {stk_p['eps_stack_kt_coeff']} {stk_p['a_stack']} {stk_p['dr0_stack']} {stk_p['dr_c_stack']} {stk_p['dr_low_stack']} {stk_p['dr_high_stack']} {stk_p['a_stack_4']} {stk_p['theta0_stack_4']} {stk_p['delta_theta_star_stack_4']} {stk_p['a_stack_5']} {stk_p['theta0_stack_5']} {stk_p['delta_theta_star_stack_5']} {stk_p['a_stack_6']} {stk_p['theta0_stack_6']} {stk_p['delta_theta_star_stack_6']} {stk_p['a_stack_1']} {-stk_p['neg_cos_phi1_star_stack']} {stk_p['a_stack_2']} {-stk_p['neg_cos_phi2_star_stack']}"

    if seq_avg:
        seq_dep_str = "seqav"
    else:
        seq_dep_str = "seqdep"

    return f"pair_coeff * * oxdna2/stk {seq_dep_str} {stk_suffix}"

def get_hb_cmd(params, seq_avg):
    hb_p = params['hydrogen_bonding']
    hb_suffix = f"{hb_p['a_hb']} {hb_p['dr0_hb']} {hb_p['dr_c_hb']} {hb_p['dr_low_hb']} {hb_p['dr_high_hb']} {hb_p['a_hb_1']} {hb_p['theta0_hb_1']} {hb_p['delta_theta_star_hb_1']} {hb_p['a_hb_2']} {hb_p['theta0_hb_2']} {hb_p['delta_theta_star_hb_2']} {hb_p['a_hb_3']} {hb_p['theta0_hb_3']} {hb_p['delta_theta_star_hb_3']} {hb_p['a_hb_4']} {hb_p['theta0_hb_4']} {hb_p['delta_theta_star_hb_4']} {hb_p['a_hb_7']} {hb_p['theta0_hb_7']} {hb_p['delta_theta_star_hb_7']} {hb_p['a_hb_8']} {hb_p['theta0_hb_8']} {hb_p['delta_theta_star_hb_8']}" # Note: suffix doesn't include first parameter

    if seq_avg:
        seq_dep_str = "seqav"
    else:
        seq_dep_str = "seqdep"

    hb_str = f"""
    pair_coeff * * oxdna2/hbond {seq_dep_str} 0.0 {hb_suffix}
    pair_coeff 1 4 oxdna2/hbond {seq_dep_str} {params['hydrogen_bonding']['eps_hb']} {hb_suffix}
    pair_coeff 2 3 oxdna2/hbond {seq_dep_str} {params['hydrogen_bonding']['eps_hb']} {hb_suffix}"""
    return hb_str

def get_xstk_cmd(params):
    xstk_p = params['cross_stacking']
    return f"""
    pair_coeff * * oxdna2/xstk {xstk_p['k_cross']} {xstk_p['r0_cross']} {xstk_p['dr_c_cross']} {xstk_p['dr_low_cross']} {xstk_p['dr_high_cross']} {xstk_p['a_cross_1']} {xstk_p['theta0_cross_1']} {xstk_p['delta_theta_star_cross_1']} {xstk_p['a_cross_2']} {xstk_p['theta0_cross_2']} {xstk_p['delta_theta_star_cross_2']} {xstk_p['a_cross_3']} {xstk_p['theta0_cross_3']} {xstk_p['delta_theta_star_cross_3']} {xstk_p['a_cross_4']} {xstk_p['theta0_cross_4']} {xstk_p['delta_theta_star_cross_4']} {xstk_p['a_cross_7']} {xstk_p['theta0_cross_7']} {xstk_p['delta_theta_star_cross_7']} {xstk_p['a_cross_8']} {xstk_p['theta0_cross_8']} {xstk_p['delta_theta_star_cross_8']}"""

def get_coaxstk_cmd(params):
    cxstk_p = params['coaxial_stacking']
    return f"""
    pair_coeff * * oxdna2/coaxstk {cxstk_p['k_coax']} {cxstk_p['dr0_coax']} {cxstk_p['dr_c_coax']} {cxstk_p['dr_low_coax']} {cxstk_p['dr_high_coax']} {cxstk_p['a_coax_1']} {cxstk_p['theta0_coax_1']} {cxstk_p['delta_theta_star_coax_1']} {cxstk_p['a_coax_4']} {cxstk_p['theta0_coax_4']} {cxstk_p['delta_theta_star_coax_4']} {cxstk_p['a_coax_5']} {cxstk_p['theta0_coax_5']} {cxstk_p['delta_theta_star_coax_5']} {cxstk_p['a_coax_6']} {cxstk_p['theta0_coax_6']} {cxstk_p['delta_theta_star_coax_6']} {cxstk_p['A_coax_1_f6']} {cxstk_p['B_coax_1_f6']}"""

def get_dh_cmd(kT, salt_conc, qeff):
    return f"""
    pair_coeff * * oxdna2/dh {kT} {salt_conc} {qeff}"""


def lammpsify_params(base_params):
    
    tmp7 = base_params['hydrogen_bonding']['theta0_hb_7']
    tmp8 = base_params['hydrogen_bonding']['theta0_hb_8']
    base_params['hydrogen_bonding']['theta0_hb_8'] = tmp7
    base_params['hydrogen_bonding']['theta0_hb_7'] = onp.pi - tmp8
    
    tmp7 = base_params['hydrogen_bonding']['delta_theta_star_hb_7']
    tmp8 = base_params['hydrogen_bonding']['delta_theta_star_hb_8']
    base_params['hydrogen_bonding']['delta_theta_star_hb_8'] = tmp7
    base_params['hydrogen_bonding']['delta_theta_star_hb_7'] = tmp8
    
    tmp7 = base_params['hydrogen_bonding']['a_hb_7']
    tmp8 = base_params['hydrogen_bonding']['a_hb_8']
    base_params['hydrogen_bonding']['a_hb_8'] = tmp7
    base_params['hydrogen_bonding']['a_hb_7'] = tmp8

    # Note: we don't know if this is just a LAMMPS thing or if we have them swapped
    tmp7 = base_params['cross_stacking']['theta0_cross_7']
    tmp8 = base_params['cross_stacking']['theta0_cross_8']
    base_params['cross_stacking']['theta0_cross_7'] = tmp8
    base_params['cross_stacking']['theta0_cross_8'] = tmp7

    tmp7 = base_params['cross_stacking']['a_cross_7']
    tmp8 = base_params['cross_stacking']['a_cross_8']
    base_params['cross_stacking']['a_cross_7'] = tmp8
    base_params['cross_stacking']['a_cross_8'] = tmp7

    tmp7 = base_params['cross_stacking']['delta_theta_star_cross_7']
    tmp8 = base_params['cross_stacking']['delta_theta_star_cross_8']
    base_params['cross_stacking']['delta_theta_star_cross_7'] = tmp8
    base_params['cross_stacking']['delta_theta_star_cross_8'] = tmp7

    tmp3 = base_params['cross_stacking']['theta0_cross_3']
    tmp2 = base_params['cross_stacking']['theta0_cross_2']
    base_params['cross_stacking']['theta0_cross_3'] = tmp2
    base_params['cross_stacking']['theta0_cross_2'] = tmp3

    tmp3 = base_params['cross_stacking']['a_cross_3']
    tmp2 = base_params['cross_stacking']['a_cross_2']
    base_params['cross_stacking']['a_cross_3'] = tmp2
    base_params['cross_stacking']['a_cross_2'] = tmp3

    tmp3 = base_params['cross_stacking']['delta_theta_star_cross_3']
    tmp2 = base_params['cross_stacking']['delta_theta_star_cross_2']
    base_params['cross_stacking']['delta_theta_star_cross_3'] = tmp2
    base_params['cross_stacking']['delta_theta_star_cross_2'] = tmp3


def stretch_tors_constructor(
        base_params, fname,
        kT=0.1, salt_conc=0.15, qeff=0.815,
        force_pn=2, torque_pnnm=10,
        k_restore=1217.58,
        save_every=660, n_steps=6600660,
        seq_avg=True, seed=30362
):
    params = deepcopy(base_params)
    lammpsify_params(params)

    if seed <= 0:
        raise RuntimeError(f"LAMMPS seed must be a nonzero positive integer")

    assert(n_steps % save_every == 0)
    force_oxdna_per_nuc = (force_pn / utils.oxdna_force_to_pn) / 2
    torque_oxdna = torque_pnnm / (utils.oxdna_force_to_pn * utils.nm_per_oxdna_length)

    ## Setup
    setup_str = """
    units lj

    dimension 3

    newton off

    boundary  p p p

    atom_style hybrid bond ellipsoid oxdna
    atom_modify sort 0 1.0

    # Pair interactions require lists of neighbours to be calculated
    neighbor 1.0 bin
    neigh_modify every 1 delay 0 check yes

    read_data data

    set atom * mass 3.1575

    group all type 1 4
    """

    ## Interaction
    interaction_str = f"""

    # oxDNA bond interactions - FENE backbone
    bond_style oxdna2/fene
    bond_coeff * {params['fene']['eps_backbone']} {params['fene']['delta_backbone']} {params['fene']['r0_backbone']}
    special_bonds lj 0 1 1

    # oxDNA pair interactions
    pair_style hybrid/overlay oxdna2/excv oxdna2/stk oxdna2/hbond oxdna2/xstk oxdna2/coaxstk oxdna2/dh
    """

    excv_str = get_excv_cmd(params)
    interaction_str += excv_str

    stk_str = get_stk_cmd(params, seq_avg, kT)
    interaction_str += stk_str

    hb_str = get_hb_cmd(params, seq_avg)
    interaction_str += hb_str

    xstk_str = get_xstk_cmd(params)
    interaction_str += xstk_str

    coaxstk_str = get_coaxstk_cmd(params)
    interaction_str += coaxstk_str

    dh_str = get_dh_cmd(kT, salt_conc, qeff)
    interaction_str += dh_str

    ## Integration
    integrate_str = f"""

    fix 1 all nve/dotc/langevin {kT} {kT} 100.0 {seed} angmom 1000

    timestep 0.01


    # Added by RK
    compute hbondEnergy all pair oxdna2/hbond
    compute excvEnergy all pair oxdna2/excv
    compute stkEnergy all pair oxdna2/stk
    compute xstkEnergy all pair oxdna2/xstk
    compute coaxstkEnergy all pair oxdna2/coaxstk
    compute dhEnergy all pair oxdna2/dh
    compute quat all property/atom quatw quati quatj quatk
    # End: added by RK

    compute         xu all property/atom xu
    compute         yu all property/atom yu
    variable        dx equal -0.5*(c_xu[39]+c_xu[42])
    variable        dy equal -0.5*(c_yu[39]+c_yu[42])
    thermo_style    custom v_dx v_dy
    run             0
    displace_atoms  all move v_dx v_dy 0 units box
    thermo_style    one

    group           blockA1 id <= 2
    group           blockA2 id >= 79
    group           blockA union blockA1 blockA2
    group           fB id 39 42
    group           torque id <> 39 42
    fix             tetherA blockA spring/self {k_restore} xyz
    variable        fxB equal {-k_restore}*(c_xu[39]+c_xu[42])
    variable        fyB equal {-k_restore}*(c_yu[39]+c_yu[42])
    fix             fB fB addforce v_fxB v_fyB {force_oxdna_per_nuc}
    fix             torque torque addtorque 0 0 {torque_oxdna}

    variable        tns equal time*3.03e-3
    variable        cpuh equal cpuremain/3600
    thermo_style    custom v_tns temp evdwl ecoul ebond eangle edihed pe v_cpuh c_hbondEnergy c_excvEnergy c_stkEnergy c_xstkEnergy c_coaxstkEnergy c_dhEnergy
    thermo          {save_every}

    timestep        0.01
    dump            coord all custom {save_every} dump.lammpstrj id type xu yu zu
    dump_modify coord sort id


    dump 4 all custom {save_every} filename.dat & 
        id mol type x y z ix iy iz vx vy vz &
        c_quat[1] c_quat[2] c_quat[3] c_quat[4] &
        angmomx angmomy angmomz
    dump_modify 4 sort id
    dump_modify 4 format line "&
        %d %d %d %22.15le %22.15le %22.15le &
        %d %d %d %22.15le %22.15le %22.15le &
        %22.15le %22.15le %22.15le %22.15le &
        %22.15le %22.15le %22.15le"

    run             {n_steps}

    shell           touch end
    """

    input_file_str = setup_str + interaction_str + integrate_str
    with open(fname, "w") as f:
        f.write(dedent(input_file_str))

    return

if __name__ == "__main__":
    ex_base_params = load(process=False)
    ex_base_params['fene']['r0_backbone'] = 0.7564
    ex_base_params['stacking']['eps_stack_base'] = 1.3523
    ex_base_params['stacking']['eps_stack_kt_coeff'] = 2.6717
    ex_base_params['hydrogen_bonding']['eps_hb'] = 1.0678
    ex_base_params['coaxial_stacking']['k_coax'] = 58.5
    ex_base_params['coaxial_stacking']['theta0_coax_1'] = onp.pi - 0.25
    ex_base_params['coaxial_stacking']['A_coax_1_f6'] = 40.0
    ex_base_params['coaxial_stacking']['B_coax_1_f6'] = onp.pi - 0.025
    stretch_tors_constructor(ex_base_params, "test.in", kT=0.1, n_steps=6600660)

    print("done")
