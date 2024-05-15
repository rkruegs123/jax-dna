# ruff: noqa
# fmt: off
import pdb
import numpy as onp
from textwrap import dedent
import pandas as pd
from pathlib import Path
from io import StringIO
from copy import deepcopy

from jax_dna.dna1.load_params import load
from jax_dna.common import utils
from jax_dna.dna2 import model



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
        override_base_params, fname,
        kT=0.1, salt_conc=0.15, qeff=0.815,
        force_pn=2, torque_pnnm=10,
        k_restore=1217.58,
        save_every=660, n_steps=6600660,
        seq_avg=True, seed=30362,
        timestep=0.01
):
    base_params = model.get_full_base_params(override_base_params, seq_avg=True) # Note: seq_avg is always true for base parameter construction
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

    timestep {timestep}


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


def stretch_tors_data_constructor(body, seq, fname):

    assert(body.center.shape[0] == 80)
    n = body.center.shape[0]

    data_str = """
    # LAMMPS data file
    80 atoms
    80 ellipsoids
    78 bonds

    4 atom types
    1 bond types

    # System size
    -25.739900 25.739800 xlo xhi
    -25.739900 25.739800 ylo yhi
    -25.739900 25.739800 zlo zhi

    Masses

    1 3.1575
    2 3.1575
    3 3.1575
    4 3.1575

    # Atom-ID, type, position, molecule-ID, ellipsoid flag, density
    Atoms

    """

    pos_str = ""
    for i in range(n):
        nuc_idx = utils.DNA_ALPHA.index(seq[i])
        nuc_pos = body.center[i]
        if i >= 40:
            strand = 2
        else:
            strand = 1
        pos_str += f"\t{i+1} {nuc_idx+1} {nuc_pos[0]} {nuc_pos[1]} {nuc_pos[2]} {strand} 1 1\n"
    data_str += pos_str

    data_str += """

    # Atom-ID, translational, rotational velocity
    Velocities

    1  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    2  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    3  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    4  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    5  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    6  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    7  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    8  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    9  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    10  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    11  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    12  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    13  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    14  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    15  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    16  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    17  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    18  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    19  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    20  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    21  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    22  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    23  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    24  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    25  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    26  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    27  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    28  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    29  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    30  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    31  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    32  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    33  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    34  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    35  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    36  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    37  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    38  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    39  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    40  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    41  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    42  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    43  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    44  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    45  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    46  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    47  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    48  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    49  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    50  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    51  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    52  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    53  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    54  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    55  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    56  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    57  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    58  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    59  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    60  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    61  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    62  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    63  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    64  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    65  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    66  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    67  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    68  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    69  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    70  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    71  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    72  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    73  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    74  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    75  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    76  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    77  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    78  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    79  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00
    80  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00

    # Atom-ID, shape, quaternion
    Ellipsoids

    """

    quat_str = ""
    for i in range(n):
        ith_quat = body.orientation[i].vec
        quat_str += f"\t {i+1} 1.173984503142341e+00  1.173984503142341e+00  1.173984503142341e+00 {ith_quat[0]} {ith_quat[1]} {ith_quat[2]} {ith_quat[3]}\n"

    data_str += quat_str

    data_str += """

    # Bond topology
    Bonds

    1  1  1  2
    2  1  2  3
    3  1  3  4
    4  1  4  5
    5  1  5  6
    6  1  6  7
    7  1  7  8
    8  1  8  9
    9  1  9  10
    10  1  10  11
    11  1  11  12
    12  1  12  13
    13  1  13  14
    14  1  14  15
    15  1  15  16
    16  1  16  17
    17  1  17  18
    18  1  18  19
    19  1  19  20
    20  1  20  21
    21  1  21  22
    22  1  22  23
    23  1  23  24
    24  1  24  25
    25  1  25  26
    26  1  26  27
    27  1  27  28
    28  1  28  29
    29  1  29  30
    30  1  30  31
    31  1  31  32
    32  1  32  33
    33  1  33  34
    34  1  34  35
    35  1  35  36
    36  1  36  37
    37  1  37  38
    38  1  38  39
    39  1  39  40
    40  1  41  42
    41  1  42  43
    42  1  43  44
    43  1  44  45
    44  1  45  46
    45  1  46  47
    46  1  47  48
    47  1  48  49
    48  1  49  50
    49  1  50  51
    50  1  51  52
    51  1  52  53
    52  1  53  54
    53  1  54  55
    54  1  55  56
    55  1  56  57
    56  1  57  58
    57  1  58  59
    58  1  59  60
    59  1  60  61
    60  1  61  62
    61  1  62  63
    62  1  63  64
    63  1  64  65
    64  1  65  66
    65  1  66  67
    66  1  67  68
    67  1  68  69
    68  1  69  70
    69  1  70  71
    70  1  71  72
    71  1  72  73
    72  1  73  74
    73  1  74  75
    74  1  75  76
    75  1  76  77
    76  1  77  78
    77  1  78  79
    78  1  79  80
    """

    data_str_dedent = '\n'.join([line.lstrip() for line in data_str.split('\n')])
    with open(fname, "w") as f:
        f.write(data_str_dedent)

    return

if __name__ == "__main__":

    # Test writing an input file
    test_write_in = False
    if test_write_in:
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

    # Test writing a data file
    test_write_data = True
    if test_write_data:

        from jax_dna.common import topology, trajectory

        sys_basedir = Path("data/test-data/lammps-oxdna2-40bp")
        top_fpath = sys_basedir / "data.top"
        # conf_fpath = sys_basedir / "data.oxdna"
        conf_fpath = "orig_data.oxdna"

        top_info = topology.TopologyInfo(top_fpath, reverse_direction=False)
        seq = top_info.seq


        conf_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True, reindex=True,
            traj_path=conf_fpath,
            # reverse_direction=True)
            reverse_direction=False)
        init_state = conf_info.get_states()[0]

        out_fname = "test_data"
        stretch_tors_data_constructor(init_state, seq, out_fname)
