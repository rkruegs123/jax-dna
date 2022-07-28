import toml
from pathlib import Path
import pdb
import numpy as np

from jax import vmap
from jax_md.rigid_body import Quaternion, RigidBody
import jax.numpy as jnp

from smoothing import get_f1_smoothing_params, get_f2_smoothing_params, get_f3_smoothing_params, \
    get_f4_smoothing_params, get_f5_smoothing_params



# FIXME: wrong?
def principal_axes_to_euler_angles(x, y, z):
    alpha = np.arccos(-z[1] / (np.sqrt(1 - z[2]**2)))
    beta = np.arccos(z[2])
    gamma = np.arccos(y[2] / np.sqrt(1 - z[2]**2))
    return alpha, beta, gamma


# Read in oxDNA file
def read_config(fpath):
    if not Path(fpath).exists():
        raise RuntimeError(f"Configuration file does not exist at location: {fpath}")

    with open(fpath) as f:
        config_lines = f.readlines()

    box_size = config_lines[1].split('=')[1].strip().split(' ')
    box_size = np.array(box_size).astype(np.float64)

    nuc_lines = config_lines[3:]
    n = len(nuc_lines)
    R = np.empty((n, 3), dtype=np.float64)
    quat = np.empty((n, 4), dtype=np.float64)
    for i, nuc_line in enumerate(nuc_lines):
        nuc_info = np.array(nuc_line.strip().split(' '), dtype=np.float64)
        assert(nuc_info.shape[0] == 15)

        com = nuc_info[:3]
        base_vector = nuc_info[3:6]
        base_normal = nuc_info[6:9]
        velocity = nuc_info[9:12]
        angular_velocity = nuc_info[12:15]

        # Method 1
        """
        alpha, beta, gamma = principal_axes_to_euler_angles(base_vector,
                                                            base_normal,
                                                            np.cross(base_normal, base_vector))

        def get_q(eenie, meenie, miney):
            q0 = np.cos(meenie / 2) * np.cos(0.5*(eenie + miney))
            q1 = np.sin(meenie / 2) * np.cos(0.5*(eenie - miney))
            q2 = np.sin(meenie / 2) * np.sin(0.5*(eenie - miney))
            q3 = np.cos(meenie / 2) * np.sin(0.5*(eenie + miney))
            q = Quaternion(np.array([q0, q1, q2, q3]))
            return q

        q = get_q(alpha, beta, gamma)
        recovered_v1 = q_to_v1(q)
        """

        # Method 2
        rot_matrix = np.array([base_vector, np.cross(base_normal, base_vector), base_normal]).T
        tr = np.trace(rot_matrix)
        q0 = np.sqrt((tr + 1) / 4)
        q1 = np.sqrt(rot_matrix[0, 0] / 2 + (1 - tr) / 4)
        q2 = np.sqrt(rot_matrix[1, 1] / 2 + (1 - tr) / 4)
        q3 = np.sqrt(rot_matrix[2, 2] / 2 + (1 - tr) / 4)

        # q = Quaternion(np.array([q0, q1, q2, q3]))
        # recovered_v1 = q_to_v1(q)
        # recovered_v2 = q_to_v2(q)
        # recovered_v3 = q_to_v3(q)

        R[i, :] = com
        quat[i, :] = np.array([q0, q1, q2, q3])

    body = RigidBody(R, Quaternion(quat))
    return body, box_size


# Transform quaternions to nucleotide orientations

## backbone-bsae orientation
def q_to_v1(q):
    q0, q1, q2, q3 = q.vec
    return jnp.array([
        q0**2 + q1**2 - q2**2 - q3**2,
        2*(q1*q2 + q0*q3),
        2*(q1*q3 - q0*q2)
    ])
Q_to_v1 = vmap(q_to_v1) # Q is system of quaternions, q is an individual quaternion

## normal orientation
def q_to_v2(q):
    q0, q1, q2, q3 = q.vec
    return jnp.array([
        2*(q1*q3 + q0*q2),
        2*(q2*q3 - q0*q1),
        q0**2 - q1**2 - q2**2 + q3**2
    ])
Q_to_v2 = vmap(q_to_v2)

## third axis (n x b)
def q_to_v3(q):
    q0, q1, q2, q3 = q.vec
    return jnp.array([
        2*(q1*q2 - q0*q3),
        q0**2 - q1**2 + q2**2 - q3**2,
        2*(q2*q3 + q0*q1)
    ])
Q_to_v3 = vmap(q_to_v3)


# FIXME: should really take temperature as input (or kT)
# FIXME: so, this means that thing rely on kT should really only be intermediate values in the *.toml file and they should be updaed to the full (whose name currently exists in the toml file) herex
def get_params(config_path="tom.toml"):
    if not Path(config_path).exists():
        raise RuntimeError(f"No file at location: {config_path}")
    params = toml.load(config_path)

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


if __name__ == "__main__":
    # final_params = get_params()

    read_config("data/polyA_10bp/generated.dat")
