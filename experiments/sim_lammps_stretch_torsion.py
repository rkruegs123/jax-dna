from pathlib import Path
import shutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import subprocess
import pdb
from copy import deepcopy
import time
import functools
import numpy as onp
import pprint

from jax import jit, vmap, lax, value_and_grad
import jax.numpy as jnp
from jax_md import space, rigid_body
import optax

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.dna2 import model, lammps_utils


checkpoint_every = 10
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def single_pitch(quartet, base_sites, displacement_fn):
    # a1 is h-bonded to b1, a2 is h-bonded to b2
    a1, b1, a2, b2 = quartet

    # get normalized base-base vectors for each base pair, 1 and 2
    bb1 = displacement_fn(base_sites[b1], base_sites[a1])
    bb2 = displacement_fn(base_sites[b2], base_sites[a2])

    bb1 = bb1[:2]
    bb2 = bb2[:2]

    bb1 = bb1 / jnp.linalg.norm(bb1)
    bb2 = bb2 / jnp.linalg.norm(bb2)

    theta = jnp.arccos(utils.clamp(jnp.dot(bb1, bb2)))

    return theta


def compute_pitches(body, quartets, displacement_fn, com_to_hb):
    # Construct the base site position in the body frame
    base_site_bf = jnp.array([com_to_hb, 0.0, 0.0])

    # Compute the space-frame base sites
    base_sites = body.center + rigid_body.quaternion_rotate(
        body.orientation, base_site_bf)

    # Compute the pitches for all quartets
    all_pitches = vmap(single_pitch, (0, None, None))(
        quartets, base_sites, displacement_fn)

    return all_pitches


def get_bp_pos(body, bp):
    return (body.center[bp[0]] + body.center[bp[1]]) / 2

def run(args):
    lammps_basedir = Path(args['lammps_basedir'])
    assert(lammps_basedir.exists())
    lammps_exec_path = lammps_basedir / "build/lmp"
    assert(lammps_exec_path.exists())

    tacoxdna_basedir = Path(args['tacoxdna_basedir'])
    assert(tacoxdna_basedir.exists())

    sample_every = args['sample_every']

    n_eq_steps = args['n_eq_steps']
    assert(n_eq_steps % sample_every == 0)
    n_eq_states = n_eq_steps // sample_every

    n_sample_steps = args['n_sample_steps']
    assert(n_sample_steps % sample_every == 0)
    n_sample_states = n_sample_steps // sample_every

    n_total_steps = n_eq_steps + n_sample_steps
    n_total_states = n_total_steps // sample_every
    assert(n_total_states == n_sample_states + n_eq_states)

    run_name = args['run_name']
    seq_avg = not args['seq_dep']
    assert(seq_avg)

    force_pn = args['force_pn']
    torque_pnnm = args['torque_pnnm']


    # Setup the logging directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)


    params_str = ""
    params_str += f"n_sample_states: {n_sample_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    # Load the system
    sys_basedir = Path("data/templates/lammps-stretch-tors")
    lammps_data_rel_path = sys_basedir / "data"
    lammps_data_abs_path = os.getcwd() / lammps_data_rel_path

    p = subprocess.Popen([tacoxdna_basedir / "src/LAMMPS_oxDNA.py", lammps_data_abs_path], cwd=run_dir)
    p.wait()
    rc = p.returncode
    if rc != 0:
        raise RuntimeError(f"tacoxDNA conversion failed with error code: {rc}")

    init_conf_fpath = run_dir / "data.oxdna"
    assert(init_conf_fpath.exists())
    os.rename(init_conf_fpath, run_dir / "init.conf")

    top_fpath = run_dir / "data.top"
    assert(top_fpath.exists())
    os.rename(top_fpath, run_dir / "sys.top")
    top_fpath = run_dir / "sys.top"

    top_info = topology.TopologyInfo(top_fpath, reverse_direction=False)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
    n = seq_oh.shape[0]
    assert(n % 2 == 0)
    n_bp = n // 2

    strand1_start = 0
    strand1_end = n_bp-1
    strand2_start = n_bp
    strand2_end = n_bp*2-1

    ## The region for which theta and distance are measured
    quartets = utils.get_all_quartets(n_nucs_per_strand=n_bp)
    quartets = quartets[4:n_bp-5]

    bp1_meas = [4, strand2_end-4]
    bp2_meas = [strand1_end-4, strand2_start+4]

    rise_per_bp = 3.4 / utils.ang_per_oxdna_length # oxDNA length units
    contour_length = quartets.shape[0] * rise_per_bp # oxDNA length units

    displacement_fn, shift_fn = space.free() # FIXME: could use box size from top_info, but not sure how the centering works.

    def compute_theta(body):
        pitches = compute_pitches(body, quartets, displacement_fn, model.com_to_hb)
        return pitches.sum()

    t_kelvin = 300.0
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    salt_conc = 0.15
    q_eff = 0.815

    def stretch_torsion(params, i, seed):
        

        sim_dir = run_dir / f"sim"
        sim_dir.mkdir(parents=False, exist_ok=False)

        base_params = model.get_full_base_params(params, seq_avg=seq_avg)


        shutil.copy(lammps_data_abs_path, sim_dir / "data")
        lammps_in_fpath = sim_dir / "in"
        lammps_utils.stretch_tors_constructor(
            base_params, lammps_in_fpath, kT=kT, salt_conc=salt_conc, qeff=q_eff,
            force_pn=force_pn, torque_pnnm=torque_pnnm,
            save_every=sample_every, n_steps=n_total_steps,
            seq_avg=seq_avg, seed=seed)

        sim_start = time.time()
        p = subprocess.Popen([lammps_exec_path, "-in", lammps_in_fpath.stem], cwd=sim_dir)
        p.wait()
        rc = p.returncode
        if rc != 0:
            raise RuntimeError(f"LAMMPS simulation failed with error code: {rc}")
        sim_end = time.time()

        # Convert via TaxoxDNA
        p = subprocess.Popen([tacoxdna_basedir / "src/LAMMPS_oxDNA.py", "data", "filename.dat"], cwd=sim_dir)
        p.wait()
        rc = p.returncode
        if rc != 0:
            raise RuntimeError(f"tacoxDNA conversion failed with error code: {rc}")

        traj_path = sim_dir / "data.oxdna"
        assert(traj_path.exists())
        top_path = sim_dir / "data.top"
        assert(top_path.exists())

        # Analyze

        ## Load states from oxDNA simulation
        load_start = time.time()
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True,
            traj_path=sim_dir / "data.oxdna",
            # reverse_direction=True)
            reverse_direction=False)
        traj_states = traj_info.get_states()
        traj_states = traj_states[1:] # ignore the initial state
        n_traj_states = len(traj_states)
        assert(n_traj_states == n_total_states)
        traj_states = traj_states[n_eq_states:] # Remove the states freom the equlibration period
        assert(len(traj_states) == n_sample_states)
        traj_states = utils.tree_stack(traj_states)
        load_end = time.time()

        ## Load the LAMMPS energies
        log_path = sim_dir / "log.lammps"
        log_df = lammps_utils.read_log(log_path)
        assert(log_df.shape[0] == n_total_states+1)
        log_df = log_df[1+n_eq_states:]

        ## Generate an energy function
        em = model.EnergyModel(displacement_fn,
                               params,
                               t_kelvin=t_kelvin,
                               salt_conc=salt_conc, q_eff=q_eff, seq_avg=seq_avg,
                               ignore_exc_vol_bonded=True # Because we're in LAMMPS
        )
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        subterms_fn = lambda body: em.compute_subterms(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        subterms_fn = jit(subterms_fn)

        ## Compute the energies via our energy function
        calc_start = time.time()
        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, calc_energies = scan(energy_scan_fn, None, traj_states)
        subterms_scan_fn = lambda state, ts: (None, subterms_fn(ts))
        _, calc_subterms = scan(subterms_scan_fn, None, traj_states)
        calc_end = time.time()

        ## Check energies
        gt_energies = (log_df.PotEng * seq_oh.shape[0]).to_numpy()
        energy_diffs = list()
        for idx, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
            diff = onp.abs(calc - gt)
            energy_diffs.append(diff)


        ## Compute the mean theta and distance
        all_thetas = list()
        all_distances = list()
        for rs_idx in range(n_sample_states):
            ref_state = traj_states[rs_idx]

            theta = compute_theta(ref_state)
            all_thetas.append(theta)

            bp1_meas_pos = get_bp_pos(ref_state, bp1_meas)
            bp2_meas_pos = get_bp_pos(ref_state, bp2_meas)
            dist = onp.abs(bp1_meas_pos[2] - bp2_meas_pos[2])
            all_distances.append(dist)

        all_thetas = onp.array(all_thetas)
        all_distances = onp.array(all_distances)

        ## Record some plots

        plt.plot(all_distances)
        plt.savefig(run_dir / "distances_traj.png")
        plt.clf()

        running_avg = onp.cumsum(all_distances) / onp.arange(1, n_sample_states+1)
        plt.plot(running_avg)
        plt.savefig(run_dir / "running_avg_distance.png")
        plt.clf()

        plt.plot(all_thetas)
        plt.savefig(run_dir / "theta_traj.png")
        plt.clf()

        running_avg = onp.cumsum(all_thetas) / onp.arange(1, n_sample_states+1)
        plt.plot(running_avg)
        plt.savefig(run_dir / "running_avg_theta.png")
        plt.clf()

        sns.distplot(calc_energies, label="Calculated", color="red")
        sns.distplot(gt_energies, label="Reference", color="green")
        plt.legend()
        plt.savefig(run_dir / f"energies.png")
        plt.clf()

        sns.histplot(energy_diffs)
        plt.savefig(run_dir / f"energy_diffs.png")
        plt.clf()

        with open(run_dir / "summary.txt", "w+") as f:
            f.write(f"Mean distance: {onp.mean(all_distances)}\n")
            f.write(f"Mean theta: {onp.mean(all_thetas)}\n")
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")

        with open(run_dir / "override_params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        return

    broken_hb = {
        'a_hb': 7.970159150635577,
        'a_hb_1': 1.5079461916426353,
        'a_hb_2': 1.5283265147299099,
        'a_hb_3': 1.5283265147299099,
        'a_hb_4': 0.43237757026552776,
        'a_hb_7': 3.9889513188121284,
        'a_hb_8': 4.,
        'delta_theta_star_hb_1': 0.6613341297749521,
        'delta_theta_star_hb_2': 0.7205006795443899,
        'delta_theta_star_hb_3': 0.7205006795443899,
        'delta_theta_star_hb_4': 0.7300151794116861,
        'delta_theta_star_hb_7': 0.47783756,
        'delta_theta_star_hb_8': 0.45,
        'dr0_hb': 0.4126684741187398,
        'dr_c_hb': 0.769592630643982,
        'dr_high_hb': 0.7424138493790637,
        'dr_low_hb': 0.37198652010697586,
        'eps_hb': 1.0840363977133736,
        'theta0_hb_1': -0.006878755518420017,
        'theta0_hb_2': 0.008904771140613523,
        'theta0_hb_3': 0.008904771140613523,
        'theta0_hb_4': 3.113701107802724,
        'theta0_hb_7': 1.59150464,
        'theta0_hb_8': 1.57079633
    }
    broken_cross = {
        'a_cross_1': 2.44014803,
        'a_cross_2': 1.47784696,
        'a_cross_3': 1.53180218,
        'a_cross_4': 1.50328919,
        'a_cross_7': 1.54135167,
        'a_cross_8': 1.79642147,
        'delta_theta_star_cross_1': 0.59856018,
        'delta_theta_star_cross_2': 0.83909825,
        'delta_theta_star_cross_3': 0.68284451,
        'delta_theta_star_cross_4': 0.70123336,
        'delta_theta_star_cross_7': 0.68048415,
        'delta_theta_star_cross_8': 0.73230915,
        'dr_c_cross': 0.72762405,
        'dr_high_cross': 0.74435034,
        'dr_low_cross': 0.39017476,
        'k_cross': 47.65627499,
        'r0_cross': 0.55514459,
        'theta0_cross_1': 0.83859634,
        'theta0_cross_2': 0.92904047,
        'theta0_cross_3': 0.90951825,
        'theta0_cross_4': 0.03413192,
        'theta0_cross_7': 0.8736833,
        'theta0_cross_8': 0.91158186
    }
    broken_stack = {
        'a_stack': 6.048146645808571,
        'a_stack_1': 1.968742289860051,
        'a_stack_2': 2.0045065095474492,
        'a_stack_4': 1.2538097327234479,
        'a_stack_5': 0.8974215615327519,
        'a_stack_6': 0.8974215615327519,
        'delta_theta_star_stack_4': 0.8108991205232058,
        'delta_theta_star_stack_5': 0.9497858612607184,
        'delta_theta_star_stack_6': 0.95,
        'dr0_stack': 0.38327058255543583,
        'dr_c_stack': 0.9095648025773204,
        'dr_high_stack': 0.7428033970398432,
        'dr_low_stack': 0.29029093732566763,
        'eps_stack_base': 1.3859168732235818,
        'eps_stack_kt_coeff': 2.705316872793202,
        'neg_cos_phi1_star_stack': -0.6445451900903784,
        'neg_cos_phi2_star_stack': -0.65,
        'theta0_stack_4': 0.047134936890393914,
        'theta0_stack_5': 0.013246086230422657,
        'theta0_stack_6': 0.013246086230422657}
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    # params['stacking'] = broken_stack
    # params['hydrogen_bonding'] = broken_hb
    failing_cross_keys = [
        'dr_c_cross', 'dr_high_cross', 'dr_low_cross', 'k_cross', 'r0_cross', # 0-4
        'a_cross_1', 'delta_theta_star_cross_1', 'theta0_cross_1', # 5-7
        'a_cross_2', 'delta_theta_star_cross_2', 'theta0_cross_2', # 8-10
        'a_cross_3', 'delta_theta_star_cross_3', 'theta0_cross_3', # 11-13
        'a_cross_4', 'delta_theta_star_cross_4', 'theta0_cross_4', # 14-16
        'a_cross_7', 'delta_theta_star_cross_7', 'theta0_cross_7', # 17-19
        'a_cross_8', 'delta_theta_star_cross_8', 'theta0_cross_8' # 20-22
    ]
    for k in failing_cross_keys[4:]:
    # for k in []:
        params['cross_stacking'][k] = broken_cross[k]
    # params['cross_stacking'] = broken_cross

    start = time.time()
    stretch_torsion(params, i=0, seed=5)
    end = time.time()
    print(f"\nSimulation took {end - start} seconds\n")



def get_parser():

    parser = argparse.ArgumentParser(description="Optimize the length under a pulling force via LAMMPS")

    parser.add_argument('--n-sample-steps', type=int, default=5000000,
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=100000,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=500,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--lammps-basedir', type=str,
                        default="/n/brenner_lab/Lab/software/lammps-stable_29Sep2021",
                        help='LAMMPS base directory')
    parser.add_argument('--tacoxdna-basedir', type=str,
                        default="/n/brenner_lab/User/rkrueger/tacoxDNA",
                        help='tacoxDNA base directory')

    parser.add_argument('--force-pn', type=float, default=2.5,
                        help="Total pulling force in pN")
    parser.add_argument('--torque-pnnm', type=float, default=0.0,
                        help="Total torque in pN*nm")
    parser.add_argument('--seq-dep', action='store_true')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
