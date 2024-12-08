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
    n_iters = args['n_iters']
    lr = args['lr']
    rmse_coeff = args['rmse_coeff']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']
    seq_avg = not args['seq_dep']
    assert(seq_avg)

    force_low_pn = args['force_low_pn']
    force_high_pn = args['force_high_pn']
    target_slope = args['target_slope']


    # Setup the logging directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    img_dir = run_dir / "img"
    img_dir.mkdir(parents=False, exist_ok=False)

    log_dir = run_dir / "log"
    log_dir.mkdir(parents=False, exist_ok=False)

    ref_traj_dir = run_dir / "ref_traj"
    ref_traj_dir.mkdir(parents=False, exist_ok=False)

    loss_path = log_dir / "loss.txt"
    times_path = log_dir / "times.txt"
    grads_path = log_dir / "grads.txt"
    neff_lo_path = log_dir / "neff_lo.txt"
    neff_hi_path = log_dir / "neff_hi.txt"
    slope_path = log_dir / "slope.txt"
    resample_log_path = log_dir / "resample_log.txt"
    iter_params_path = log_dir / "iter_params.txt"

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


    def run_sim(sim_dir, params, force_pn, seed):

        shutil.copy(lammps_data_abs_path, sim_dir / "data")
        lammps_in_fpath = sim_dir / "in"
        lammps_utils.stretch_tors_constructor(
            params, lammps_in_fpath, kT=kT, salt_conc=salt_conc, qeff=q_eff,
            force_pn=force_pn, torque_pnnm=0.0,
            save_every=sample_every, n_steps=n_total_steps,
            seq_avg=seq_avg, seed=seed)

        sim_start = time.time()
        p = subprocess.Popen([lammps_exec_path, "-in", lammps_in_fpath.stem], cwd=sim_dir)
        p.wait()
        rc = p.returncode
        if rc != 0:
            raise RuntimeError(f"LAMMPS simulation failed with error code: {rc}")
        sim_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Simulation took {sim_end - sim_start} seconds\n")

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
        with open(resample_log_path, "a") as f:
            f.write(f"- Loading took {load_end - load_start} seconds\n")

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
        with open(resample_log_path, "a") as f:
            f.write(f"- Calculating energies took {calc_end - calc_start} seconds\n")

        ## Check energies
        gt_energies = (log_df.PotEng * seq_oh.shape[0]).to_numpy()
        energy_diffs = list()
        for idx, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
            diff = onp.abs(calc - gt)
            energy_diffs.append(diff)


        ## Compute the mean theta
        all_thetas = list()
        for rs_idx in range(n_sample_states):
            ref_state = traj_states[rs_idx]
            theta = compute_theta(ref_state)
            all_thetas.append(theta)

        all_thetas = onp.array(all_thetas)

        ## Record some plots

        plt.plot(all_thetas)
        plt.savefig(sim_dir / "theta_traj.png")
        plt.clf()

        running_avg = onp.cumsum(all_thetas) / onp.arange(1, n_sample_states+1)
        plt.plot(running_avg)
        plt.savefig(sim_dir / "running_avg.png")
        plt.clf()

        sns.distplot(calc_energies, label="Calculated", color="red")
        sns.distplot(gt_energies, label="Reference", color="green")
        plt.legend()
        plt.savefig(sim_dir / f"energies.png")
        plt.clf()

        sns.histplot(energy_diffs)
        plt.savefig(sim_dir / f"energy_diffs.png")
        plt.clf()

        with open(sim_dir / "summary.txt", "w+") as f:
            f.write(f"Mean theta: {onp.mean(all_thetas)}\n")
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")

        return traj_states, calc_energies, all_thetas


    def get_ref_states(params, i, seed):
        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        sim_lo_dir = iter_dir / f"sim-lo"
        sim_lo_dir.mkdir(parents=False, exist_ok=False)
        traj_states_lo, calc_energies_lo, all_thetas_lo = run_sim(sim_lo_dir, params, force_low_pn, seed)

        sim_hi_dir = iter_dir / f"sim-hi"
        sim_hi_dir.mkdir(parents=False, exist_ok=False)
        traj_states_hi, calc_energies_hi, all_thetas_hi = run_sim(sim_hi_dir, params, force_high_pn, seed)


        running_avg_lo = onp.cumsum(all_thetas_lo) / onp.arange(1, n_sample_states+1)
        running_avg_hi = onp.cumsum(all_thetas_hi) / onp.arange(1, n_sample_states+1)

        # slope is (theta0_hi - theta0_low) / (force_hi - force_low)
        running_avg_diff = (running_avg_hi - running_avg_lo) * (1000 / 36)
        running_avg_slope = running_avg_diff / (force_high_pn - force_low_pn)
        plt.plot(running_avg_slope)
        plt.savefig(iter_dir / "running_avg_slope.png")
        plt.clf()

        theta0_lo = onp.mean(all_thetas_lo)
        theta0_hi = onp.mean(all_thetas_hi)
        theta0_diff = (theta0_hi - theta0_lo) * (1000 / 36)
        fin_slope = theta0_diff / (force_high_pn - force_low_pn)

        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"Final slope: {fin_slope}\n")

        return traj_states_lo, calc_energies_lo, all_thetas_lo, traj_states_hi, calc_energies_hi, all_thetas_hi


    @jit
    def loss_fn(params, ref_states_lo, ref_energies_lo, ref_thetas_lo, ref_states_hi, ref_energies_hi, ref_thetas_hi):

        # Setup energy function
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
        energy_scan_fn = lambda state, rs: (None, energy_fn(rs))

        # Compute expected theta at low forces
        _, new_energies_lo = scan(energy_scan_fn, None, ref_states_lo)

        diffs_lo = new_energies_lo - ref_energies_lo # element-wise subtraction
        boltzs_lo = jnp.exp(-beta * diffs_lo)
        denom_lo = jnp.sum(boltzs_lo)
        weights_lo = boltzs_lo / denom_lo

        expected_theta_lo = jnp.dot(weights_lo, ref_thetas_lo)

        # Compute expected theta at low forces
        _, new_energies_hi = scan(energy_scan_fn, None, ref_states_hi)

        diffs_hi = new_energies_hi - ref_energies_hi # element-wise subtraction
        boltzs_hi = jnp.exp(-beta * diffs_hi)
        denom_hi = jnp.sum(boltzs_hi)
        weights_hi = boltzs_hi / denom_hi

        expected_theta_hi = jnp.dot(weights_hi, ref_thetas_hi)

        # Get final loss
        expected_theta_diff = (expected_theta_hi - expected_theta_lo) * (1000 / 36)
        expected_slope = expected_theta_diff / (force_high_pn - force_low_pn)
        mse = (expected_slope - target_slope)**2
        rmse = jnp.sqrt(mse)

        # weights = jnp.concatenate([weights_lo, weights_hi])
        n_eff_lo = jnp.exp(-jnp.sum(weights_lo * jnp.log(weights_lo)))
        n_eff_hi = jnp.exp(-jnp.sum(weights_hi * jnp.log(weights_hi)))

        return rmse_coeff*rmse, (n_eff_lo, n_eff_hi, expected_slope)

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    params = deepcopy(model.EMPTY_BASE_PARAMS)
    if seq_avg:
        params["stacking"] = deepcopy(model.default_base_params_seq_avg["stacking"])
        params["hydrogen_bonding"] = deepcopy(model.default_base_params_seq_avg["hydrogen_bonding"])
        params["cross_stacking"] = deepcopy(model.default_base_params_seq_avg["cross_stacking"])
        del params["cross_stacking"]["dr_c_cross"]
        del params["cross_stacking"]["dr_low_cross"]
        del params["cross_stacking"]["dr_high_cross"]
    else:
        params["stacking"] = deepcopy(model.default_base_params_seq_dep["stacking"])
        params["hydrogen_bonding"] = deepcopy(model.default_base_params_seq_dep["hydrogen_bonding"])
        params["cross_stacking"] = deepcopy(model.default_base_params_seq_dep["cross_stacking"])
        del params["cross_stacking"]["dr_c_cross"]
        del params["cross_stacking"]["dr_low_cross"]
        del params["cross_stacking"]["dr_high_cross"]

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # min_n_eff = int(2*n_sample_states * min_neff_factor) # 2x because we simulate at two different forces
    min_n_eff = int(n_sample_states * min_neff_factor) # 2x because we simulate at two different forces

    all_losses = list()
    all_slopes = list()
    all_n_effs_lo = list()
    all_n_effs_hi = list()
    all_ref_losses = list()
    all_ref_slopes = list()
    all_ref_times = list()

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")

    start = time.time()
    ref_states_lo, ref_energies_lo, ref_thetas_lo, ref_states_hi, ref_energies_hi, ref_thetas_hi = get_ref_states(params, i=0, seed=30362)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")

    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()

        (loss, (n_eff_lo, n_eff_hi, curr_slope)), grads = grad_fn(
            params, ref_states_lo, ref_energies_lo, ref_thetas_lo,
            ref_states_hi, ref_energies_hi, ref_thetas_hi)
        num_resample_iters += 1

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_slopes.append(curr_slope)

        if n_eff_lo < min_n_eff or n_eff_hi < min_n_eff or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0
            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- n_eff_hi was {n_eff_hi}...")
                f.write(f"- n_eff_lo was {n_eff_lo}. Resampling...\n")

            start = time.time()
            ref_states_lo, ref_energies_lo, ref_thetas_lo, ref_states_hi, ref_energies_hi, ref_thetas_hi = get_ref_states(params, i=i, seed=i)
            end = time.time()
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (loss, (n_eff_lo, n_eff_hi, curr_slope)), grads = grad_fn(
                params, ref_states_lo, ref_energies_lo, ref_thetas_lo,
                ref_states_hi, ref_energies_hi, ref_thetas_hi)

            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_slopes.append(curr_slope)


        iter_end = time.time()

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(neff_lo_path, "a") as f:
            f.write(f"{n_eff_lo}\n")
        with open(neff_hi_path, "a") as f:
            f.write(f"{n_eff_hi}\n")
        with open(slope_path, "a") as f:
            f.write(f"{curr_slope}\n")
        with open(times_path, "a") as f:
            f.write(f"{iter_end - iter_start}\n")
        with open(iter_params_path, "a") as f:
            f.write(f"{pprint.pformat(params)}\n")

        all_losses.append(loss)
        all_n_effs_lo.append(n_eff_lo)
        all_n_effs_hi.append(n_eff_hi)
        all_slopes.append(curr_slope)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)


        plt.plot(onp.arange(i+1), all_losses, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points", color="blue")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"losses_iter{i}.png")
        plt.clf()


        plt.plot(onp.arange(i+1), all_slopes, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_slopes, marker='o', label="Resample points", color="blue")
        plt.axhline(y=target_slope, linestyle='--', label="Target Slope", color='red')
        plt.legend()
        plt.ylabel("Expected Slope")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"slopes_iter{i}.png")
        plt.clf()

def get_parser():

    parser = argparse.ArgumentParser(description="Optimize the length under a pulling force via LAMMPS")

    parser.add_argument('--n-sample-steps', type=int, default=3000000,
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
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")

    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")

    parser.add_argument('--force-low-pn', type=float, default=2.5,
                        help="Total low pulling force in pN")
    parser.add_argument('--force-high-pn', type=float, default=10,
                        help="Total high pulling force in pN")
    parser.add_argument('--target-slope', type=float, default=-0.05,
                        help="Target slope")
    parser.add_argument('--rmse-coeff', type=float, default=100.0,
                        help="Coefficient for the RMSE")
    parser.add_argument('--seq-dep', action='store_true')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
