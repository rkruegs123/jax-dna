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
import random

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
    random.seed(args['seed'])

    lammps_basedir = Path(args['lammps_basedir'])
    assert(lammps_basedir.exists())
    lammps_exec_path = lammps_basedir / "build/lmp"
    assert(lammps_exec_path.exists())

    tacoxdna_basedir = Path(args['tacoxdna_basedir'])
    assert(tacoxdna_basedir.exists())

    sample_every = args['sample_every']
    n_sims = args['n_sims']

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

    def stretch_torsion(params):

        sim_dir = run_dir / f"sim"
        sim_dir.mkdir(parents=False, exist_ok=False)

        base_params = model.get_full_base_params(params, seq_avg=seq_avg)

        procs = list()
        for r in range(n_sims):
            repeat_dir = sim_dir / f"r{r}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            repeat_seed = random.randrange(100)

            shutil.copy(lammps_data_abs_path, repeat_dir / "data")
            lammps_in_fpath = repeat_dir / "in"
            lammps_utils.stretch_tors_constructor(
                base_params, lammps_in_fpath, kT=kT, salt_conc=salt_conc, qeff=q_eff,
                force_pn=force_pn, torque_pnnm=torque_pnnm,
                save_every=sample_every, n_steps=n_total_steps,
                seq_avg=seq_avg, seed=repeat_seed)

            procs.append(subprocess.Popen([lammps_exec_path, "-in", lammps_in_fpath.stem], cwd=repeat_dir))

        for p in procs:
            p.wait()

        for p in procs:
            rc = p.returncode
            if rc != 0:
                raise RuntimeError(f"LAMMPS simulation failed with error code: {rc}")

        # Convert via TaxoxDNA
        for r in range(n_sims):
            repeat_dir = sim_dir / f"r{r}"
            p = subprocess.Popen([tacoxdna_basedir / "src/LAMMPS_oxDNA.py", "data", "filename.dat"], cwd=repeat_dir)
            p.wait()
            rc = p.returncode
            if rc != 0:
                raise RuntimeError(f"tacoxDNA conversion failed with error code: {rc}")


        combine_cmd = "cat "
        for r in range(n_sims):
            repeat_dir = sim_dir / f"r{r}"
            combine_cmd += f"{repeat_dir}/data.oxdna "
        combine_cmd += f"> {sim_dir}/output.dat"
        combine_proc = subprocess.run(combine_cmd, shell=True)
        if combine_proc.returncode != 0:
            raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")


        # Analyze

        ## Load states from oxDNA simulation
        load_start = time.time()
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True,
            traj_path=sim_dir / "output.dat",
            # reverse_direction=True)
            reverse_direction=False)
        full_traj_states = traj_info.get_states()
        assert(len(full_traj_states) == (1+n_total_states)*n_sims)
        sim_freq = 1+n_total_states
        traj_states = list()
        for r in range(n_sims):
            sim_states = full_traj_states[r*sim_freq:(r+1)*sim_freq]
            sampled_sim_states = sim_states[1+n_eq_states:]
            assert(len(sampled_sim_states) == n_sample_states)
            traj_states.append(sampled_sim_states)
        assert(len(traj_states) == n_sample_states*n_sims)
        traj_states = utils.tree_stack(traj_states)
        load_end = time.time()

        ## Load the LAMMPS energies
        log_dfs = list()
        for r in range(n_sims):
            repeat_dir = sim_dir / f"r{r}"
            log_path = repeat_dir / "log.lammps"
            rpt_log_df = lammps_utils.read_log(log_path)
            assert(rpt_log_df.shape[0] == n_total_states+1)
            rpt_log_df = rpt_log_df[1+n_eq_states:]
            log_dfs.append(rpt_log_df)
        log_df = pd.concat(log_dfs, ignore_index=True)

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

    params = deepcopy(model.EMPTY_BASE_PARAMS)
    start = time.time()
    stretch_torsion(params)
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
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('--n-sims', type=int, default=2,
                        help="Number of simulations")
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
