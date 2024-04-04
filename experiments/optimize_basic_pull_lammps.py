from pathlib import Path
import shutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.dna2 import model, lammps_utils

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
    assert(n_eq_steps % eq_every == 0)
    n_eq_states = n_eq_steps // eq_every

    n_sample_steps = args['n_sample_steps']
    assert(n_sample_steps % sample_every == 0)
    n_sample_states = n_sample_steps // sample_every

    n_total_steps = n_eq_steps + n_sample_steps
    n_total_states = n_total_steps // sample_every
    assert(n_total_states == n_sample_states + n_eq_states)

    run_name = args['run_name']
    n_iters = args['n_iters']
    lr = args['lr']
    target_dist = args['target_dist']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']


    # Setup the logging directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    img_dir = run_dir / "img"
    img_dir.mkdir(parents=False, exist_ok=False)

    ref_traj_dir = run_dir / "ref_traj"
    ref_traj_dir.mkdir(parents=False, exist_ok=False)

    loss_path = run_dir / "loss.txt"
    times_path = run_dir / "times.txt"
    grads_path = run_dir / "grads.txt"
    neff_path = run_dir / "neff.txt"
    dist_path = run_dir / "dist.txt"
    resample_log_path = run_dir / "resample_log.txt"

    params_str = ""
    params_str += f"n_sample_states: {n_sample_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    # Load the system
    sys_basedir = Path("data/templates/lammps-stretch-tors")
    lammps_data_path = sys_basedir / "data"

    p = subprocess.Popen([tacoxdna_basedir / "src/LAMMPS_oxDNA.py", lammps_data_path], cwd=run_dir)
    p.wait()
    rc = p.returncode
    if rc != 0:
        raise RuntimeError(f"tacoxDNA conversion failed with error code: {rc}")

    init_conf_fpath = run_dir / "data.oxdna"
    assert(init_conf_fpath.exists())
    os.rename(init_conf_path, run_dir / "init.conf")

    top_fpath = run_dir / "data.top"
    assert(top_fpath.exists())
    os.rename(top_fpath, run_dir / "sys.top")

    top_info = topology.TopologyInfo(top_path, reverse_direction=False)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
    n = seq_oh.shape[0]
    assert(n % 2 == 0)
    n_bp = n // 2

    strand1_start = 0
    strand1_end = n_bp-1
    strand2_start = n_bp
    strand2_end = n_bp*2-1

    bp1_meas = [4, strand2_end-4]
    bp2_meas = [strand1_end-4, strand2_start+4]


    displacement_fn, shift_fn = FIXME

    t_kelvin = 300.0
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    salt_conc = 0.15
    q_eff = 0.815
    seq_avg = True

    def get_ref_states(params, i, seed):
        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        sim_dir = iter_dir / f"sim"
        sim_dir.mkdir(parents=False, exist_ok=False)

        shutil.copy(lammps_data_path, sim_dir / "data")
        lammps_in_fpath = sim_dir / "in"
        lammps_utils.stretch_tors_constructor(
            params, lammps_in_fpath, kT=kT, salt_conc=salt_conc, q_eff=q_eff,
            force_pn=FIXME, torque_pnnm=0.0,
            save_every=sample_every, n_steps=n_total_steps,
            seq_avg=seq_avg, seed=seed)

        sim_start = time.time()
        p = subprocess.Popen([lammps_exec_path, "-in", lammps_in_fpath], cwd=sim_dir)
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

        ## Generate an energy function
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin,
                               salt_conc=salt_conc, q_eff=q_eff)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        ## Compute the energies via our energy function
        calc_start = time.time()
        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, calc_energies = scan(energy_scan_fn, None, traj_states)
        calc_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Calculating energies took {calc_end - calc_start} seconds\n")

        ## Check energies
        gt_energies = log_df.PotEng * seq_oh.shape[0]
        energy_diffs = list()
        for calc, gt in zip(calc_energies, gt_energies):
            diff = onp.abs(calc - gt)
            energy_diffs.append(diff)

        ## Compute the mean distance
        all_distances = list()
        for rs_idx in range(n_sample_states):
            ref_state = traj_states[rs_idx]
            bp1_meas_pos = get_bp_pos(ref_state, bp1_meas)
            bp2_meas_pos = get_bp_pos(ref_state, bp2_meas)
            dist = onp.abs(bp1_meas_pos[2] - bp2_meas_pos[2])
            all_distances.append(dist)

        all_distances = onp.array(all_distances)

        ## Record some plots

        plt.plot(all_distances)
        plt.savefig(iter_dir / "distances_traj.png")
        plt.clf()

        running_avg = onp.cumsum(all_distances) / onp.arange(1, n_sample_states+1)
        plt.plot(running_avg)
        plt.savefig(iter_dir / "running_avg.png")
        plt.clf()

        sns.distplot(calc_energies, label="Calculated", color="red")
        sns.distplot(gt_energies, label="Reference", color="green")
        plt.legend()
        plt.savefig(iter_dir / f"energies.png")
        plt.clf()

        sns.histplot(energy_diffs)
        plt.savefig(iter_dir / f"energy_diffs.png")
        plt.clf()

        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"Mean distance: {onp.mean(all_distances)}\n")
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        return traj_states, calc_energies, all_distances

    # FIXME: do a single call to get_ref_states. Then test. Don't forget to add arugment for force in pN. Will need to somehow initialize override base params that we want to optimize over.



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
    parser.add_argument('--target-dist', type=float,
                        help="Target persistence length in nanometers")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
