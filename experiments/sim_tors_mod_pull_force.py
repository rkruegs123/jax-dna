"""Replicating `Compute_Extension_Torsion_oxDNA.cpp` in JCTC work."""
import pdb
from pathlib import Path
from tqdm import tqdm
import time
import numpy as onp
from copy import deepcopy
import functools
import matplotlib.pyplot as plt
import shutil
import argparse
import seaborn as sns

import jax
import jax.numpy as jnp
from jax import jit, vmap, random, lax, tree_util, pmap
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint, ext_force
from jax_dna.dna1 import model
from jax_dna.loss import pitch

from jax.config import config
config.update("jax_enable_x64", True)


checkpoint_every = 50
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
    bb1 = bb1 / jnp.linalg.norm(bb1)
    bb2 = displacement_fn(base_sites[b2], base_sites[a2])
    bb2 = bb2 / jnp.linalg.norm(bb2)

    # Compute angle *assuming* vectors lie in same x-y plane
    theta = jnp.arccos(utils.clamp(jnp.dot(bb1[:2], bb2[:2])))
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

    
def get_all_quartets(n_nucs_per_strand):
    s1_nucs = list(range(n_nucs_per_strand))
    s2_nucs = list(range(n_nucs_per_strand, n_nucs_per_strand*2))
    s2_nucs.reverse()

    bps = list(zip(s1_nucs, s2_nucs))
    n_bps = len(s1_nucs)
    all_quartets = list()
    for i in range(n_bps-1):
        bp1 = bps[i]
        bp2 = bps[i+1]
        all_quartets.append(bp1 + bp2)
    return jnp.array(all_quartets, dtype=jnp.int32)



def run(args):

    # Load arguments
    key_seed = args['key_seed']
    key = random.PRNGKey(key_seed)
    n_sims = args['n_sims']
    n_steps_per_batch = args['n_steps_per_batch']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    run_name = args['run_name']
    dt = args['dt']
    assert(n_steps_per_batch % sample_every == 0)
    spring_k = args['spring_k']
    force_pn = args['force']
    force_oxdna = force_pn / utils.oxdna_force_to_pn


    displacement_fn, shift_fn = space.free()
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    gamma = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64))
    mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)

    n_bp = 40
    strand1_start = 0
    strand1_end = n_bp-1
    strand2_start = n_bp
    strand2_end = n_bp*2-1

    bps1_harm = onp.array([onp.arange(0, strand1_start+2),
                           onp.arange(strand2_end, strand2_end-2, -1)]).T
    bp2_harm = [strand1_end-1, strand2_start+1]


    quartets = get_all_quartets(n_nucs_per_strand=n_bp)
    quartets = quartets[4:n_bp-5] # Restrict to central n_bp-10 bp

    bp1_meas = [4, strand2_end-4]
    bp2_meas = [strand1_end-4, strand2_start+4]

    rise_per_bp = 3.4 / utils.ang_per_oxdna_length # oxDNA length units
    contour_length = quartets.shape[0] * rise_per_bp # oxDNA length units

    # FIXME: use real oxDNA theta0. Maybe from a calculation of the pitch?
    # theta0 is 35 degrees per bp -- http://nanobionano.unibo.it/StrutturisticaAcNucl/BryantCozzarelliBustamanteDNATorqueMeasurement.pdf
    exp_theta0_per_bp = 35 * jnp.pi/180.0 # radians
    exp_theta0 = exp_theta0_per_bp * quartets.shape[0]

    # Load the system
    sys_basedir = Path("data/templates/tors-mod-40bp")
    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path, reverse_direction=True
    )

    init_body = conf_info.get_states()[0]
    def get_bp_pos(body, bp):
        return (body.center[bp[0]] + body.center[bp[1]]) / 2
    all_init_bp1 = vmap(get_bp_pos, (None, 0))(init_body, bps1_harm)
    init_bp2 = get_bp_pos(init_body, bp2_harm)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    
    def sim_torsional(spring_k, key, n_eq_steps, sample_every, n_steps_per_batch, batch_size):

        # Setup our force fn
        base_energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)

        def harmonic_bias(body):
            all_bp1_pos = vmap(get_bp_pos, (None, 0))(body, bps1_harm)
            all_bp1_sq_diff = (all_bp1_pos - all_init_bp1)**2
            bp1_term = all_bp1_sq_diff.sum()

            bp2_pos = get_bp_pos(body, bp2_harm)
            bp2_sq_diff = (bp2_pos - init_bp2)**2
            bp2_xy_term = bp2_sq_diff[:2].sum()
            bp2_z_term = (bp2_pos[2] - all_init_bp1[1][2])**2
            bp2_term = bp2_xy_term + bp2_z_term

            bias_val = bp1_term + bp2_term
            return 0.5*spring_k*bias_val

        def energy_fn(body, **kwargs):
            base_energy = base_energy_fn(body, **kwargs)
            bias_val = harmonic_bias(body)
            return base_energy + bias_val

        _, force_fn = ext_force.get_force_fn(
            energy_fn, top_info.n, displacement_fn,
            bp2_harm, [0, 0, force_oxdna/2], [0, 0, 0, 0])


        # Setup equilibration
        assert(n_steps_per_batch % sample_every == 0)
        num_points_per_batch = n_steps_per_batch // sample_every

        # Setup sampling simulation
        def batch_fn(batch_key):

            init_fn, step_fn = simulate.nvt_langevin(force_fn, shift_fn, dt, kT, gamma)
            init_state = init_fn(batch_key, init_body, mass=mass)

            def fori_step_fn(t, state):
                return step_fn(state)
            fori_step_fn = jit(fori_step_fn)

            @jit
            def scan_fn(state, step):
                state = lax.fori_loop(0, sample_every, fori_step_fn, state)
                return state, state.position

            eq_state = lax.fori_loop(0, n_eq_steps, fori_step_fn, init_state)
            _, traj = lax.scan(scan_fn, eq_state, jnp.arange(num_points_per_batch))
            return traj

        # Equilibrate and sample
        batch_keys = random.split(key, batch_size)
        batch_trajs = vmap(batch_fn)(batch_keys)

        num_bases = batch_trajs.center.shape[2]
        assert(batch_trajs.center.shape[3] == 3)

        combined_center = batch_trajs.center.reshape(-1, num_bases, 3)
        combined_quat_vec = batch_trajs.orientation.vec.reshape(-1, num_bases, 4)

        combined_traj = rigid_body.RigidBody(
            center=combined_center,
            orientation=rigid_body.Quaternion(combined_quat_vec))

        return combined_traj



    # Setup the logging directoroy
    if run_name is None:
        raise RuntimeError(f"Must set a run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    shutil.copy(top_path, run_dir / "sys.top")

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    start = time.time()
    traj = sim_torsional(spring_k, key, n_eq_steps, sample_every, n_steps_per_batch, n_sims)
    end = time.time()
    print(f"- Simulation finished: {end - start} seconds\n")


    # Write all the trajectories (after we've done relevant analysis)
    traj_info = trajectory.TrajectoryInfo(
        top_info, read_from_states=True, states=traj, box_size=conf_info.box_size)
    traj_info.write(run_dir / f"traj.dat", reverse=True)


    # Compute the torsional modulus
    def compute_theta(body):
        pitches = compute_pitches(body, quartets, displacement_fn, model.com_to_hb)
        return pitches.sum()
    compute_theta = jit(compute_theta)

    fjoules_per_oxdna_energy = utils.joules_per_oxdna_energy * 1e15 # 1e15 fJ per J
    fm_per_oxdna_length = utils.ang_per_oxdna_length * 1e5 # 1e5 fm per Angstrom
    all_theta = list()
    all_theta0 = list()
    running_avg_freq = 10
    min_rs_idx = 50 # minimum idx for running average
    all_c_fjfm = list()

    n_ref_states = traj.center.shape[0]

    for rs_idx in tqdm(range(n_ref_states)):
        ref_state = traj[rs_idx]
        theta = compute_theta(ref_state)
        all_theta.append(theta)
        theta0 = onp.mean(all_theta)
        all_theta0.append(theta0)

        if rs_idx % running_avg_freq == 0 and rs_idx > min_rs_idx:

            # curr_theta0 = onp.mean(all_theta)
            # dtheta = onp.array(all_theta) - curr_theta0
            # dtheta_sqr = dtheta**2

            # avg_dtheta_sqr = onp.mean(dtheta_sqr)
            avg_dtheta_sqr = onp.var(all_theta)

            C_oxdna = (kT * contour_length) / avg_dtheta_sqr # oxDNA units
            C_fjfm = C_oxdna * fjoules_per_oxdna_energy * fm_per_oxdna_length
            all_c_fjfm.append(C_fjfm)


    all_theta = onp.array(all_theta)
    all_theta0 = onp.array(all_theta0)
    theta0 = onp.mean(all_theta0)

    dtheta = all_theta - theta0
    dtheta_sqr = dtheta**2

    ## Plot theta trajectory
    plt.plot(all_theta)
    plt.ylabel("Theta")
    plt.axhline(y=theta0, linestyle="--", label="Theta0")
    plt.legend()
    plt.savefig(run_dir / "theta_traj.png")
    plt.clf()

    ## Plot running average of theta0
    plt.plot(all_theta0, label="Running Avg.")
    plt.ylabel("Theta0")
    plt.axhline(y=exp_theta0, linestyle="--", label="Expected Theta0")
    plt.legend()
    plt.savefig(run_dir / "theta0_running_avg.png")
    plt.clf()


    ## Plot running average of C
    plt.plot(all_c_fjfm)
    plt.ylabel("C (fJfm)")
    plt.title("C running average")
    plt.savefig(run_dir / "c_running_avg.png")
    plt.clf()

    ## Plot the distribution of thetas
    sns.histplot(dtheta)
    plt.savefig(run_dir / f"dtheta_dist.png")
    plt.clf()

    sns.histplot(dtheta_sqr)
    plt.savefig(run_dir / f"dtheta_sqr_dist.png")
    plt.clf()

    ## Compute the final C
    avg_dtheta_sqr = onp.mean(dtheta_sqr)
    C_oxdna = (kT * contour_length) / avg_dtheta_sqr # oxDNA units
    C_fjfm = C_oxdna * fjoules_per_oxdna_energy * fm_per_oxdna_length


    summary_str = f"Avg. dtheta sqr: {avg_dtheta_sqr}\n"
    summary_str += f"Avg. dtheta: {onp.mean(dtheta)}\n"
    summary_str += f"C (oxDNA units): {C_oxdna}\n"
    summary_str += f"C (fJfm): {C_fjfm}\n"
    summary_str += f"Theta_0: {theta0}\n"
    with open(run_dir / "summary.txt", "w+") as f:
        f.write(summary_str)


def get_parser():
    parser = argparse.ArgumentParser(description="Simulate torsional modulus with x- and y- constrained")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--n-sims', type=int, default=2,
                        help="Number of simulations.")
    parser.add_argument('--n-steps-per-batch', type=int, default=int(1e7),
                        help="Number of total steps (post-equilibration) for sampling per batch.")
    parser.add_argument('--n-eq-steps', type=int, help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=int(1e4),
                        help="Frequency of sampling reference states.")
    parser.add_argument('--key-seed', type=int, default=0, help="Integer seed for key")
    parser.add_argument('--spring-k', type=float, default=10.0,
                        help="Spring constant for the harmonic bias")
    parser.add_argument('--force', type=float, default=10.0,
                        help="Pulling force in pN")
    parser.add_argument('--dt', type=float, default=5e-3,
                        help="Timestep")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
