import pdb
from pathlib import Path
from copy import deepcopy
import pprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp
import subprocess
import pandas as pd
import shutil
import seaborn as sns

import jax
import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax
from jax_md import space, rigid_body

from jax_dna.common import utils, topology, trajectory, center_configuration
from jax_dna.loss import geometry, pitch, propeller
from jax_dna.dna1 import model, oxdna_utils

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)




def run(args, oxdna_path, num_threads=4):
    # Load parameters
    n_iters = args['n_iters']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    lr = args['lr']
    max_approx_iters = args['max_approx_iters']
    min_neff_factor = args['min_neff_factor']
    n_sample_steps = args['n_sample_steps']
    assert(n_sample_steps % sample_every == 0)
    n_ref_states = n_sample_steps // sample_every
    plot_every = args['plot_every']
    run_name = args['run_name']
    target_ptwist = args['target_ptwist']
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"

    # Setup the logging directoroy
    if run_name is None:
        raise RuntimeError(f"Must set a run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    img_dir = run_dir / "img"
    img_dir.mkdir(parents=False, exist_ok=False)

    ref_traj_dir = run_dir / "ref_traj"
    ref_traj_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "run_params.txt", "w+") as f:
        f.write(params_str)

    # Load the system
    sys_basedir = Path("data/templates/simple-helix")
    input_template_path = sys_basedir / "input"

    top_path = sys_basedir / "sys.top"
    # top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    top_info = topology.TopologyInfo(top_path, reverse_direction=False)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        # reverse_direction=True
        reverse_direction=False

    )
    box_size = conf_info.box_size

    # displacement_fn, shift_fn = space.periodic(box_size)
    displacement_fn, shift_fn = space.free()

    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT

    # Infrastructure to generate reference states
    def get_ref_states(params, init_conf_path, i, seed):
        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        shutil.copy(top_path, iter_dir / "sys.top")
        init_conf_info = trajectory.TrajectoryInfo(
            top_info,
            read_from_file=True, traj_path=init_conf_path,
            # reverse_direction=True
            reverse_direction=False
        )
        centered_init_conf_info = center_configuration.center_conf(top_info, init_conf_info)
        centered_init_conf_info.write(iter_dir / "init.conf", reverse=False, write_topology=False)
        # shutil.copy(init_conf_path, iter_dir / "init.conf")

        # recompile oxDNA with the new parameters
        oxdna_utils.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=num_threads)

        # write input file for oxDNA simulation
        oxdna_utils.rewrite_input_file(
            input_template_path, iter_dir,
            temp=f"{t_kelvin}K", steps=n_sample_steps,
            init_conf_path=str(iter_dir / "init.conf"),
            top_path=str(iter_dir / "sys.top"),
            save_interval=sample_every, seed=seed,
            equilibration_steps=n_eq_steps, dt=dt
        )
        input_path = iter_dir / "input"

        # run oxDNA simulation
        oxdna_process = subprocess.run([oxdna_exec_path, input_path])
        rc = oxdna_process.returncode
        if rc != 0:
            raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")

        # Load states from oxDNA simulation
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True,
            traj_path=iter_dir / "output.dat",
            # reverse_direction=True)
            reverse_direction=False)
        traj_states = traj_info.get_states()
        traj_states = utils.tree_stack(traj_states)

        # Load the oxDNA energies
        energy_path = iter_dir / "energy.dat"
        energy_df = pd.read_csv(
            energy_path,
            names=["time", "potential_energy", "kinetic_energy", "total_energy"],
            delim_whitespace=True)

        # Generate an energy function
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)

        # Check energies
        calc_energies = vmap(energy_fn)(traj_states)
        gt_energies = energy_df.iloc[1:, :].potential_energy.to_numpy() * seq_oh.shape[0]

        atol_places = 3
        energy_diffs = list()
        for calc, gt in zip(calc_energies, gt_energies):
            diff = onp.abs(calc - gt)
            energy_diffs.append(diff)
            """
            if diff >= 10**(-atol_places):
                pdb.set_trace()
                raise RuntimeError(f"Predicted energies do not match")
            """

        unweighted_ptwists = vmap(compute_avg_ptwist)(traj_states)


        # Logging
        ptwist_running_avg = [jnp.mean(unweighted_ptwists[:i]) for i in range (1, n_ref_states+1)]
        plt.plot(ptwist_running_avg)
        plt.title("Prop. Twist Running Average")
        plt.ylabel("Prop. Twist")
        plt.xlabel("Sample")
        plt.savefig(iter_dir / f"running_avg.png")
        plt.clf()

        # Plot histogram of ptwists
        sns.histplot(unweighted_ptwists)
        plt.savefig(iter_dir / f"ptwists.png")
        plt.clf()

        # Plot the energy differences
        sns.histplot(energy_diffs)
        plt.savefig(iter_dir / f"energy_diffs.png")
        plt.clf()

        # Plot the energies
        sns.distplot(calc_energies, label="Calculated", color="red")
        sns.distplot(gt_energies, label="Reference", color="green")
        plt.legend()
        plt.savefig(iter_dir / f"energies.png")
        plt.clf()

        # Record the loss
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        # Return states and energies
        return traj_states, calc_energies, iter_dir, unweighted_ptwists

    # Construct the loss function terms

    ## note: we don't include the end base pairs due to fraying
    compute_helical_diameters, helical_diam_loss_fn = geometry.get_helical_diameter_loss_fn(
        top_info.bonded_nbrs[1:-1], displacement_fn, model.com_to_backbone)

    compute_bb_distances, bb_dist_loss_fn = geometry.get_backbone_distance_loss_fn(
        top_info.bonded_nbrs, displacement_fn, model.com_to_backbone)

    simple_helix_quartets = jnp.array([
        [1, 14, 2, 13], [2, 13, 3, 12],
        [3, 12, 4, 11], [4, 11, 5, 10],
        [5, 10, 6, 9]])
    compute_avg_pitch, pitch_loss_fn = pitch.get_pitch_loss_fn(
        simple_helix_quartets, displacement_fn, model.com_to_hb)

    simple_helix_bps = jnp.array([[1, 14], [2, 13], [3, 12],
                                  [4, 11], [5, 10], [6, 9]])
    compute_avg_ptwist, ptwist_loss_fn = propeller.get_propeller_loss_fn(simple_helix_bps)

    @jit
    def loss_fn(params, ref_states: rigid_body.RigidBody, ref_energies, unweighted_ptwists):

        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)

        # Compute the weights
        energy_fn = lambda body: em.energy_fn(body,
                                              seq=seq_oh,
                                              bonded_nbrs=top_info.bonded_nbrs,
                                              unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)
        new_energies = vmap(energy_fn)(ref_states)
        diffs = new_energies - ref_energies # element-wise subtraction
        boltzs = jnp.exp(-beta * diffs)
        denom = jnp.sum(boltzs)
        weights = boltzs / denom

        # Compute the weighted observable
        weighted_ptwists = weights * unweighted_ptwists # element-wise multiplication
        expected_ptwist = jnp.sum(weighted_ptwists)

        # Compute effective sample size
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        return (expected_ptwist - target_ptwist)**2, (n_eff, expected_ptwist)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Setup the optimization
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    # params["fene"] = model.DEFAULT_BASE_PARAMS["fene"]
    params["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    ref_states, ref_energies, curr_ref_dir, unweighted_ptwists = get_ref_states(params, conf_path, i=0, seed=0)

    min_n_eff = int(n_ref_states * min_neff_factor)
    all_losses = list()
    all_eptwists = list()
    all_ref_losses = list()
    all_ref_eptwists = list()
    all_ref_times = list()

    loss_path = run_dir / "loss.txt"
    grads_path = run_dir / "grads.txt"
    params_per_iter_path = run_dir / "params_per_iter.txt"

    # Do the thing
    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        (loss, (n_eff, expected_ptwist)), grads = grad_fn(params, ref_states, ref_energies, unweighted_ptwists)
        num_resample_iters += 1

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_eptwists.append(expected_ptwist)
            with open(curr_ref_dir / "summary.txt", "a") as f:
                f.write(f"Loss: {loss}\n")

        if n_eff < min_n_eff or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0

            print(f"Resampling reference states...")

            prev_lastconf_path = curr_ref_dir / "last_conf.dat"
            ref_states, ref_energies, curr_ref_dir, unweighted_ptwists = get_ref_states(params, prev_lastconf_path, i=i, seed=i)

            (loss, (n_eff, expected_ptwist)), grads = grad_fn(params, ref_states, ref_energies, unweighted_ptwists)

            all_ref_losses.append(loss)
            all_ref_eptwists.append(expected_ptwist)
            all_ref_times.append(i)

            with open(curr_ref_dir / "summary.txt", "a") as f:
                f.write(f"Loss: {loss}\n")

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(params_per_iter_path, "a") as f:
            f.write(f"{pprint.pformat(params)}\n")
        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(grads)}\n")
        all_losses.append(loss)
        all_eptwists.append(expected_ptwist)

        print(f"Loss: {loss}")
        print(f"Effective sample size: {n_eff}")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if i % plot_every == 0:

            # Plot the losses
            plt.plot(onp.arange(i+1), all_losses, linestyle="--")
            plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"DiffTRE Propeller Twist Optimization, Neff factor={min_neff_factor}")
            plt.savefig(img_dir / f"losses_iter{i}.png")
            plt.clf()

            # Plot the persistence lengths
            plt.plot(onp.arange(i+1), all_eptwists, linestyle="--", color='blue')
            plt.scatter(all_ref_times, all_ref_eptwists, marker='o', label="Resample points", color='blue')
            plt.axhline(y=target_ptwist, linestyle='--', label="Target p. twist", color='red')
            plt.xlabel("Iteration")
            plt.ylabel("Expected Propeller Twist (deg)")
            plt.legend()
            plt.title(f"DiffTRE Propeller Twist Optimization, Neff factor={min_neff_factor}")
            plt.savefig(img_dir / f"eptwists_iter{i}.png")
            plt.clf()


    # Plot the losses
    plt.plot(all_losses, linestyle="--")
    plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"DiffTRE Propeller Twist Optimization, Neff factor={min_neff_factor}")
    plt.savefig(img_dir / f"final_losses.png")
    plt.clf()

    # Plot the persistence lengths
    plt.plot(all_eptwists, linestyle="--", color='blue')
    plt.scatter(all_ref_times, all_ref_eptwists, marker='o', label="Resample points", color='blue')
    plt.axhline(y=target_ptwist, linestyle='--', label="Target p. twist", color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Expected Propeller Twist (deg)")
    plt.legend()
    plt.title(f"DiffTRE Propeller Twist Optimization, Neff factor={min_neff_factor}")
    plt.savefig(img_dir / f"final_eptwists.png")
    plt.clf()

    return



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize structural properties using differentiable trajectory reweighting")

    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--n-eq-steps', type=int, default=10000,
                        help="Number of equilibration steps")
    parser.add_argument('--n-sample-steps', type=int, default=100000,
                        help="Number of total steps for sampling reference states")
    parser.add_argument('--plot-every', type=int, default=10,
                        help="Frequency of plotting data from gradient descent epochs")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")
    parser.add_argument('--target-ptwist', type=float, default=propeller.TARGET_PROPELLER_TWIST,
                        help="Target persistence length in degrees")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")

    args = vars(parser.parse_args())

    oxdna_path = Path("/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/")
    run(args, oxdna_path)
