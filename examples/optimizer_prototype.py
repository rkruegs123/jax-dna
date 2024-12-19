import functools
import time

import cloudpickle
import jax
import jax.numpy as jnp
import jax_md
import ray
import ray.runtime_env
from jax import export

import jax_dna.input.topology as topology
import jax_dna.input.trajectory as trajectory
import jax_dna.input.toml as toml_reader
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.energy.base as jdna_energy
import jax_dna.energy.configuration as jdna_energy_config
import jax_dna.losses.observable_wrappers as jdna_losses
import jax_dna.observables as jd_obs
import jax_dna.utils.types as jdt
import jax_dna.simulators.jax_md as jmd
import jax_dna.simulators.io as jd_sio
from examples import optimizer_prototype_serial


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/home/ryan/repos/jax-dna/examples/fn_cache")

def main():
    topology_fname = "data/sys-defs/simple-helix/sys.top"
    traj_fname = "data/sys-defs/simple-helix/bound_relaxed.conf"
    simulation_config = "jax_dna/input/dna1/default_simulation.toml"
    energy_config = "jax_dna/input/dna1/default_energy.toml"

    top = topology.from_oxdna_file(topology_fname)
    seq = jnp.array(top.seq_one_hot)
    traj = trajectory.from_file(
        traj_fname,
        top.strand_counts,
    )
    experiment_config = toml_reader.parse_toml(simulation_config)
    energy_config = toml_reader.parse_toml(energy_config)

    displacement_fn, shift_fn = jax_md.space.free()
    key = jax.random.PRNGKey(0)

    # change to R
    init_body = traj.states[0].to_rigid_body()

    dt = experiment_config["dt"]
    kT = experiment_config["kT"]
    diff_coef = experiment_config["diff_coef"]
    rot_diff_coef = experiment_config["rot_diff_coef"]

    gamma = jax_md.rigid_body.RigidBody(
        center=jnp.array([kT/diff_coef], dtype=jnp.float64),
        orientation=jnp.array([kT/rot_diff_coef], dtype=jnp.float64),
    )
    mass = jax_md.rigid_body.RigidBody(
        center=jnp.array([experiment_config["nucleotide_mass"]], dtype=jnp.float64),
        orientation=jnp.array([experiment_config["moment_of_inertia"]], dtype=jnp.float64),
    )

    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        dna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    configs = [
        # single param
        dna1_energy.FeneConfiguration.from_dict(energy_config["fene"], ("eps_backbone",)),
        # multiple params
        dna1_energy.BondedExcludedVolumeConfiguration.from_dict(energy_config["bonded_excluded_volume"], ("eps_exc","dr_star_base")),
        # shortcut for all params, though you could list them all too
        dna1_energy.StackingConfiguration.from_dict(energy_config["stacking"] | {"kt": kT }, ("*",)),
        dna1_energy.UnbondedExcludedVolumeConfiguration.from_dict(energy_config["unbonded_excluded_volume"]),
        dna1_energy.HydrogenBondingConfiguration.from_dict(energy_config["hydrogen_bonding"]),
        dna1_energy.CrossStackingConfiguration.from_dict(energy_config["cross_stacking"]),
        dna1_energy.CoaxialStackingConfiguration.from_dict(energy_config["coaxial_stacking"]),
    ]

    opt_params = [c.opt_params for c in configs]

    # configs and energy functions should be in the same order
    # though maybe we can be more clever in the future
    energy_fns = [
        dna1_energy.Fene,
        dna1_energy.BondedExcludedVolume,
        dna1_energy.Stacking,
        dna1_energy.UnbondedExcludedVolume,
        dna1_energy.HydrogenBonding,
        dna1_energy.CrossStacking,
        dna1_energy.CoaxialStacking,
    ]

    sampler = jmd.JaxMDSimulator(
        energy_configs=configs,
        energy_fns=energy_fns,
        topology=top,
        simulator_params=jmd.StaticSimulatorParams(
            seq=seq,
            mass=mass,
            bonded_neighbors=top.bonded_neighbors,
            n_steps=5_000,
            checkpoint_every=0,
            dt=dt,
            kT=kT,
            gamma=gamma,
        ),
        space=(displacement_fn, shift_fn),
        transform_fn=transform_fn,
        simulator_init=jax_md.simulate.nvt_langevin,
        neighbors=jmd.NoNeighborList(unbonded_nbrs=top.unbonded_neighbors),
    )


    def sim_fn(opt_params):
        prop_twist = jd_obs.propeller.PropellerTwist(
            rigid_body_transform_fn=transform_fn,
            h_bonded_base_pairs=jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]]),
        )

        def curr_f(opts):
            sim_traj, sim_meta = sampler.run(opts, init_body, 5_000, key)
            return prop_twist(sim_traj).mean(), (sim_traj, sim_meta)

        j, t = jax.jacfwd(curr_f, has_aux=True)(opt_params)

        return j, t


    ray.init(runtime_env={
        "env_vars": {"JAX_ENABLE_X64": "true"},
        "py_modules":[optimizer_prototype_serial],
    })

    exported_f = export.export(jax.jit(sim_fn))(opt_params)
    serialized_f: bytearray = exported_f.serialize()
    gettable_f = ray.put(serialized_f)

    def wrapped_fn(opt_params):
        jax.config.update(
            "jax_compilation_cache_dir",
            "/home/ryan/repos/jax-dna/examples/fn_cache"
        )
        import sys
        if "examples.optimizer_prototype_serial" not in sys.modules:
            from examples import optimizer_prototype_serial


        return export.deserialize(ray.get(gettable_f)).call(opt_params)


    n_local_runs = 3
    n_remote_runs = 3
    n_reps_parallel_runs = 3, 2

    for i in range(n_local_runs):
        print("Local run", i, "=======================================================")
        start = time.time()
        _ = wrapped_fn(opt_params)[1][0].rigid_body.center.block_until_ready()
        print("time: ", time.time() - start)

    remote_simfn = ray.remote(wrapped_fn)
    remote_simfn = remote_simfn.options()

    for i in range(n_remote_runs):
        print("Remote run", i, "=======================================================")
        start = time.time()
        result = remote_simfn.remote(opt_params)
        _ = ray.get(result)
        print("time: ", time.time() - start)


    n_reps, n_jobs = n_reps_parallel_runs
    for i in range(n_reps):
        print(f"Parallel {n_jobs} runs {i} ==================================================")
        start = time.time()
        result = ray.get([remote_simfn.remote(opt_params) for _ in range(n_jobs)])
        print("time: ", time.time() - start)


if __name__=="__main__":
    main()
