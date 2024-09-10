import functools

import jax
import jax.numpy as jnp
import jax_md
import jax_dna.input.topology as topology
import jax_dna.input.trajectory as trajectory
import jax_dna.input.toml as toml_reader
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.energy.base as jdna_energy
import jax_dna.energy.configuration as jdna_energy_config
import jax_dna.utils.types as jdt
import jax_dna.simulators.jax_md as jmd

jax.config.update("jax_enable_x64", True)

if __name__=="__main__":

    topology_fname = "data/templates/simple-helix/sys.top"
    traj_fname = "data/templates/simple-helix/init.conf"
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
        simulator_params=jmd.StaticSimulatorParams(
            seq=seq,
            mass=mass,
            bonded_neighbors=top.bonded_neighbors,
            n_steps=experiment_config["n_steps"],
            checkpoint_every=experiment_config["checkpoint_interval"],
            dt=dt,
            kT=kT,
            gamma=gamma,
        ),
        space=(displacement_fn, shift_fn),
        transform_fn=transform_fn,
        simulator_init=jax_md.simulate.nvt_langevin,
        neighbors=jmd.NoNeighborList(unbonded_nbrs=top.unbonded_neighbors),
    )


    fn = jax.jit(lambda opts: sampler.run(opts, init_body, experiment_config["n_steps"], key))
    opt_params = [c.opt_params for c in configs]



    transformed_fns = [
        e_fn(
            displacement_fn=displacement_fn,
            params=(e_c | param).init_params(),
        )
        for param, e_c, e_fn in zip(opt_params, configs, energy_fns, strict=True)
    ]

    composed_energy_fn = jdna_energy.ComposedEnergyFunction(
        energy_fns=transformed_fns,
        rigid_body_transform_fn=transform_fn,
    )
    outs = fn(opt_params)
    print(outs.center.shape, outs.orientation.vec.shape)
    ce = jax.vmap(lambda x: composed_energy_fn(x, seq, top.bonded_neighbors, top.unbonded_neighbors.T))(outs)
    print(ce[:5])



    # jax.grad(lambda opts: loss(fn(opts), target))

    # def loss_fn(trajectory, target) -> float:
    #     return trajectory - target

    # grad_fn = jax.jit(jax.grad(lambda opts: loss_fn(opts, sim_fn)))


    # graddable_fn = lambda op: sampler.run(op, init_body, experiment_config["n_steps"], key).center.sum()
    # grad_fn = jax.jit(jax.grad(graddable_fn))
    # print(grad_fn(opt_params))