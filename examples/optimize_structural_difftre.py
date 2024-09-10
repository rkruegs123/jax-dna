import functools

import jax
import jax.numpy as jnp
import jax_md
import matplotlib.pyplot as plt
import numpy as np
import optax

import jax_dna.energy.dna1 as dna1_energy
import jax_dna.gradient_estimators.difftre as difftre
import jax_dna.input.topology as topology
import jax_dna.input.trajectory as trajectory
import jax_dna.input.toml as toml_reader
import jax_dna.simulators.oxdna as oxdna
import jax_dna.losses.observable_wrappers as loss_wrapper
import jax_dna.observables as obs



def main() -> None:
    topology_fname = "data/templates/simple-helix/sys.top"
    traj_fname = "data/templates/simple-helix/init.conf"
    simulation_config = "jax_dna/input/dna1/default_simulation.toml"
    energy_config = "jax_dna/input/dna1/default_energy.toml"
    input_dir = "data/templates/simple-helix"

    n_eq_steps = 10_000
    n_samples_steps = 100_000
    sample_every = 1_000
    lr  = 0.001
    min_n_eff_factor = 0.95
    seed = 0
    n_epochs = 100

    key = jax.random.PRNGKey(seed)
    n_ref_states = n_samples_steps // sample_every

    top = topology.from_oxdna_file(topology_fname)
    seq = jnp.array(top.seq_one_hot)
    traj = trajectory.from_file(
        traj_fname,
        top.strand_counts,
    )
    experiment_config = toml_reader.parse_toml(simulation_config)
    energy_config = toml_reader.parse_toml(energy_config)

    kT = experiment_config["kT"]

    energy_configs = [
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

    opt_params = [c.opt_params for c in energy_configs]

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

    loss_fns = [
        functools.partial(
            loss_wrapper.ObservableLossFn(
                observable=obs.propeller.PropellerTwist(
                    h_bonded_base_pairs=jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
                ),
                loss_fn=loss_wrapper.SquaredError(),
            ),
            target=obs.propeller.TARGETS["oxDNA"]
        ),
    ]

    sim_init_fn = functools.partial(
        oxdna.oxDNASimulator,
        input_dir=input_dir,
    )


    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        dna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    ge = difftre.DiffTRe(
        beta=1/kT,
        n_eq_steps = n_eq_steps,
        min_n_eff = int(n_ref_states * min_n_eff_factor),
        sample_every = sample_every,
        space = jax_md.space.free(),
        topology = top,
        rigid_body_transform_fn = transform_fn,
        energy_configs = energy_configs,
        energy_fns = energy_fns,
        loss_fns = loss_fns,
        sim_init_fn = sim_init_fn,
        n_sim_steps = n_samples_steps,
        key = key,
        ref_states = None,
        ref_energies = None,

    ).intialize(opt_params)

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(opt_params)


    for _ in range(n_epochs):
        ge, grads, loss, losses = ge(opt_params, loss_fns, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        opt_params = optax.apply_updates(opt_params, updates)

        print(f"Loss: {loss}")






if __name__=="__main__":
    main()