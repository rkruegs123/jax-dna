"""An example of running a simple optimization using jax_md.

Important: This assumes that the current working is the root directory of the
repository. i.e. this file was invoked using:

``python -m examples.simple_optimizations.jaxmd.jaxmd``
"""

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_md
import optax

import jax_dna.energy.dna1 as jdna_energy
import jax_dna.input.topology as jdna_top
import jax_dna.input.trajectory as jdna_traj
import jax_dna.losses.observable_wrappers as jdna_losses
import jax_dna.observables as jdna_obs
import jax_dna.simulators.jax_md as jdna_jaxmd
import jax_dna.utils.types as jdna_types

jax.config.update("jax_enable_x64", True)

def main():
    # configs specific to this file
    run_config = {
        "n_sim_steps": 20_000,
        "n_opt_steps": 25,
        "learning_rate": 0.00001,
    }

    experiment_dir = Path("data/templates/simple-helix")

    topology = jdna_top.from_oxdna_file(experiment_dir / "sys.top")
    initial_positions = (
        jdna_traj.from_file(
            experiment_dir / "init.conf",
            topology.strand_counts,
        )
        .states[0]
        .to_rigid_body()
    )

    experiment_config, energy_config = jdna_energy.default_configs()

    dt = experiment_config["dt"]
    kT = experiment_config["kT"]
    diff_coef = experiment_config["diff_coef"]
    rot_diff_coef = experiment_config["rot_diff_coef"]

    # These are special values for the jax_md simulator
    gamma = jax_md.rigid_body.RigidBody(
        center=jnp.array([kT / diff_coef], dtype=jnp.float64),
        orientation=jnp.array([kT / rot_diff_coef], dtype=jnp.float64),
    )
    mass = jax_md.rigid_body.RigidBody(
        center=jnp.array([experiment_config["nucleotide_mass"]], dtype=jnp.float64),
        orientation=jnp.array([experiment_config["moment_of_inertia"]], dtype=jnp.float64),
    )

    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        jdna_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    # The jax_md simulator needs an energy function. We can use the default
    # energy functions and configurations for dna1 simulations. For more
    # information on energy functions and configurations, see the documentation.
    energy_fn_configs = jdna_energy.default_energy_configs()

    # For this example were only going to optimize the parameters that are
    # associated with the Stacking energy function.
    params = []
    for ec in energy_fn_configs:
        params.append(
            ec.opt_params if isinstance(ec, jdna_energy.StackingConfiguration) else {}
        )
    # we're not going to optimize wrt the seq specific stacking weights
    for op in params:
        if "ss_stack_weights" in op:
            del op["ss_stack_weights"]



    energy_fns = jdna_energy.default_energy_fns()

    simulator = jdna_jaxmd.JaxMDSimulator(
        energy_configs=energy_fn_configs,
        energy_fns=energy_fns,
        topology=topology,
        simulator_params=jdna_jaxmd.StaticSimulatorParams(
            seq=jnp.array(topology.seq),
            mass=mass,
            bonded_neighbors=topology.bonded_neighbors,
            checkpoint_every=500,
            dt=dt,
            kT=kT,
            gamma=gamma,
        ),
        space=jax_md.space.free(),
        transform_fn=transform_fn,
        simulator_init=jax_md.simulate.nvt_langevin,
        neighbors=jdna_jaxmd.NoNeighborList(unbonded_nbrs=topology.unbonded_neighbors),
    )

    # ==========================================================================
    # Up until this point this is identical to running a simulation, save for
    # the definition of `params`, to run an optimization we need to define a few
    # more things: a loss function, a function that computes the loss given a
    # set of parameters, and  an optimizer
    # ==========================================================================

    # The ObservableLossFn class is a convenience wrapper for computing the the
    # loss of an observable. In this case, we are using the propeller twist and
    # the loss is squared error. the ObservableLossFn class implements __call__
    # that takes the output of the simulation, the target, and weights and
    # returns the loss and the measured observable.
    loss_fn = jdna_losses.ObservableLossFn(
        observable=jdna_obs.propeller.PropellerTwist(
            rigid_body_transform_fn=transform_fn,
            h_bonded_base_pairs=jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
        ),
        loss_fn=jdna_losses.RootMeanSquaredError(),
        return_observable=True,
    )

    # make the weights 1/n, where n is the number of time steps, so a simple
    # average
    weights = jnp.ones(
        run_config["n_sim_steps"],
        dtype=jnp.float64,
    ) / run_config["n_sim_steps"]

    target_prop_twist = jnp.array(jdna_obs.propeller.TARGETS["oxDNA"], dtype=jnp.float64)
    def graddable_loss(in_params:jdna_types.Params, in_key:jax.random.PRNGKey) -> tuple[float, tuple[float, jax.random.PRNGKey]]:
        in_key, subkey = jax.random.split(in_key)
        sim_out = simulator.run(in_params, initial_positions, run_config["n_sim_steps"], subkey)
        loss, ptwist = loss_fn(sim_out, target_prop_twist, weights)
        return (loss, (ptwist, in_key))

    # we can use this function to calculate the gradients of the loss and
    # the other items we care about the loss, the prop twist, and to curry
    # the key for the simulation
    grad_fn = jax.jit(jax.value_and_grad(graddable_loss, has_aux=True))

    # Now we setup an simple optimization loop. This is just to show an example.
    # In practice, ``jax_dna`` has abstracted and generalized this process in
    # the ``jax_dna.optimization`` module.

    key = jax.random.PRNGKey(1234)
    optimizer = optax.adam(learning_rate=run_config["learning_rate"])
    opt_state = optimizer.init(params)

    for i in range(run_config["n_opt_steps"]):
        (loss, (prop_twist, key)), grads = grad_fn(params, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        print(f"Step {i}: Loss: {loss}, Measured : {prop_twist} Target: {target_prop_twist}")


if __name__=="__main__":
    main()