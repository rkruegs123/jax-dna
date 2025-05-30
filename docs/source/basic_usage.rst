Basic Usage
===========

.. _installation:

Installation
------------

To use ``jax_dna``, first install it using ``pip``:

.. code-block:: bash

    pip install git+https://github.com/rkruegs123/jax-dna.git


Basic Usage
-----------

The two basic use cases for ``jax_dna`` are:

1. Running a simulation
2. Running an optimization

In this section, we'll discuss how to run simple simulations and simple
optimizations.



Running a single simulation
***************************

``jax_dna`` currently supports two simulation engine:
`jax_md <https://github.com/jax-md/jax-md>`_ and
`oxDNA <https://dna.physics.ox.ac.uk/index.php/Main_Page>`_.

Regardless of which engine you choose, setting up the system to simulate is the
same, as ``jax_dna`` supports reading ``oxDNA`` input, topology, and trajectory
files. See :doc:`autoapi/jax_dna/input/index` for more details on the input
format.

.. code-block:: python

    import jax_dna.input.topology as jdna_top
    import jax_dna.input.trajectory as jdna_traj

    topology = topology.from_oxdna_file("path/to/oxdna/topology.top")
    initial_positions = jdna_traj.from_file("path/to/oxdna/trajectory.conf").states[0].to_rigid_body()


Using ``jax_md``
^^^^^^^^^^^^^^^^

To run a simulation using ``jax_md`` requires a working ``jax`` installation
which is installed alongside ``jax_dna`` via ``pip`` if it isn't installed
already. For information on installing ``jax_md`` and ``jax``, please refer to
the their respective documentation. For more details on the ``jax_md`` simulator
see :doc:`autoapi/jax_dna/simulators/jax_md/index`.

Running a simulation using ``jax_md`` involves reading some input data as shown
above and then building the energy function:


.. code-block:: python

    import functools

    import jax.numpy as jnp
    import jax_md

    import jax_dna.energy.dna1 as jdna_energy

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
    params = [{} for _ in range(len(energy_fn_configs))]
    energy_fns = jdna_energy.default_energy_fns()

    # Build the energy function
    energy_function = jdna_jax_md.build_energy_function(topology, initial_positions)

The variable ``energy_function`` is a function that takes in a set of rigid
bodies and returns the total energy of the system. To run a simulation, we pass
that function to the ``jax_md`` simulator:


.. code-block:: python

    import jax_dna.simulators.jax_md as jdna_jaxmd

    simulator = jdna_jaxmd.JaxMDSimulator(
        energy_configs=energy_fn_configs,
        energy_fns=energy_fns,
        topology=topology,
        simulator_params=jdna_jaxmd.StaticSimulatorParams(
            seq=jnp.array(topology.seq),
            mass=mass,
            bonded_neighbors=topology.bonded_neighbors,
            # this is gradient checkpointing which isn't used in this examples
            checkpoint_every=100,
            dt=dt,
            kT=kT,
            gamma=gamma,
        ),
        space=jax_md.space.free(),
        transform_fn=transform_fn,
        simulator_init=jax_md.simulate.nvt_langevin,
        neighbors=jdna_jaxmd.NoNeighborList(unbonded_nbrs=topology.unbonded_neighbors),
    )

    key = jax.random.PRNGKey(0)
    sim_fn = jax.jit(lambda opts: simulator.run(opts, initial_positions, run_config["n_steps"], key))
    trajectory = sim_fn(params)

A runnable version of this example can be found in the examples
`folder <https://github.com/ssec-jhu/jax-dna/tree/master/examples/simulations/jaxmd>`_
in the repository.


Using ``oxDNA``
^^^^^^^^^^^^^^^

When running oxDNA simulations, ``jax_dna`` acts as a thin wrapper around the
``oxDNA`` executable. To run a simulation, you need to have a working oxDNA
installation. For more information on installing oxDNA, please refer to the
oxDNA documentation. Additionally, the following environment variable must
point to the oxDNA executable: ``OXDNA_BIN_PATH``

.. code-block:: python

    from pathlib import Path

    import jax_dna.input.trajectory as jdna_traj
    import jax_dna.input.topology as jdna_top
    import jax_dna.simulators.oxdna as jdna_oxdna
    import jax_dna.utils.types as jdna_types

    input_dir = Path("path/to/oxdna-input/dir")

    simulator = jdna_oxdna.oxDNASimulator(
        input_dir=input_dir,
        sim_type=jdna_types.oxDNASimulatorType.DNA1,
    )

    simulator.run()

    trajectory = jdna_traj.from_file(
        input_dir / "output.dat",
        strand_lengths=jdna_top.from_oxdna_file(input_dir / "sys.top").strand_counts,
    )

    print("Length of trajectory: ", trajectory.state_rigid_body.center.shape[0])


.. https://jwodder.github.io/kbits/posts/rst-hyperlinks/#gotcha-duplicate-link-text

A runnable version of this example can be found in the examples
`folder <https://github.com/ssec-jhu/jax-dna/tree/master/examples/simulations/oxdna>`__
in the repository.

Running a simple optimization
*****************************

The main advantage in using ``jax_dna`` is the ability to run optimizations. The
optimizations can be run directly through the simulation using ``jax_md`` or
using ``oxDNA`` and the `DiffTRe
<https://www.nature.com/articles/s41467-021-27241-4>`_ algorithm.

As an example we will run a simple optimization, that will find the energy
function parameters that produce a desired propeller twist.

This setup is the same for using either the ``jax_md`` or ``oxDNA`` simulators
but the implementation is slightly different.

Using ``jax_md``
^^^^^^^^^^^^^^^^

Below is an example of running an optimization using ``jax_md``. The example
will optimize the energy function parameters to produce the target propeller
twist.

First setup the system:

.. code-block::python

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
    import jax_dna.ui.loggers.jupyter as jupyter_logger
    import jax_dna.utils.types as jdna_types

    jax.config.update("jax_enable_x64", True)


    run_config = {
        "n_sim_steps": 20_000,
        "n_opt_steps": 25,
        "learning_rate": 0.00001,
    }


    experiment_dir = Path("data/sys-defs/simple-helix")

    topology = jdna_top.from_oxdna_file(experiment_dir / "sys.top")
    initial_positions = (
        jdna_traj.from_file(
            experiment_dir / "bound_relaxed.conf",
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



Then setup the energy function, configs, and get the parameters that we want to
optimize:

.. code-block::python
    energy_fns = jdna_energy.default_energy_fns()
    energy_fn_configs = jdna_energy.default_energy_configs()

    params = []
    for ec in energy_fn_configs:
        params.append(
            ec.opt_params if isinstance(ec, jdna_energy.StackingConfiguration) else {}
        )
    # we're not going to optimize wrt the seq specific stacking weights
    for op in params:
        if "ss_stack_weights" in op:
            del op["ss_stack_weights"]


Next setup the simulator:


.. code-block::python
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

Now set up the loss that will be optimized and the function that we will just to
calculate the loss and gradients:

.. code-block::python
    loss_fn = jdna_losses.ObservableLossFn(
        observable=jdna_obs.propeller.PropellerTwist(
            rigid_body_transform_fn=transform_fn,
            h_bonded_base_pairs=jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
        ),
        loss_fn=jdna_losses.RootMeanSquaredError(),
        return_observable=True,
    )

    # we're going to ignore the first 10% of the simulation steps for the loss calculation
    eq_steps = int(run_config["n_sim_steps"] * 0.1)
    other_steps = run_config["n_sim_steps"] - eq_steps
    weights = jnp.concat([
        jnp.zeros(eq_steps, dtype=jnp.float64),
        jnp.ones(other_steps, dtype=jnp.float64)/other_steps
    ])
    target_prop_twist = jnp.array(jdna_obs.propeller.TARGETS["oxDNA"], dtype=jnp.float64)
    def graddable_loss(in_params:jdna_types.Params, in_key:jax.random.PRNGKey) -> tuple[float, tuple[float, jax.random.PRNGKey]]:
        in_key, subkey = jax.random.split(in_key)
        sim_out = simulator.run(in_params, initial_positions, run_config["n_sim_steps"], subkey)
        loss, ptwist = loss_fn(sim_out, target_prop_twist, weights)
        return (loss, (ptwist, in_key))

    grad_fn = jax.jit(jax.value_and_grad(graddable_loss, has_aux=True))


Finally, run the optimization:

.. code-block::python
    key = jax.random.PRNGKey(1234)
    optimizer = optax.adam(learning_rate=run_config["learning_rate"])
    opt_state = optimizer.init(params)

    for i in range(run_config["n_opt_steps"]):
        (loss, (prop_twist, key)), grads = grad_fn(params, key)

        print("loss", loss)
        print("prop_twist", prop_twist)
        print("target_ptwist", target_prop_twist)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)


As the optimization runs you should see the propeller twist getting closer to
the target propeller twist (with some noisiness).



Using ``oxDNA``
^^^^^^^^^^^^^^^

Different from ``jax_md`` we cannot differentiate through the oxDNA simulation.
Instead we use the ``DiffTRe`` algorithm to optimize the energy. The
optimization for ``oxDNA`` / ``DiffTRe`` is more complicated than the ``jax_md``
optimization. For these kinds of optimizations go to :doc:`advanced_usage`.
