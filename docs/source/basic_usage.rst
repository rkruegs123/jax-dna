Basic Usage
===========

.. _installation:

Installation
------------

To use ``jax_dna``, first install it using ``pip``:

.. code-block:: bash

    pip install jax_dna


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
`folder <https://github.com/ssec-jhu/jax-dna/tree/master/examples/simulations/jax_md>`_
in the repository.


Using ``oxDNA``
^^^^^^^^^^^^^^^







Running a simple optimization
*****************************

How to run a simple optimization
