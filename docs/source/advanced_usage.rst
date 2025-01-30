Advanced Usage
==============

Custom Energy Functions
-----------------------

``jax_dna`` supports custom energy functions



Advanced Optimizations
----------------------

Beyond the simple optimization covered in :doc:`basic_usage` more sophisticated
optimizations require multiple heterogenous simulations and with multiple kinds
of loss functions. To accommodate this, ``jax_dna`` sets up optimizations using
the following abstractions:

.. image:: ../_static/jax_dna_opt_diagram.svg
    :align: center


- ``Simulator``: A ``Simulator`` is actor that that exposes one or more
  ``Observables``.
- ``Observable``: An ``Observable`` is something produced by a ``Simulator``. It
  can be a trajectory, scalar, vector, or a tensor. Or really anything that an ``Objective`` needs to compute its the loss/gradients.
- ``Objective``: An ``Objective`` is an actor that takes in one or more
  ``Observables`` and returns the gradients of the ``Objective`` with respect to
  the parameters we want to optimize.
- ``Optimizer``: An ``Optimizer`` coordinates running the ``Simulators`` and to
  produce the ``Observables`` that are needed by the ``Objectives`` to optimize
  the parameters we are interested in.


Using these abstractions ``jax_dna`` leverages the `ray <https://ray.io>`_
library to run ``Simulators`` and ``Objectives`` in parallel across multiple
heterogenous devices. This allows for ``jax_dna`` to schedule ``Simulators`` and
calculate gradients using ``Objectives`` in parallel. This is particularly useful
when the ``Simulators`` are slow to run and the ``Objectives`` are expensive to
compute.

See `advanced_optimizations
<https://github.com/ssec-jhu/jax-dna/tree/master/examples/advanced_optimizations/oxdna>`_
for more details and examples.
