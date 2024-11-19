import dataclasses as dc
import itertools
import typing
from typing import Any

import chex
import jax
import ray


import examples.simulator_actor as sim_actor

import jax_dna.energy.base as jdna_energy
import jax_dna.energy.configuration as jdna_energy_config
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.losses.observable_wrappers as jdna_losses
import jax_dna.optimization.base as jdna_optimization
import jax_dna.simulators.base as jdna_simulators
import jax_dna.simulators.io as jdna_sio
import jax_dna.utils.types as jdna_types



def main():
    logger = None

    optimization_config = {
        "n_steps": 100,
    }

    energy_fns = dna1_energy.default_energy_fns()
    energy_configs = dna1_energy.default_configs()
    opt_params = [c.opt_params for c in energy_configs]




    objectives = []
    simulators = []

    opt = jdna_optimization.Optimization(
        objectives=objectives,
        simulators=simulators,
    )

    for i in range(optimization_config["n_steps"]):
        opt, grads = opt.step(params)
        params = opt.update_params(params, grads)

        for objective in opt.objectives:
            log_values = objective.get_latest_values()
            for (name, value) in log_values:
                logger.log_metric(name, value, step=i)



if __name__=="__main__":
    main()