

import functools
import logging
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jax_dna.energy as jdna_energy
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.input.toml as toml_reader
import jax_dna.utils.types as jdna_types
import jax_dna.simulators.oxdna as oxdna
from jax_dna.input import topology

import matplotlib.pyplot as plt
import pandas as pd


os.environ[oxdna.BIN_PATH_ENV_VAR] = str(Path("../oxDNA/build/bin/oxDNA").resolve())
os.environ[oxdna.BUILD_PATH_ENV_VAR] =  str(Path("../oxDNA/build").resolve())

jax.config.update("jax_enable_x64", True)


def main():
    objective_logging_config = {
        "level":logging.DEBUG,
        "filename":"objective.log",
        "filemode":"w",
    }
    simulator_logging_config = objective_logging_config | {"filename": "simulator.log"}


    simulation_config = toml_reader.parse_toml("jax_dna/input/dna1/default_simulation.toml")
    kT = simulation_config["kT"]
    energy_config = toml_reader.parse_toml("jax_dna/input/dna1/default_energy.toml")

    energy_fns = dna1_energy.default_energy_fns()
    energy_configs = dna1_energy.default_configs()
    opt_params = [ec.opt_params for ec in dna1_energy.default_configs()]

    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        dna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    topology_fname = "data/templates/simple-helix/sys.top"
    top = topology.from_oxdna_file(topology_fname)

    energy_fn_builder_fn = jdna_energy.energy_fn_builder(
        energy_fns=energy_fns,
        energy_configs=energy_configs,
        transform_fn=transform_fn,
    )

    def energy_fn_builder(params: jdna_types.Params) -> callable:
        return jax.vmap(
            lambda trajectory: energy_fn_builder_fn(params)(
                trajectory.rigid_body,
                seq=jnp.array(top.seq),
                bonded_neighbors=top.bonded_neighbors,
                unbonded_neighbors=top.unbonded_neighbors.T,
            )
        )

    input_dir = "data/templates/simple-helix"
    simulator = oxdna.oxDNASimulator(
        input_dir=input_dir,
        sim_type=jdna_types.oxDNASimulatorType.DNA1,
        energy_configs=energy_configs,
        n_build_threads=4,
        logger_config=simulator_logging_config,
    )

    run_params = [{} for _ in opt_params]
    traj = simulator.run(run_params)
    energies = energy_fn_builder(run_params)(traj)
    print(energies)

    energy_logs = {}

    names = [ "fene", "b_exc", "stack", "n_exc", "hb", "cr_stack", "cx_stack"]
    oxdna_energies = pd.read_csv("data/templates/simple-helix/split_energy.dat", sep="\s+", names=["t"] + names)
    for name, ef, ec in zip(names, energy_fns, energy_configs):
        energy_fn_builder_fn = jdna_energy.energy_fn_builder(
            energy_fns=[ef],
            energy_configs=[ec],
            transform_fn=transform_fn,
        )
        def energy_fn_builder(params: jdna_types.Params) -> callable:
            return jax.vmap(
                lambda trajectory: energy_fn_builder_fn(params)(
                    trajectory.rigid_body,
                    seq=jnp.array(top.seq),
                    bonded_neighbors=top.bonded_neighbors,
                    unbonded_neighbors=top.unbonded_neighbors.T,
                )
            )
        energy_logs[name] = energy_fn_builder([{}])(traj)

    fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(4, 16))
    for i, name in enumerate(names):

        axes[i, 0].hist(oxdna_energies[name].values[1:])
        axes[i, 1].hist(energy_logs[name] / top.n_nucleotides)
        diffs = oxdna_energies[name].values[1:] - (energy_logs[name] / top.n_nucleotides)
        axes[i, 2].hist(diffs[diffs!=0], range=(-np.abs(diffs).max(), np.abs(diffs).max()))

        if i==0:
            axes[i, 0].set_title("oxDNA: " + name)
            axes[i, 1].set_title("JAX-DNA: " + name)
            axes[i, 2].set_title("Difference excl. 0's: " + name)
        else:
            axes[i, 0].set_title(name)
            axes[i, 1].set_title(name)
            axes[i, 2].set_title(name)


    plt.tight_layout()
    plt.show()




if __name__=="__main__":
    main()