"""An example of running a simulation using oxDNA.

Important: This assumes that the current working is the root directory of the
repository. i.e. this file was invoked using:

``python -m examples.simulations.oxdna.oxDNA``
"""
from pathlib import Path

import jax_dna.input.trajectory as jdna_traj
import jax_dna.input.topology as jdna_top
import jax_dna.simulators.oxdna as jdna_oxdna
import jax_dna.utils.types as jdna_types

# This depends on a working oxDNA installation and the `oxdna` executable
# should be set using the following environment variable:
# jdna_oxdna.BIN_PATH_ENV_VAR

def main():

    input_dir = Path("data/templates/simple-helix")

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


if __name__ == "__main__":
    main()


