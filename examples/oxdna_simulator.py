import os

import jax_dna.simulators.oxdna.oxdna as jd_oxdna


if __name__=="__main__":
    # ox dna is compiled and built in the parent dir to the repository.
    # we have to use a bunch of parent directories to get to the binary to
    # get to the top of the repository
    os.environ[jd_oxdna.BIN_PATH_ENV_VAR] = "../../../../oxDNA/build/bin/oxDNA"
    os.chdir("./data/templates/simple-helix")
    traj = jd_oxdna.oxDNASimulator().run(
        "."
    )

    print(traj)

