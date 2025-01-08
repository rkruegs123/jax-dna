"""OXDNA sampler module.

Run an jax_dna simulation using an oxDNA sampler.
"""

import os
import subprocess
from pathlib import Path

import chex

import jax_dna.energy.configuration as jd_energy
import jax_dna.input.oxdna_input as jd_oxdna
import jax_dna.input.topology as jd_top
import jax_dna.input.trajectory as jd_traj
import jax_dna.simulators.base as jd_base
import jax_dna.simulators.io as jd_sio
import jax_dna.simulators.oxdna.utils as oxdna_utils
import jax_dna.utils.types as jd_types

REQUIRED_KEYS = {
    "oxnda_bin",
    "input_directory",
}

ERR_OXDNA_NOT_FOUND = "OXDNA binary not found at: {}"
ERR_MISSING_REQUIRED_KEYS = "Missing required keys: {}"
ERR_INPUT_FILE_NOT_FOUND = "Input file not found: {}"
ERR_OXDNA_FAILED = "OXDNA simulation failed"
OXDNA_TRAJECTORY_FILE_KEY = "trajectory_file"
OXDNA_TOPOLOGY_FILE_KEY = "topology"

BIN_PATH_ENV_VAR = "OXDNA_BIN_PATH"
ERR_BIN_PATH_NOT_SET = "OXDNA_BIN_PATH environment variable not set"
BUILD_PATH_ENV_VAR = "OXDNA_BUILD_PATH"
ERR_BUILD_PATH_NOT_SET = "OXDNA_BUILD_PATH environment variable not set"
ERR_BUILD_SETUP_FAILED = "OXDNA build setup failed wiht return code: {}"


@chex.dataclass
class oxDNASimulator(jd_base.BaseSimulation):  # noqa: N801 oxDNA is a special word
    """A sampler base on running an oxDNA simulation."""

    input_dir: str
    sim_type: jd_types.oxDNASimulatorType
    energy_configs: list[jd_energy.BaseConfiguration]
    n_build_threads: int = 4

    def run(
        self,
        opt_params: list[jd_types.Params] | None = None,
        **kwargs,  # noqa: ARG002 we want to satisfy the interface
    ) -> jd_traj.Trajectory:
        """Run an oxDNA simulation."""

        # The user may want to override the current parameters in the oxdna binary
        # If so, we need to update the parameters in the src/model.h file
        if opt_params is not None:
            print("Updating parameters")
            self._update_params(new_params=opt_params)
            print("Parameters updated")

        init_dir = Path(self.input_dir)

        input_file = init_dir / "input"
        if not input_file.exists():
            raise FileNotFoundError(ERR_INPUT_FILE_NOT_FOUND.format(input_file))

        if BIN_PATH_ENV_VAR not in os.environ:
            raise ValueError(ERR_BIN_PATH_NOT_SET)

        oxdna_config = jd_oxdna.read(init_dir / "input")
        output_file = init_dir / oxdna_config["trajectory_file"]

        print("Running oxDNA simulation")
        with Path(init_dir / "oxdna.out.log").open("w") as f, Path(init_dir / "oxdna.err.log").open("w") as f_err:
            subprocess.run(
                [  # noqa: S603
                    os.environ[BIN_PATH_ENV_VAR],
                    "input",
                ],
                stdout=f,
                stderr=f_err,
                check=True,
                cwd=init_dir,
            )
        print("Simulation complete")
        # read the output trajectory file
        topology = jd_top.from_oxdna_file(init_dir / oxdna_config["topology"])
        # return the trajectory
        trajectory = jd_traj.from_file(output_file, topology.strand_counts, is_oxdna=True)

        # if we have changed things in oxDNA, restore the files to the way they were
        if opt_params:
            self._restore_params()

        return jd_sio.SimulatorTrajectory(
            rigid_body=trajectory.state_rigid_body,
        )

    def _update_params(self, *, new_params: list[dict], **kwargs) -> None:
        """Update the simulation.

        This function will recompile the oxDNA binary with the new parameters.
        """
        if BUILD_PATH_ENV_VAR not in os.environ:
            raise ValueError(ERR_BUILD_PATH_NOT_SET)

        build_dir = Path(os.environ[BUILD_PATH_ENV_VAR])

        with (
            Path(build_dir / "jax_dna.cmake.std.log").open("w") as f,
            Path(build_dir / "jax_dna.cmake.err.log").open("w") as f_err,
        ):
            completed_proc = subprocess.run(["cmake", ".."], cwd=build_dir, stdout=f, stderr=f_err, check=True)

        if completed_proc.returncode != 0:
            raise ValueError(ERR_BUILD_SETUP_FAILED.format(completed_proc.returncode))

        updated_params = [(ec | np).init_params() for ec, np in zip(self.energy_configs, new_params, strict=True)]

        # check for existing src/model.h file save a copy if we haven't already
        old_model_h = build_dir.parent.joinpath("src/model.h.old")
        model_h = build_dir.parent.joinpath("src/model.h")
        orig_text = model_h.read_text()
        if not old_model_h.exists():
            # copy the original, so we can restore it later
            old_model_h.write_text(orig_text)

        # update the values in the src/model.h
        oxdna_utils.update_params(
            model_h, [up.to_dictionary(include_dependent=True, exclude_non_optimizable=True) for up in updated_params]
        )

        # rebuild the binary
        with (
            Path(build_dir / "jax_dna.make.std.log").open("w") as f,
            Path(build_dir / "jax_dna.make.err.log").open("w") as f_err,
        ):
            completed_proc = subprocess.run(
                ["make", f"-j{self.n_build_threads}"],
                cwd=build_dir,
                check=True,
                stdout=f,
                stderr=f_err,
            )

        if completed_proc.returncode != 0:
            # restore the original src/model.h
            model_h.write_text(orig_text)
            raise ValueError(ERR_BUILD_SETUP_FAILED.format(completed_proc.returncode))

    def _restore_params(self) -> None:
        """Restore the original parameters."""
        build_dir = Path(os.environ[BUILD_PATH_ENV_VAR])
        old_model_h = build_dir.parent.joinpath("src/model.h.old")
        model_h = build_dir.parent.joinpath("src/model.h")
        if old_model_h.exists():
            # restore the original src/model.h
            old_model_h.replace(model_h)
        else:
            raise FileNotFoundError("Original model.h file not found")
