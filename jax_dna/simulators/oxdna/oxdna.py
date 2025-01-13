"""OXDNA sampler module.

Run an jax_dna simulation using an oxDNA sampler.
"""

import logging
import os
import subprocess
import typing
import warnings
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
WARN_CANT_GUESS_BIN_LOC = (
    "Could not guess the location of the {} binary, be sure {} is set to its location for oxDNA recompilation."
)
ERR_ORIG_MODEL_H_NOT_FOUND = "Original model.h file not found, looked at {}"

MAKE_BIN_ENV_VAR = "MAKE_BIN_PATH"
CMAKE_BIN_ENV_VAR = "CMAKE_BIN_PATH"

CMAKE_MAKE_BIN_LOC_GUESSES = [
    "/bin/{}",
    "/usr/bin/{}",
    "/snap/bin/{}",
    r"C:\Program Files (x86)\GnuWin32\bin\{}.exe",
]

logger = logging.getLogger(__name__)


# We do not force the user the set this because they may not be recompiling oxDNA
def _guess_binary_location(bin_name: str, env_var: str) -> Path | None:
    """Guess the location of a binary."""
    guessed_path = None
    for guess in CMAKE_MAKE_BIN_LOC_GUESSES:
        pth = Path(guess.format(bin_name))
        if pth.exists():
            guessed_path = pth
            break

    if guessed_path is None:
        warnings.warn(WARN_CANT_GUESS_BIN_LOC.format(bin_name, env_var), stacklevel=2)
        logger.debug(WARN_CANT_GUESS_BIN_LOC.format(bin_name, env_var))
    return os.environ.get(env_var, None) or guessed_path


@chex.dataclass
class oxDNASimulator(jd_base.BaseSimulation):  # noqa: N801 oxDNA is a special word
    """A sampler base on running an oxDNA simulation."""

    input_dir: str
    sim_type: jd_types.oxDNASimulatorType
    energy_configs: list[jd_energy.BaseConfiguration]
    n_build_threads: int = 4
    logger_config: dict[str, typing.Any] | None = None

    def __post_init__(self, *args, **kwds) -> None:
        """Check the validity of the configuration."""
        self._initialize_logger()

    def _initialize_logger(self) -> None:
        config = self.logger_config if self.logger_config is not None else {}
        level = config.get("level", logging.INFO)
        logger = logging.getLogger(__name__)
        logger.setLevel(level)

        if config.get("filename", None):
            handler = logging.FileHandler(config["filename"])
            handler.setLevel(level)
        else:
            handler = logging.StreamHandler()
            handler.setLevel(level)

        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
        self._logger = logger

    def run(
        self,
        opt_params: list[jd_types.Params] | None = None,
        **kwargs,  # noqa: ARG002 we want to satisfy the interface
    ) -> jd_traj.Trajectory:
        """Run an oxDNA simulation."""
        # The user may want to override the current parameters in the oxdna binary
        # If so, we need to update the parameters in the src/model.h file
        if not self._logger.handlers:
            self._initialize_logger()

        if opt_params is not None:
            self._update_params(new_params=opt_params)

        init_dir = Path(self.input_dir)
        input_file = init_dir / "input"

        self._logger.info("oxDNA input file: %s", input_file)

        if not input_file.exists():
            raise FileNotFoundError(ERR_INPUT_FILE_NOT_FOUND.format(input_file))

        if BIN_PATH_ENV_VAR not in os.environ:
            raise ValueError(ERR_BIN_PATH_NOT_SET)

        oxdna_config = jd_oxdna.read(init_dir / "input")
        output_file = init_dir / oxdna_config["trajectory_file"]

        std_out_file = init_dir / "oxdna.out.log"
        std_err_file = init_dir / "oxdna.err.log"
        self._logger.info("Starting oxDNA simulation")
        self._logger.debug(
            "oxDNA std_out->%s, std_err->%s",
            std_out_file,
            std_err_file,
        )
        with std_out_file.open("w") as f_std, std_err_file.open("w") as f_err:
            subprocess.run(
                [  # noqa: S603
                    os.environ[BIN_PATH_ENV_VAR],
                    "input",
                ],
                stdout=f_std,
                stderr=f_err,
                check=True,
                cwd=init_dir,
            )
        self._logger.info("oxDNA simulation complete")

        # read the output trajectory file
        topology = jd_top.from_oxdna_file(init_dir / oxdna_config["topology"])
        # return the trajectory
        trajectory = jd_traj.from_file(output_file, topology.strand_counts, is_oxdna=True)

        # if we have changed things in oxDNA, restore the files to the way they were
        if opt_params:
            self._restore_params()

        self._logger.debug(
            "oxDNA trajectory com size: %s",
            str(trajectory.state_rigid_body.center.shape),
        )
        return jd_sio.SimulatorTrajectory(
            rigid_body=trajectory.state_rigid_body,
        )

    def _update_params(self, *, new_params: list[dict]) -> None:
        """Update the simulation.

        This function will recompile the oxDNA binary with the new parameters.
        """
        if BUILD_PATH_ENV_VAR not in os.environ:
            raise ValueError(ERR_BUILD_PATH_NOT_SET)
        _cmake_bin = _guess_binary_location("cmake", CMAKE_BIN_ENV_VAR)
        _make_bin = _guess_binary_location("make", MAKE_BIN_ENV_VAR)

        logger.debug("cmake_bin: %s", _cmake_bin)
        logger.debug("make_bin: %s", _make_bin)

        self._logger.info("Updating oxDNA parameters")

        build_dir = Path(os.environ[BUILD_PATH_ENV_VAR])
        self._logger.debug("build_dir: %s", build_dir)

        std_out = build_dir / "jax_dna.cmake.std.log"
        std_err = build_dir / "jax_dna.cmake.err.log"
        self._logger.debug(
            "running cmake: std_out->%s, std_err->%s",
            std_out,
            std_err,
        )
        with std_out.open("w") as f_std, std_err.open("w") as f_err:
            if _cmake_bin is None:
                raise FileNotFoundError(ERR_OXDNA_NOT_FOUND.format("cmake"))

            completed_proc = subprocess.run(
                [_cmake_bin, ".."],
                shell=False,  # noqa: S603 false positive
                cwd=build_dir,
                stdout=f_std,
                stderr=f_err,
                check=True,
            )
        self._logger.debug("cmake completed")

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
        std_out = build_dir / "jax_dna.make.std.log"
        std_err = build_dir / "jax_dna.make.err.log"
        self._logger.debug(
            "running make with %d processes: std_out->%s, std_err->%s",
            self.n_build_threads,
            std_out,
            std_err,
        )
        with std_out.open("w") as f_std, std_err.open("w") as f_err:
            completed_proc = subprocess.run(
                [_make_bin, f"-j{self.n_build_threads}"],
                shell=False,  # noqa: S603 false positive
                cwd=build_dir,
                check=True,
                stdout=f_std,
                stderr=f_err,
            )

        if completed_proc.returncode != 0:
            # restore the original src/model.h
            model_h.write_text(orig_text)
            raise ValueError(ERR_BUILD_SETUP_FAILED.format(completed_proc.returncode))

        self._logger.info("oxDNA binary rebuilt")

    def _restore_params(self) -> None:
        """Restore the original parameters."""
        logger.debug("Restoring oxDNA parameters to original values")
        build_dir = Path(os.environ[BUILD_PATH_ENV_VAR])
        old_model_h = build_dir.parent.joinpath("src/model.h.old")
        model_h = build_dir.parent.joinpath("src/model.h")
        if old_model_h.exists():
            # restore the original src/model.h
            old_model_h.replace(model_h)
        else:
            raise FileNotFoundError(ERR_ORIG_MODEL_H_NOT_FOUND.format(old_model_h))
