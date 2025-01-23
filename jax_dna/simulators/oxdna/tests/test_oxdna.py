"""Tests for oxDNA simulator."""

import os
import shutil
import uuid
from pathlib import Path

import jax_dna.utils.types as typ
import pytest
from jax_dna.simulators import oxdna

file_dir = Path(os.path.realpath(__file__)).parent


def test_guess_binary_location() -> None:
    """tests the guess_binary_location function."""

    assert oxdna._guess_binary_location("bash", "OXDNA_BIN_PATH") is not None
    assert oxdna._guess_binary_location("zamboomafoo", "MAKE_BIN_PATH") is None


def setup_test_dir(add_input: bool = True):  # noqa: FBT001,FBT002
    """Setup the test directory."""
    test_dir = file_dir / f"test_data/{uuid.uuid4()}"
    test_dir.mkdir(parents=True)
    if add_input:
        with (test_dir / "input").open("w") as f:
            f.write("trajectory_file = test.conf\ntopology = test.top\n")

        shutil.copyfile(
            "data/test-data/dna1/simple-helix/generated.top",
            test_dir / "test.top",
        )
        shutil.copyfile(
            "data/test-data/dna1/simple-helix/start.conf",
            test_dir / "test.conf",
        )
    return test_dir


def tear_down_test_dir(test_dir: str):
    """Tear down the test directory."""
    shutil.rmtree(test_dir)

    if len(os.listdir(Path(test_dir).parent)) == 0:
        test_dir.parent.rmdir()


def test_oxdna_init():
    """Test the oxDNA simulator initialization."""
    test_dir = setup_test_dir()
    sim = oxdna.oxDNASimulator(
        input_dir=test_dir,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[],
    )
    tear_down_test_dir(test_dir)
    assert str(sim["input_dir"]) == str(test_dir)


def test_oxdna_run_raises_fnf():
    """Test that the oxDNA simulator raises FileNotFoundError."""
    test_dir = setup_test_dir(add_input=False)
    sim = oxdna.oxDNASimulator(
        input_dir=test_dir,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[],
    )
    with pytest.raises(FileNotFoundError, match=oxdna.ERR_INPUT_FILE_NOT_FOUND[:10]):
        sim.run()
    tear_down_test_dir(test_dir)


def test_oxdna_run_raises_bin_path_not_set():
    """Test that the oxDNA simulator raises ValueError."""
    test_dir = setup_test_dir()
    sim = oxdna.oxDNASimulator(
        input_dir=test_dir,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[],
    )
    with pytest.raises(ValueError, match=oxdna.ERR_BIN_PATH_NOT_SET[:10]):
        sim.run()
    tear_down_test_dir(test_dir)


def test_oxdna_run():
    """Test the oxDNA simulator run function."""
    os.environ[oxdna.BIN_PATH_ENV_VAR] = "echo"
    test_dir = setup_test_dir()
    sim = oxdna.oxDNASimulator(
        input_dir=test_dir,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[],
    )
    sim.run()
    with (test_dir / "oxdna.out.log").open() as f:
        assert f.read() == "input\n"
    tear_down_test_dir(test_dir)
    del os.environ[oxdna.BIN_PATH_ENV_VAR]


def test_oxdna_restore_params() -> None:
    """Tests oxdna restore params"""

    test_dir = setup_test_dir()
    tmp_build_dir = test_dir / "build"

    expected_text = "Testing text"
    tmp_src = test_dir / "src"
    tmp_src.mkdir()

    (tmp_src / "model.h.old").write_text(expected_text)
    (tmp_src / "model.h").write_text("Will be removed")

    os.environ[oxdna.BUILD_PATH_ENV_VAR] = str(tmp_build_dir)

    sim = oxdna.oxDNASimulator(
        input_dir=test_dir,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[],
    )
    sim._restore_params()

    assert (tmp_src / "model.h").read_text() == expected_text
    assert not (tmp_src / "model.h.old").exists()

    tear_down_test_dir(test_dir)


def test_oxdna_update_params_raises() -> None:
    """Test for oxdna _update_params, fails for missing build dir"""

    test_dir = setup_test_dir()

    if os.environ.get(oxdna.BUILD_PATH_ENV_VAR):
        del os.environ[oxdna.BUILD_PATH_ENV_VAR]

    sim = oxdna.oxDNASimulator(
        input_dir=test_dir,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[],
    )

    with pytest.raises(ValueError, match=oxdna.ERR_BUILD_PATH_NOT_SET[:10]):
        sim._update_params(new_params=[{}])

    tear_down_test_dir(test_dir)


def test_oxdna_update_params() -> None:
    """Test for oxdna _update_params, fails for missing build dir"""

    test_dir = setup_test_dir()
    build_dir = Path(test_dir) / "build"
    build_dir.mkdir()
    src_dir = Path(test_dir) / "src"
    src_dir.mkdir()

    model_h = test_dir.parent.parent / "test_data/test.model.h"

    (src_dir / "model.h").write_text(model_h.read_text())

    os.environ[oxdna.BUILD_PATH_ENV_VAR] = str(build_dir)
    os.environ[oxdna.CMAKE_BIN_ENV_VAR] = "echo"
    os.environ[oxdna.MAKE_BIN_ENV_VAR] = "echo"

    class MockEnergyConfig:
        def __init__(self, params):
            self.params = params

        def init_params(self) -> "MockEnergyConfig":
            return self

        def to_dictionary(self, include_dependent, exclude_non_optimizable) -> dict:  # noqa: ARG002
            return self.params

        def __or__(self, other: dict):
            return MockEnergyConfig(self.params | other)

    oxdna.oxDNASimulator(
        input_dir=test_dir,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[MockEnergyConfig({}), MockEnergyConfig({})],
    )._update_params(
        new_params=[
            {
                "FENE_DELTA": 5.0,
                "HYDR_THETA8_T0": 1.5707963267948966,
                "HYDR_T3_MESH_POINTS": "HYDR_T2_MESH_POINTS",
                "CXST_T5_MESH_POINTS": 6,
            },
            {},
        ]
    )

    assert (src_dir / "model.h.old").read_text().splitlines()[-10:] == model_h.read_text().splitlines()[-10:]
    assert (src_dir / "model.h").read_text().splitlines()[-10:] != (
        test_dir.parent / "expected.model.h"
    ).read_text().splitlines()[-10:]

    for env_var in [oxdna.BUILD_PATH_ENV_VAR, oxdna.CMAKE_BIN_ENV_VAR, oxdna.MAKE_BIN_ENV_VAR]:
        del os.environ[env_var]

    tear_down_test_dir(test_dir)


if __name__ == "__main__":
    test_oxdna_update_params()
