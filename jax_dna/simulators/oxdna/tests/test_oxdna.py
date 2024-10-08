"""Tests for oxDNA simulator."""

import os
import shutil
import uuid
from pathlib import Path

import pytest
from jax_dna.simulators import oxdna

file_dir = Path(os.path.realpath(__file__)).parent


def setup_test_dir(add_input: bool = True):  # noqa: FBT001,FBT002
    """Setup the test directory."""
    test_dir = file_dir / f"test_data/{uuid.uuid4()}"
    test_dir.mkdir(parents=True)
    if add_input:
        with (test_dir / "input").open("w") as f:
            f.write("trajectory_file = test.conf\ntopology = test.top\n")

        shutil.copyfile(
            "data/test-data/simple-helix/generated.top",
            test_dir / "test.top",
        )
        shutil.copyfile(
            "data/test-data/simple-helix/start.conf",
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
    sim = oxdna.oxDNASimulator(input_dir=test_dir)
    tear_down_test_dir(test_dir)
    assert str(sim["input_dir"]) == str(test_dir)


def test_oxdna_run_raises_fnf():
    """Test that the oxDNA simulator raises FileNotFoundError."""
    test_dir = setup_test_dir(add_input=False)
    sim = oxdna.oxDNASimulator(input_dir=test_dir)
    with pytest.raises(FileNotFoundError, match=oxdna.ERR_INPUT_FILE_NOT_FOUND[:10]):
        sim.run()
    tear_down_test_dir(test_dir)


def test_oxdna_run_raises_bin_path_not_set():
    """Test that the oxDNA simulator raises ValueError."""
    test_dir = setup_test_dir()
    sim = oxdna.oxDNASimulator(input_dir=test_dir)
    with pytest.raises(ValueError, match=oxdna.ERR_BIN_PATH_NOT_SET[:10]):
        sim.run()
    tear_down_test_dir(test_dir)


def test_oxdna_run():
    """Test the oxDNA simulator run function."""
    os.environ[oxdna.BIN_PATH_ENV_VAR] = "echo"
    test_dir = setup_test_dir()
    sim = oxdna.oxDNASimulator(input_dir=test_dir)
    sim.run()
    with (test_dir / "oxdna.log").open() as f:
        assert f.read() == "input\n"
    tear_down_test_dir(test_dir)
    del os.environ[oxdna.BIN_PATH_ENV_VAR]
