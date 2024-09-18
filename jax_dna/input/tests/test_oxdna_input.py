"""Test for oxDNA input file reader."""

from pathlib import Path

import pytest

import jax_dna.input.oxdna_input as oi

TEST_FILES_DIR = Path(__file__).parent / "test_files"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", (1, True)),
        ("-1.5", (-1.5, True)),
        ("1.5.5", (0, False)),
    ],
)
def test_parse_numeric(value: str, expected: tuple[float | int, bool]) -> None:
    """Test _parse_numeric function."""
    assert oi._parse_numeric(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", (True, True)),
        ("false", (False, True)),
        ("TRUE", (True, True)),
        ("fALSe", (False, True)),
        ("Truee", (False, False)),
    ],
)
def test_parse_boolean(value: str, expected: tuple[bool, bool]) -> None:
    """Test _parse_boolean function."""
    assert oi._parse_boolean(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", 1),
        ("-1.5", -1.5),
        ("true", True),
        ("false", False),
        ("string", "string"),
        ("1-numString", "1-numString"),
    ],
)
def test_parse_value(value: str, expected: str | float | bool) -> None:
    """Test _parse_value function."""
    assert oi._parse_value(value) == expected


def test_read() -> None:
    expected = {
        "T": "296.15K",
        "steps": 10000,
        "conf_file": "init.conf",
        "topology": "sys.top",
        "trajectory_file": "output.dat",
        "time_scale": "linear",
        "print_conf_interval": 100,
        "print_energy_every": 100,
        "interaction_type": "DNA_nomesh",
        "seed": 0,
        "lastconf_file": "last_conf.dat",
        "list_type": "no",
        "restart_step_counter": True,
        "energy_file": "energy.dat",
        "equilibration_steps": 0,
    }

    assert oi.read(TEST_FILES_DIR / "test_oxdna_simple_helix_input_trunc.txt") == expected
