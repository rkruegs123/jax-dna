"""Tests for jax_dna.input.toml"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jax_dna.input import toml

TEST_FILES_DIR = Path(__file__).parent / "test_files"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3.0000", 3.0),
        ("-3.0", -3.0),
        ("3.0e-1", 0.3),
        ("3 * pi", 3 * np.pi),
        ("3**2", 9),
        ("2 + 2", 4),
        ("Hello", "Hello"),
        ("x", "x"),
    ],
)
def test_parse_str(value: str, expected: str) -> None:
    """Tests the jax_dna.input.toml.parse_str function."""
    result = toml.parse_str(value)
    assert result == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3.0", 3.0),
        ([1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0])),
        (["3.0", "2+2", "3 * pi"], [3.0, 4, 3 * np.pi]),
    ],
)
def test_parse_value(value: str | float | list[str] | list[float], expected: str | float | np.ndarray) -> None:
    """Tests the jax_dna.input.toml.parse_value function."""
    result = toml.parse_value(value)
    if isinstance(result, np.ndarray):
        np.testing.assert_allclose(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("file_path", "key", "expected"),
    [
        ("test_flat.toml", None, {"testing": "test", "number": 42, "arr": [1, 2, 3], "math": np.pi * 3}),
        ("test.toml", "key1", {"hello": "world", "number": 42}),
        ("test.toml", "key2", {"pi": np.pi, "math": 4}),
    ],
)
def test_parse_toml(
    file_path: Path | str,
    key: str | None,
    expected: dict[str, Any],
) -> None:
    """Tests the jax_dna.input.toml.parse_toml function."""
    result = toml.parse_toml(TEST_FILES_DIR / Path(file_path), key)
    assert result
