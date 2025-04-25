# ruff: noqa: N802,FBT001
"""Tests for jax_dna.energy.configuration"""

from typing import Any

import chex
import pytest

from jax_dna.energy import configuration


@chex.dataclass(frozen=True)
class MockConfig(configuration.BaseConfiguration):
    a: float | None = None
    b: float | None = None
    c: float | None = None
    d: float | None = None
    required_params: tuple[str] = ("a", "b", "c")
    non_optimizable_required_params: tuple[str] = ("c",)
    dependent_params: tuple[str] = ("d",)


def test_BaseConfiguration_init_raises_missing_required_params() -> None:
    """Tests the initialization of BaseConfiguration."""

    with pytest.raises(ValueError, match=configuration.ERR_MISSING_REQUIRED_PARAMS.format(props="c")):
        MockConfig(a=1, b=2)


def test_BaseConfiguration_init_unoptimizable_params():
    """Tests the initialization of BaseConfiguration."""

    with pytest.raises(
        ValueError, match=configuration.ERR_OPT_DEPENDENT_PARAMS.format(req_params="a,b", given_params="d")
    ):
        MockConfig(a=1, b=2, c=4, params_to_optimize=("a", "b", "d"))

    MockConfig(a=1, b=2, c=4, params_to_optimize=configuration.BaseConfiguration.OPT_ALL)


@pytest.mark.parametrize(
    ("in_config", "expected"),
    [
        (
            {"a": 1, "b": 2, "c": 3, "params_to_optimize": ("a",)},
            {
                "a": 1,
            },
        ),
        (
            {"a": 1, "b": 2, "c": 3, "d": 4, "params_to_optimize": configuration.BaseConfiguration.OPT_ALL},
            {"a": 1, "b": 2},
        ),
        ({"a": 1, "b": 2, "c": 3, "d": 4, "params_to_optimize": ()}, {}),
    ],
)
def test_BaseConfiguration_opt_params(in_config: dict[str, Any], expected: dict[str, Any]) -> None:
    """Tests the opt_params property of BaseConfiguration."""

    test_config = MockConfig(**in_config)
    assert test_config.opt_params == expected


def test_BaseConfiguration_init_params() -> None:
    """Tests the init_params method of BaseConfiguration."""

    with pytest.warns(Warning, match=configuration.WARN_INIT_PARAMS_NOT_IMPLEMENTED):
        test_config = MockConfig(a=1, b=2, c=3)
        test_config.init_params()


def test_BaseConfiguration_from_dict() -> None:
    """Tests the from_dict method of BaseConfiguration."""

    test_dict = {"a": 1, "b": 2, "c": 3, "params_to_optimize": ("a",)}
    updated_params = {"params_to_optimize": ("a", "b")}
    test_config = MockConfig.from_dict(test_dict, params_to_optimize=updated_params["params_to_optimize"])
    assert test_config.opt_params == {"a": 1, "b": 2}


@pytest.mark.parametrize(
    ("merged_object", "expected", "raises"),
    [
        (MockConfig(a=4, b=5, c=6), {"a": 4, "b": 5, "c": 6}, False),
        ({"a": 4, "b": 5, "c": 6}, {"a": 4, "b": 5, "c": 6}, False),
        (3, {}, True),
    ],
)
def test_BaseConfiguration_or(merged_object: Any, expected: dict, raises: bool) -> None:
    init_config = MockConfig(a=1, b=2, c=3)

    if raises:
        with pytest.raises(TypeError):
            init_config | merged_object
    else:
        out_config = init_config | merged_object
        assert all(out_config[k] == expected[k] for k in expected)


def test_BaseConfiguration_to_dictionary() -> None:
    """Tests the to_dictionary method of BaseConfiguration."""

    test_config = MockConfig(a=1, b=2, c=3, d=4)
    assert test_config.to_dictionary(
        include_dependent=False,
        exclude_non_optimizable=True,
    ) == {"a": 1, "b": 2}

    assert test_config.to_dictionary(
        include_dependent=True,
        exclude_non_optimizable=True,
    ) == {"a": 1, "b": 2, "d": 4}

    assert test_config.to_dictionary(
        include_dependent=True,
        exclude_non_optimizable=False,
    ) == {"a": 1, "b": 2, "c": 3, "d": 4}
