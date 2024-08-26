# ruff: noqa: N802,FBT001,FBT002 - Ignore the BaseEnergyFunction/ComposableEnergyFunction names in functions and boolean arg rules
"""Tests for jax_dna.energy.base"""

import re

import jax_md
import numpy as np
import pytest

from jax_dna.energy import base


def _make_base_energy_function(
    with_displacement: bool = False,
) -> base.BaseEnergyFunction | tuple[base.BaseEnergyFunction, jax_md.space.DisplacementFn]:
    """Helper function to create a BaseEnergyFunction."""
    displacement_fn, _ = jax_md.space.free()
    be = base.BaseEnergyFunction(displacement_fn=displacement_fn)
    vals = be
    if with_displacement:
        vals = (be, displacement_fn)
    return vals


def test_BaseEnergyFunction_displacement_mapped() -> None:
    """Tests that the behavior of the displacement function is consistent."""
    a = np.array(
        [
            [1, 1],
            [0, 0],
        ]
    )
    b = np.array([[2, 1], [2, 0]])
    be, displacement_fn = _make_base_energy_function(with_displacement=True)
    np.testing.assert_allclose(be.displacement_mapped(a, b), jax_md.space.map_bond(displacement_fn)(a, b))


def test_BaseEnergyFunction_add() -> None:
    """Test the __add__ function for BaseEnergyFunction."""
    be = _make_base_energy_function()
    actual = be + be
    assert all(isinstance(e, base.BaseEnergyFunction) for e in actual.energy_fns)
    assert actual.weights is None


def test_BaseEnergyFunction_add_raises() -> None:
    """Test the __add__ function for BaseEnergyFunction with invalid args."""
    be = _make_base_energy_function()
    expected_err = re.escape(base.ERR_UNSUPPORTED_OPERATION.format(op="+", left=type(be), right=int))
    with pytest.raises(TypeError, match=expected_err):
        be + 3


def test_BaseEnergyFunction_mul() -> None:
    """Test the __mul__ function for BaseEnergyFunction."""
    coef = 2
    be = _make_base_energy_function()
    actual = be * coef

    assert len(actual.energy_fns) == 1
    assert all(isinstance(e, base.BaseEnergyFunction) for e in actual.energy_fns)
    assert len(actual.weights) == 1
    assert actual.weights[0] == coef


def test_BaseEnergyFunction_mul_raises() -> None:
    """Test the __add__ function for BaseEnergyFunction with invalid args."""
    be = _make_base_energy_function()
    expected_err = re.escape(base.ERR_UNSUPPORTED_OPERATION.format(op="*", left=type(be), right=type(be)))
    with pytest.raises(TypeError, match=expected_err):
        be * be


def test_BaseEnergyFunction_call_raises() -> None:
    """Test the __call__ function for BaseEnergyFunction with invalid args."""
    be = _make_base_energy_function()
    expected_err = re.escape(base.ERR_CALL_NOT_IMPLEMENTED)
    with pytest.raises(NotImplementedError, match=expected_err):
        be(None, None, None, None)


def test_ComposedEnergyFunction_init() -> None:
    """Test the initialization params of ComposedEnergyFunction"""
    be = _make_base_energy_function()
    cef = base.ComposedEnergyFunction(energy_fns=[be])

    assert cef.energy_fns == [be]
    assert cef.weights is None
    assert cef.rigid_body_transform_fn is None


def test_ComposedEnergyFunction_init_raises() -> None:
    """Test the invalid initialization params of ComposedEnergyFunction"""
    be = _make_base_energy_function()
    expected_err = re.escape(base.ERR_COMPOSED_ENERGY_FN_TYPE_ENERGY_FNS)
    with pytest.raises(TypeError, match=expected_err):
        base.ComposedEnergyFunction(energy_fns=[be, 3])


def test_ComposedEnergyFunction_init_raises_lengths() -> None:
    """Test the __call__ function for ComposedEnergyFunction with invalid args."""
    be = _make_base_energy_function()
    with pytest.raises(ValueError, match=re.escape(base.ERR_COMPOSED_ENERGY_FN_LEN_MISMATCH)):
        base.ComposedEnergyFunction(energy_fns=[be], weights=np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    ("init_weights", "expected_weights"),
    [
        (None, None),
        (np.array([1.0]), np.array([1.0, 1.0])),
        (np.array([3.0]), np.array([3.0, 3.0])),
    ],
)
def test_ComposedEnergyFunction_add_energy_function(
    init_weights: np.ndarray | None,
    expected_weights: np.ndarray | None,
) -> None:
    """Test the add_energy_function method of ComposedEnergyFunction."""

    be = _make_base_energy_function()
    cef = base.ComposedEnergyFunction(energy_fns=[be], weights=init_weights)
    cef = cef.add_energy_fn(be, 1.0 if init_weights is None else init_weights[0])

    assert len(cef.energy_fns) == 2  # noqa: PLR2004 ignore magic number error
    if init_weights is None:
        assert cef.weights is None
    else:
        np.testing.assert_allclose(cef.weights, expected_weights)


if __name__ == "__main__":
    pass
