import jax
import numpy as np
import pytest
import sympy as sp

from jax_dna.energy import potentials
from jax_dna.energy.tests import symbolic_potentials

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

# functional forms from oxDNA paper
# https://ora.ox.ac.uk/objects/uuid:b2415bb2-7975-4f59-b5e2-8c022b4a3719/files/mdcac62bc9133143fc05070ed20048c50
# Section 2.4.1


@pytest.mark.parametrize(
    ("r", "eps", "r0", "delt"),
    [
        (5, 3, 2, 10),
    ],
)
def test_v_fene(r: float, eps: float, r0: float, delt: float):
    """tests v_fene potential. Function is equation 2.1 form the oxDNA paper."""
    actual = potentials.v_fene(r, eps, r0, delt)
    expected = symbolic_potentials.v_fene(r, eps, r0, delt)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("r", "eps", "r0", "a"),
    [
        (5, 3, 2, 10),
    ],
)
def test_v_morse(r: float, eps: float, r0: float, a: float):
    """tests v_morse potential. Function is equation 2.2 form the oxDNA paper."""
    actual = float(potentials.v_morse(r, eps, r0, a))
    expected = symbolic_potentials.v_morse(r, eps, r0, a)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("r", "k", "r0"),
    [
        (5, 3, 2),
    ],
)
def test_v_harmonic(r: float, k: float, r0: float):
    """tests v_harmonic potential. Function is equation 2.3 form the oxDNA paper."""
    actual = float(potentials.v_harmonic(r, k, r0))
    expected = symbolic_potentials.v_harmonic(r, k, r0)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("r", "k", "r0"),
    [
        (5, 3, 2),
    ],
)
def test_harmonic(r: float, k: float, r0: float):
    """tests harmonic potential. Function is equation 2.3 form the oxDNA paper."""
    actual = float(potentials.v_harmonic(r, k, r0))
    expected = symbolic_potentials.v_harmonic(r, k, r0)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("r", "eps", "sigma"),
    [
        (5, 3, 2),
    ],
)
def test_v_lj(r: float, eps: float, sigma: float):
    """tests Lennard-Jones potential. Function is equation 2.4 form the oxDNA paper."""
    actual = float(potentials.v_lj(r, eps, sigma))
    expected = symbolic_potentials.v_lj(r, eps, sigma)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("theta", "a", "theta0"),
    [
        (5, 3, 2),
    ],
)
def test_v_mod(theta: float, a: float, theta0: float):
    """tests modified potential. Function is equation 2.5 form the oxDNA paper."""
    actual = float(potentials.v_mod(theta, a, theta0))
    expected = symbolic_potentials.v_mod(theta, a, theta0)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("x", "b", "x_c"),
    [
        (5, 3, 2),
    ],
)
def test_v_smooth(x: float, b: float, x_c: float):
    """tests smooth potential. Function is equation 2.6 form the oxDNA paper."""
    actual = float(potentials.v_smooth(x, b, x_c))
    expected = symbolic_potentials.v_smooth(x, b, x_c)
    np.testing.assert_allclose(actual, expected)
