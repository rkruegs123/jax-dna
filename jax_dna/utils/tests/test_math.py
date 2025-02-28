import numpy as np
import pytest

import jax_dna.utils.math as jdm
import jax_dna.utils.tests.symbolic_math as sm


@pytest.mark.parametrize(
    ("x", "y", "z", "expected_psi", "expected_theta", "expected_phi"),
    [
        (
            np.array([[1, 0, 0]]),
            np.array([[0, 1, 0]]),
            np.array([[0, 0, 1]]),
            np.array([0]),
            np.array([0]),
            np.array([0]),
        ),
        (
            np.array([[1 / np.sqrt(2), 1 / np.sqrt(2), 0]]),
            np.array([[1 / np.sqrt(2), -1 / np.sqrt(2), 0]]),
            np.array([[0, 0, 1]]),
            np.array([np.pi / 4]),
            np.array([0]),
            np.array([0]),
        ),
    ],
)
def test_principal_axes_to_euler_angles(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    expected_psi: np.ndarray,
    expected_theta: np.ndarray,
    expected_phi: np.ndarray,
):
    psi, theta, phi = jdm.principal_axes_to_euler_angles(x, y, z)

    assert np.allclose(psi, expected_psi)
    assert np.allclose(theta, expected_theta)
    assert np.allclose(phi, expected_phi)


@pytest.mark.parametrize(
    ("psi", "theta", "phi", "expected_quaternion"),
    [
        (
            np.array([0]),
            np.array([0]),
            np.array([0]),
            np.array([1, 0, 0, 0]),
        ),
    ],
)
def test_euler_angles_to_quaternion(
    psi: np.ndarray, theta: np.ndarray, phi: np.ndarray, expected_quaternion: np.ndarray
):
    quaternion = jdm.euler_angles_to_quaternion(psi, theta, phi)
    assert np.allclose(quaternion, expected_quaternion)


@pytest.mark.parametrize(
    ("x", "eps"),
    [
        (-3.0, 1e-10),
        (0.0, 1e-10),
        (3.0, 1e-10),
    ],
)
def test_smooth_abs(x: float, eps: float):
    """Test the smooth_abs function."""
    assert np.allclose(jdm.smooth_abs(x, eps), sm.smooth_abs(x, eps))


@pytest.mark.parametrize(
    ("x", "lo", "hi", "expected"),
    [
        (-3.0, -1.0, 1.0, -1.0),
        (3.0, -1.0, 1.0, 1.0),
        (0.0, -1.0, 1.0, 0.0),
    ],
)
def test_clamp(x: float, lo: float, hi: float, expected: float) -> float:
    """Test the clamp function."""
    assert np.allclose(jdm.clamp(x, lo, hi), sm.clamp(x, lo, hi))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])),
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])),
    ],
)
def test_mult(a: np.ndarray, b: np.ndarray):
    """Test the mult function."""

    assert np.allclose(jdm.mult(a, b), (a * b).sum(axis=1))
