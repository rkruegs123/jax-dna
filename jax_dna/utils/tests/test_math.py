import jax_dna.utils.math as jdm
import numpy as np
import pytest


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
