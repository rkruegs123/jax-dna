"""Tests for the wlc observable."""

import jax.numpy as jnp
import numpy as np
import pytest

import jax_dna.observables.wlc as w
import jax_dna.utils.types as jd_types


@pytest.mark.parametrize(
    ("x"),
    [
        (1),
        (2),
        (3),
    ],
)
def test_coth(x: float) -> None:
    """Test the coth function."""

    # https://mathworld.wolfram.com/HyperbolicCotangent.html
    # equation 1
    def expected(x: float) -> float:
        return (np.exp(2 * x) + 1) / (np.exp(2 * x) - 1)

    np.testing.assert_allclose(w.coth(x), expected(x))


def orig_extension(
    force: float,
    l0: float,
    lp: float,
    k: float,
    kT: float,  # noqa: N803 -- kT is a special unit variable
) -> float:
    y = ((force * l0**2) / (lp * kT)) ** (1 / 2)
    return l0 * (1 + force / k - kT / (2 * force * l0) * (1 + y * w.coth(y)))


@pytest.mark.parametrize(
    ("force", "l0", "lp", "k", "kT"),
    [
        (1, 1, 1, 1, 1),
        (jnp.ones(3), jnp.ones(3), jnp.ones(3), jnp.ones(3), 1),
    ],
)
def test_calculate_extension(
    force: jd_types.ARR_OR_SCALAR,
    l0: jd_types.ARR_OR_SCALAR,
    lp: jd_types.ARR_OR_SCALAR,
    k: jd_types.ARR_OR_SCALAR,
    kT: float,  # noqa: N803 -- kT is a special unit variable
) -> None:
    """Test the calculate_extension function."""

    # this assumes the original implementation is correct
    np.testing.assert_allclose(w.calculate_extension(force, l0, lp, k, kT), orig_extension(force, l0, lp, k, kT))


@pytest.mark.parametrize(
    ("coeffs", "extensions", "forces", "kT"),
    [
        (jnp.ones(3), jnp.ones(3), jnp.ones(3), 1),
    ],
)
def test_loss(
    coeffs: jd_types.ARR_OR_SCALAR,
    extensions: jd_types.ARR_OR_SCALAR,
    forces: jd_types.ARR_OR_SCALAR,
    kT: float,  # noqa: N803 -- kT is a special unit variable
) -> None:
    """Test the loss function."""

    # this assumes the original implementation is correct
    ext = orig_extension(forces, coeffs[0], coeffs[1], coeffs[2], kT)
    expected = extensions - ext

    np.testing.assert_allclose(w.loss(coeffs, extensions, forces, kT), expected)


if __name__ == "__main__":
    test_loss()
