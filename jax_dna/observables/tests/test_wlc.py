"""Tests for the wlc observable."""

import jax.numpy as jnp
import numpy as np
import pytest

import jax_dna.observables.wlc as w
import jax_dna.utils.types as jd_types
import jax_dna.utils.units as jd_units


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


def test_fit_wlc():
    """Test the fit_wlc function."""
    expected = [33.951588, 43.467876, 2131.197638]
    t_kelvin = 296.15
    kT = jd_units.get_kt(t_kelvin)  # noqa: N806 -- kT is a special unit variable

    # Values provided by T. Ouldridge
    per_nuc_forces = jnp.array([0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.375])
    total_forces = per_nuc_forces * 2.0
    extensions = jnp.array([35.0, 36.67, 37.84, 38.37, 38.71, 38.98, 39.19, 39.46])

    init_guess = jnp.array([39.87, 50.60, 44.54])  # initialize to the true values

    # Perform the fit
    res = w.fit_wlc(extensions, total_forces, init_guess, kT)
    l0_nm = res[0] * jd_units.NM_PER_OXDNA_LENGTH
    lp_nm = res[1] * jd_units.NM_PER_OXDNA_LENGTH
    k_pn = res[2] * jd_units.PN_PER_OXDNA_FORCE
    np.testing.assert_allclose(
        [l0_nm, lp_nm, k_pn],
        expected,
        rtol=0.0001,
    )
