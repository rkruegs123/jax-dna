"""Tests for jax_dna.optimization.objective"""

import pathlib
import typing
from collections.abc import Callable

import jax.numpy as jnp
import jax_md
import numpy as np
import pytest

import jax_dna.input.tree as jdna_tree
import jax_dna.optimization.objective as o
import jax_dna.simulators.io as jdna_sio
import jax_dna.utils.types as jdna_types

file_location = pathlib.Path(__file__).parent
data_dir = file_location / "data"


def mock_return_function(should_return: typing.Any) -> Callable:
    """Return a function that returns the given value."""

    def mock_function(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return should_return

    return mock_function


@pytest.mark.parametrize(
    ("required_observables", "needed_observables", "logging_observables", "grad_or_loss_fn", "expected_missing"),
    [
        (None, ["a"], ["c"], lambda x: x, "required_observables"),
        (["a"], None, ["c"], lambda x: x, "needed_observables"),
        (["a"], ["a"], None, lambda x: x, "logging_observables"),
        (["a"], ["a"], ["c"], None, "grad_or_loss_fn"),
    ],
)
def test_objective_init_raises(
    required_observables: list[str],
    needed_observables: list[str],
    logging_observables: list[str],
    grad_or_loss_fn: typing.Callable[[tuple[str, ...]], jdna_types.Grads],
    expected_missing: str,
) -> None:
    """Test the __init__ function for Objective raises for missing required arg."""

    with pytest.raises(ValueError, match=o.ERR_MISSING_ARG.format(missing_arg=expected_missing)):
        o.Objective(
            name="test",
            required_observables=required_observables,
            needed_observables=needed_observables,
            logging_observables=logging_observables,
            grad_or_loss_fn=grad_or_loss_fn,
        )


def test_objective_required_observables() -> None:
    """Test the required_observables property of Objective."""
    obj = o.Objective(
        name="test",
        required_observables=["a"],
        needed_observables=["a"],
        logging_observables=["c"],
        grad_or_loss_fn=lambda x: x,
    )

    assert obj.required_observables() == ["a"]


def test_objective_needed_observables() -> None:
    """Test the needed_observables property of Objective."""
    obj = o.Objective(
        name="test",
        required_observables=["a", "b"],
        needed_observables=["a"],
        logging_observables=["c"],
        grad_or_loss_fn=lambda x: x,
    )

    assert obj.needed_observables() == ["a"]


def test_objective_logging_observables() -> None:
    """Test the logging_observables property of Objective."""
    obj = o.Objective(
        name="test",
        required_observables=["a", "b", "c"],
        needed_observables=[],
        logging_observables=["a", "b"],
        grad_or_loss_fn=lambda x: x,
    )

    # simulate getting the observables
    obj._obtained_observables = [
        ("a", 1.0),
        ("b", 2.0),
        ("c", 3.0),
    ]

    # we are only logging two so we should only get those
    expected = [
        ("a", 1.0),
        ("b", 2.0),
    ]
    assert obj.logging_observables() == expected


@pytest.mark.parametrize(
    ("required_observables", "obtained_observables", "expected"),
    [
        (["a"], [("a", 1.0)], True),
        (["a"], [("b", 1.0)], False),
        (["a", "b"], [("a", 1.0), ("b", 2.0)], True),
        (["a", "b"], [("a", 1.0)], False),
    ],
)
def test_objective_is_ready(
    required_observables: list[str],
    obtained_observables: list[tuple[str, float]],
    expected: bool,  # noqa: FBT001
) -> None:
    """Test the is_ready method of Objective."""
    obj = o.Objective(
        name="test",
        required_observables=required_observables,
        needed_observables=[],
        logging_observables=[],
        grad_or_loss_fn=lambda x: x,
    )

    # simulate getting the observables
    obj._obtained_observables = obtained_observables

    assert obj.is_ready() == expected


@pytest.mark.parametrize(
    ("required_observables", "needed_observables", "update_collection", "expected_obtained", "expected_needed"),
    [
        (["a", "b"], ["a", "b"], [("a", ("test_a", {"test": 1.0}))], [("a", {"test": 1.0})], ["b"]),
        (["a"], ["a"], [("a", ("test_a", {"test": 1.0}))], [("a", {"test": 1.0})], []),
    ],
)
def test_objective_update(
    required_observables: list[str],
    needed_observables: list[str],
    update_collection: list[tuple[str, str]],
    expected_obtained: list[tuple[str, float]],
    expected_needed: list[str],
) -> None:
    """Test the update method of Objective."""

    if not data_dir.exists():
        data_dir.mkdir()

    updates = []
    for update in update_collection:
        observable, (file_name, data) = update
        updates.append(([observable], [data_dir / file_name]))
        jdna_tree.save_pytree(data, data_dir / file_name)

    obj = o.Objective(
        name="test",
        required_observables=required_observables,
        needed_observables=needed_observables,
        logging_observables=[],
        grad_or_loss_fn=lambda x: x,
    )

    obj.update(updates)

    for fname in data_dir.iterdir():
        fname.unlink()
    data_dir.rmdir()

    assert obj.obtained_observables() == expected_obtained
    assert obj.needed_observables() == expected_needed


def test_objective_calculate() -> None:
    """Test the calculate method of Objective."""
    obj = o.Objective(
        name="test",
        required_observables=["a", "b", "c"],
        needed_observables=["a", "b"],
        logging_observables=[],
        grad_or_loss_fn=mock_return_function((1.0, 0.0)),
    )

    # simulate getting the observables
    obj._obtained_observables = [
        ("a", 1.0),
        ("b", 2.0),
        ("c", 3.0),
    ]

    # simulate the calculate
    result = obj.calculate()

    assert result == 1.0


def test_objective_post_step() -> None:
    obj = o.Objective(
        name="test",
        required_observables=["a", "b", "c"],
        needed_observables=["a", "b"],
        logging_observables=[],
        grad_or_loss_fn=lambda x: x,
    )

    # simulate getting the observables
    obj._obtained_observables = [
        ("c", 3.0),
    ]

    # simulate the post step
    obj.post_step(opt_params={})

    assert obj._obtained_observables == []
    assert obj.needed_observables() == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("beta", "new_energies", "ref_energies", "expected_weights", "expected_neff"),
    [
        (1, np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1 / 3, 1 / 3, 1 / 3]), np.array(1.0, dtype=np.float64)),
    ],
)
def test_compute_weights_and_neff(
    beta: float,
    new_energies: np.ndarray,
    ref_energies: np.ndarray,
    expected_weights: np.ndarray,
    expected_neff: float,
) -> None:
    """Test the weights calculation in for a Difftre Objective."""
    weights, neff = o.compute_weights_and_neff(beta, new_energies, ref_energies)
    assert np.allclose(weights, expected_weights)
    assert np.allclose(neff, expected_neff)


@pytest.mark.parametrize(
    (
        "opt_params",
        "energy_fn_builder",
        "beta",
        "ref_states",
        "ref_energies",
        "expected_loss",
        "expected_measured_value",
    ),
    [
        (
            {},
            lambda _: mock_return_function(np.array([1, 2, 3])),
            1.0,
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            0.0,
            ("test", 1.0),
        ),
    ],
)
def test_compute_loss(
    opt_params: dict[str, float],
    energy_fn_builder: typing.Callable[[dict[str, float]], typing.Callable[[np.ndarray], np.ndarray]],
    beta: float,
    ref_states: np.ndarray,
    ref_energies: np.ndarray,
    expected_loss: float,
    expected_measured_value: tuple[str, float],
) -> None:
    """Test the loss calculation in for a Difftre Objective."""
    expected_aux = (np.array(1.0), expected_measured_value, np.array([1, 2, 3]))
    loss_fn = mock_return_function((expected_loss, (expected_measured_value, {})))

    loss, aux = o.compute_loss(opt_params, energy_fn_builder, beta, loss_fn, ref_states, ref_energies)

    assert loss == expected_loss

    def eq(a, b) -> bool:
        if isinstance(a, np.ndarray | jnp.ndarray):
            assert np.allclose(a, b)
        elif isinstance(a, tuple):
            [eq(x, y) for x, y in zip(a, b, strict=True)]
        else:
            assert a == b

    for a, ea in zip(aux, expected_aux, strict=True):
        eq(a, ea)


@pytest.mark.parametrize(
    ("energy_fn_builder", "opt_params", "trajectory_key", "beta", "n_equilibration_steps", "missing_arg"),
    [
        (None, {}, "test", 1.0, 1, "energy_fn_builder"),
        (lambda _: mock_return_function(np.array([1, 2, 3])), None, "test", 1.0, 1, "opt_params"),
        (lambda _: mock_return_function(np.array([1, 2, 3])), {"a": 1}, None, 1.0, 1, "trajectory_key"),
        (lambda _: mock_return_function(np.array([1, 2, 3])), {"a": 1}, "test", None, 1, "beta"),
        (lambda _: mock_return_function(np.array([1, 2, 3])), {"a": 1}, "test", 1.0, None, "n_equilibration_steps"),
    ],
)
def test_difftreobjective_init_raises(
    energy_fn_builder: Callable[[jdna_types.Params], Callable[[np.ndarray], np.ndarray]],
    opt_params: jdna_types.Params,
    trajectory_key: str,
    beta: float,
    n_equilibration_steps: int,
    missing_arg: str,
) -> None:
    required_observables = ["a"]
    needed_observables = ["b"]
    logging_observables = ["c"]
    grad_or_loss_fn = lambda x: x

    with pytest.raises(ValueError, match=o.ERR_MISSING_ARG.format(missing_arg=missing_arg)):
        o.DiffTReObjective(
            name="test",
            required_observables=required_observables,
            needed_observables=needed_observables,
            logging_observables=logging_observables,
            grad_or_loss_fn=grad_or_loss_fn,
            energy_fn_builder=energy_fn_builder,
            opt_params=opt_params,
            trajectory_key=trajectory_key,
            beta=beta,
            n_equilibration_steps=n_equilibration_steps,
        )


def test_difftreobjective_calculate() -> None:
    """Test the calculate method of DifftreObjective."""
    obj = o.DiffTReObjective(
        name="test",
        required_observables=["test"],
        needed_observables=["test"],
        logging_observables=[],
        grad_or_loss_fn=mock_return_function((1.0, (("test", 1.0), {}))),
        energy_fn_builder=lambda _: mock_return_function(np.ones(100)),
        opt_params={"test": 1.0},
        trajectory_key="test",
        beta=1.0,
        n_equilibration_steps=10,
    )

    # simulate getting the observables
    obj._obtained_observables = [
        (
            "test",
            jdna_sio.SimulatorTrajectory(
                rigid_body=jax_md.rigid_body.RigidBody(
                    center=np.arange(110),
                    orientation=jax_md.rigid_body.Quaternion(
                        vec=np.arange(440).reshape(110, 4),
                    ),
                )
            ),
        ),
    ]

    # simulate the calculate
    expected_grad = {"test": jnp.array(0.0)}
    actual_grad = obj.calculate()

    assert actual_grad == expected_grad


def test_difftreobjective_post_step() -> None:
    """test thge post_step method of DiffTReObjective."""
    obj = o.DiffTReObjective(
        name="test",
        required_observables=["test"],
        needed_observables=["test"],
        logging_observables=[],
        grad_or_loss_fn=mock_return_function((1.0, 0.0)),
        energy_fn_builder=lambda _: mock_return_function(np.ones(100)),
        opt_params={"test": 1.0},
        trajectory_key="test",
        beta=1.0,
        n_equilibration_steps=10,
    )

    mock_traj = ("test", "some array data")
    # simulate getting the observables
    obj._obtained_observables = [
        ("test", "some array data"),
        ("loss", 1.0),
    ]

    # run the post step
    new_params = {"test": 2.0}
    obj.post_step(opt_params=new_params)

    assert obj._obtained_observables == [mock_traj]
    assert obj._opt_params == new_params
