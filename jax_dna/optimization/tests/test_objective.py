"""Tests for jax_dna.optimization.objective"""

import pathlib
import typing

import pytest

import jax_dna.input.tree as jdna_tree
import jax_dna.optimization.objective as o
import jax_dna.utils.types as jdna_types

file_location = pathlib.Path(__file__).parent
data_dir = file_location / "data"


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
            required_observables=required_observables,
            needed_observables=needed_observables,
            logging_observables=logging_observables,
            grad_or_loss_fn=grad_or_loss_fn,
        )


def test_objective_required_observables() -> None:
    """Test the required_observables property of Objective."""
    obj = o.Objective(
        required_observables=["a"], needed_observables=["a"], logging_observables=["c"], grad_or_loss_fn=lambda x: x
    )

    assert obj.required_observables() == ["a"]


def test_objective_needed_observables() -> None:
    """Test the needed_observables property of Objective."""
    obj = o.Objective(
        required_observables=["a", "b"],
        needed_observables=["a"],
        logging_observables=["c"],
        grad_or_loss_fn=lambda x: x,
    )

    assert obj.needed_observables() == ["a"]


def test_objective_logging_observables() -> None:
    """Test the logging_observables property of Objective."""
    obj = o.Objective(
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
        required_observables=required_observables,
        needed_observables=[],
        logging_observables=[],
        grad_or_loss_fn=lambda x: x,
    )

    # simulate getting the observables
    obj._obtained_observables = obtained_observables

    assert obj.is_ready(params={}) == expected


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
        required_observables=required_observables,
        needed_observables=needed_observables,
        logging_observables=[],
        grad_or_loss_fn=lambda x: x,
    )

    obj.update(updates)

    for fname in data_dir.iterdir():
        fname.unlink()
    data_dir.rmdir()

    assert obj._obtained_observables == expected_obtained
    assert obj.needed_observables() == expected_needed


def test_objective_post_step() -> None:
    obj = o.Objective(
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
