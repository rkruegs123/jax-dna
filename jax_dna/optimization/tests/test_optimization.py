"""Tests for optimization module."""

from typing import Any

import numpy as np
import pytest

import jax_dna.optimization.optimization as jdna_optimization


def mock_get_fn(value: Any):
    return value


jdna_optimization.get_fn = mock_get_fn
jdna_optimization.wait_fn = jdna_optimization.split_by_ready
jdna_optimization.grad_update_fn = lambda x, y: (x, y)


class RemoteFn:
    def __init__(self, value: Any):
        self.value = value

    def remote(self, *args, **kwargs):  # noqa: ARG002 -- This is just for testing
        return self.value


class MockObjectiveActor:
    def __init__(
        self,
        name: str,
        ready: bool,  # noqa: FBT001 -- This is just for testing
        calc_value: Any = None,
        needed_observables: list[str] = [],  # noqa: B006 -- This is just for testing
    ):
        self._name = name
        self.ready = ready
        self.calc_value = calc_value
        self.needed_obs = needed_observables

    @property
    def name(self):
        return RemoteFn(self._name)

    @property
    def is_ready(self):
        return RemoteFn(self.ready)

    @property
    def calculate(self):
        return RemoteFn(self.calc_value)

    @property
    def needed_observables(self):
        return RemoteFn(self.needed_obs)

    @property
    def update(self):
        self.needed_obs = []
        self.ready = True
        return RemoteFn(None)

    @property
    def post_step(self):
        return RemoteFn(None)


class MockRunningSim:
    def __init__(self, hex_id: str, is_ready: bool = False):  # noqa: FBT001,FBT002 -- This is just for testing
        self.hex_id = hex_id
        self.ready = is_ready

    def task_id(self):
        hex_val = self.hex_id

        class MockHex:
            def hex(self):
                return hex_val

        return MockHex()

    @property
    def is_ready(self):
        return RemoteFn(self.ready)


class MockSimulatorActor:
    def __init__(
        self,
        name: str = "test",
        run_value: Any = None,
        hex_id: str = "1234",
        exposes: list[str] = [],  # noqa: B006 -- This is just for testing
    ):
        self._name = name
        self.run_val = run_value
        self.hex_id = hex_id
        self.expose_values = exposes

    @property
    def name(self):
        return RemoteFn(self._name)

    @property
    def run(self):
        return RemoteFn(MockRunningSim(self.hex_id, is_ready=True))

    @property
    def exposes(self):
        return RemoteFn(self.expose_values)


class MockOptimizer:
    def init(self, params):
        return params

    def update(self, grads, opt_state, params):  # noqa: ARG002 -- This is just for testing
        return {}, opt_state


def test_split_by_ready():
    ready = MockObjectiveActor(name="test", ready=True)
    not_ready = MockObjectiveActor(name="test", ready=False)

    objectives = [ready, not_ready]

    ready, not_ready = jdna_optimization.split_by_ready(objectives)

    assert len(ready) == 1
    assert len(not_ready) == 1

    assert ready[0] == objectives[0]
    assert not_ready[0] == objectives[1]


@pytest.mark.parametrize(
    ("objectives", "simulators", "aggregate_grad_fn", "optimizer", "expected_err"),
    [
        ([], [MockSimulatorActor()], lambda x: x, MockOptimizer(), jdna_optimization.ERR_MISSING_OBJECTIVES),
        (
            [MockObjectiveActor(name="test", ready=True)],
            [],
            lambda x: x,
            MockOptimizer(),
            jdna_optimization.ERR_MISSING_SIMULATORS,
        ),
        (
            [MockObjectiveActor(name="test", ready=True)],
            [MockSimulatorActor()],
            None,
            MockOptimizer(),
            jdna_optimization.ERR_MISSING_AGG_GRAD_FN,
        ),
        (
            [MockObjectiveActor(name="test", ready=True)],
            [MockSimulatorActor()],
            lambda x: x,
            None,
            jdna_optimization.ERR_MISSING_OPTIMIZER,
        ),
    ],
)
def test_optimization_post_init_raises(
    objectives,
    simulators,
    aggregate_grad_fn,
    optimizer,
    expected_err,
):
    with pytest.raises(ValueError, match=expected_err):
        jdna_optimization.Optimization(
            objectives=objectives, simulators=simulators, aggregate_grad_fn=aggregate_grad_fn, optimizer=optimizer
        )


def test_optimzation_step():
    """Test that the optimization step."""

    opt = jdna_optimization.Optimization(
        objectives=[
            MockObjectiveActor(name="test", ready=True, calc_value=1, needed_observables=["q_1"]),
            MockObjectiveActor(name="test", ready=False, calc_value=2, needed_observables=["q_2"]),
        ],
        simulators=[
            MockSimulatorActor(name="test", run_value="test-1", exposes=["q_1"], hex_id="abcd"),
            MockSimulatorActor(name="test", run_value="test-2", exposes=["q_2"], hex_id="1234"),
        ],
        aggregate_grad_fn=np.mean,
        optimizer=MockOptimizer(),
    )

    opt_state, grads = opt.step(params={"test": 1})
    assert opt_state is not None
    assert grads == ({"test": 1}, {})


def test_optimization_post_step():
    """Test that the optimizer state is updated after a step."""
    opt = jdna_optimization.Optimization(
        objectives=[MockObjectiveActor(name="test", ready=True)],
        simulators=[MockSimulatorActor()],
        aggregate_grad_fn=lambda x: x,
        optimizer=MockOptimizer(),
        optimizer_state="old",
    )

    new_state = "new"
    opt = opt.post_step(optimizer_state=new_state, opt_params={})
    assert opt.optimizer_state == new_state
