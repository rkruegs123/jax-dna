import typing

import ray

import jax_dna.utils.types as jdna_types
import jax_dna.simulators.io as jdna_sio

@ray.remote
class SimulatorActor:
    def __init__(
        self,
        fn: typing.Callable[[jdna_types.Params, jdna_types.MetaData], tuple[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData]],
        exposes: list[str],
        meta_data: dict[str, typing.Any],
        writer_fn: typing.Callable[[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData, jdna_types.PathOrStr], None] = None,
    ):
        self._fn = fn
        self._exposes = exposes
        self._meta_data = meta_data
        self._writer_fn = writer_fn


    def exposes(self) -> list[str]:
        return self._exposes


    def run(
        self,
        params: jdna_types.Params,
    ) -> tuple[jdna_sio.SimulatorTrajectory, jdna_sio.SimulatorMetaData, dict[str, typing.Any]]:
        outs, aux = self._fn(params, self._meta_data)

        if self._writer_fn is not None:
            return self._writer_fn(outs, aux, self._meta_data)

        return outs, aux, self._meta_data