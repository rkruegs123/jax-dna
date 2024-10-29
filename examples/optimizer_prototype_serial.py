
import cloudpickle
import jax
import jax_md
from jax import export

import jax_dna.simulators.io as jd_sio


export.register_pytree_node_serialization(
    jd_sio.SimulatorTrajectory,
    serialized_name="jd_sio.SimulatorTrajectory",
    serialize_auxdata=lambda x: bytes(cloudpickle.dumps(x)),
    deserialize_auxdata=lambda x: cloudpickle.loads(x),
)

export.register_pytree_node_serialization(
    jd_sio.SimulatorMetaData,
    serialized_name="jd_sio.SimulatorMetaData",
    serialize_auxdata=lambda x: bytes(cloudpickle.dumps(x)),
    deserialize_auxdata=lambda x: cloudpickle.loads(x),
)

def serialize_quaternion_aux(q: tuple):
    return bytes(cloudpickle.dumps(q))

def deserialize_quaternion_aux(vec: tuple[jax.numpy.ndarray]):
    return cloudpickle.loads(vec)


export.register_pytree_node_serialization(
    jax_md.rigid_body.Quaternion,
    serialized_name="jax_md.rigid_body.Quaternion",
    serialize_auxdata=serialize_quaternion_aux,
    deserialize_auxdata=deserialize_quaternion_aux,
)
