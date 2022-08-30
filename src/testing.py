import pdb

from jax.config import config as jax_config
import jax.numpy as jnp
import numpy as onp
from jax import jit

from jax_md import simulate
from jax_md import space
from jax_md import util
from jax_md import rigid_body
from jax_md.rigid_body import RigidBody, Quaternion

from utils import DEFAULT_TEMP
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import get_one_hot

from nn_energy import nn_energy_fn_factory
from other_pairs_energy import other_pairs_energy_fn_factory_fixed
from trajectory import TrajectoryInfo
from topology import TopologyInfo
from get_params import get_default_params

from jax.config import config
config.update("jax_enable_x64", True)

FLAGS = jax_config.FLAGS
DYNAMICS_STEPS = 100

f32 = util.f32
f64 = util.f64

DTYPE = [f32]
if FLAGS.jax_enable_x64:
    DTYPE += [f64]



if __name__ == "__main__":

    #import matplotlib.pyplot as plt
    from tqdm import tqdm

    """
    from potential import f1_dr_stack

    rs = onp.linspace(0.2, 0.6, 20)
    ys = f1_dr_stack(rs)
    plt.plot(rs, ys)
    plt.show()
    pdb.set_trace()
    """



    """
    conf_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/oxpy-testing/relaxed.conf"
    traj_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/oxpy-testing/output.dat"
    top_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/oxpy-testing/test.top"
    """

    # traj_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/test.dat"
    # top_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/polyA_10bp/generated.top"

    top_path = "data/polyA_10bp/generated.top"
    traj_path = "data/polyA_10bp/test_nose_hoover.dat"


    top_info = TopologyInfo(top_path, reverse_direction=True)
    traj_info = TrajectoryInfo(top_info, traj_path=traj_path, reverse_direction=True)

    params = get_default_params()
    seq_oh = jnp.array(get_one_hot(traj_info.top_info.seq), dtype=f64)

    displacement, shift = space.periodic(traj_info.box_size)

    base_site = jnp.array(
        [com_to_hb, 0.0, 0.0]
    )
    stack_site = jnp.array(
        [com_to_stacking, 0.0, 0.0]
    )
    back_site = jnp.array(
        [com_to_backbone, 0.0, 0.0]
    )



    _, nn_subterms_fn = nn_energy_fn_factory(displacement,
                                                     back_site=back_site,
                                                     stack_site=stack_site,
                                                     base_site=base_site,
                                                     neighbors=traj_info.top_info.bonded_nbrs)

    _, other_pairs_subterms_fn = other_pairs_energy_fn_factory_fixed(displacement,
                                                             back_site=back_site,
                                                             stack_site=stack_site,
                                                             base_site=base_site,
                                                             neighbors=traj_info.top_info.unbonded_nbrs)



    nn_subterms_fn = jit(nn_subterms_fn)
    # dynamic_energy_fn = jit(dynamic_energy_fn)
    es = list()
    n = traj_info.top_info.n
    for state in tqdm(traj_info.states):
        fene, exc_vol_bonded, stack = nn_subterms_fn(state, seq_oh, params)
        exc_vol_unbonded, v_hb, cross_stack, coax_stack = other_pairs_subterms_fn(state, seq_oh, params)

        avg_subterms = [
            ('backbone', fene / n),
            ('exc_vol_bonded', exc_vol_bonded / n),
            ('stack', stack / n),
            ('exc_vol_unbonded', exc_vol_unbonded / n),
            ('v_hb', v_hb / n),
            ('cross_stack', cross_stack / n),
            ('coax_stack', coax_stack / n)
        ]
        pdb.set_trace()
        # es.append(fene + exc_vol_bonded + stack + exc_vol_unbonded)
        es.append(fene + exc_vol_bonded + stack)

    plt.plot(list(range(len(es))), es)
    plt.show()

    pdb.set_trace()
    print("done")
