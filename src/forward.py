import pdb


from jax import jit
from jax import random
from jax.config import config as jax_config
import jax.numpy as jnp

from jax_md import simulate
from jax_md import space
from jax_md import util
from jax_md.rigid_body import RigidBody, Quaternion
from jax_md import quantity

from utils import DEFAULT_TEMP
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import nucleotide_mass, get_kt, moment_of_inertia
from utils import get_one_hot

from get_params import get_default_params
from trajectory import TrajectoryInfo
from topology import TopologyInfo
from energy import energy_fn_factory

from jax.config import config
config.update("jax_enable_x64", True)

import langevin

f64 = util.f64

mass = RigidBody(center=jnp.array([nucleotide_mass]), orientation=jnp.array([moment_of_inertia]))
base_site = jnp.array(
    [com_to_hb, 0.0, 0.0], dtype=f64
)
stack_site = jnp.array(
    [com_to_stacking, 0.0, 0.0], dtype=f64
)
back_site = jnp.array(
    [com_to_backbone, 0.0, 0.0], dtype=f64
)



def forward(top_info, config_info, steps, t=DEFAULT_TEMP, mgamma=0.1, sim_type="langevin"):
    body = config_info.states[0]

    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    n = top_info.n

    displacement, shift_fn = space.periodic(config_info.box_size)
    key = random.PRNGKey(0)
    # key, pos_key, quat_key = random.split(key, 3)

    energy_fn = energy_fn_factory(displacement,
                                  back_site, stack_site, base_site,
                                  top_info.bonded_nbrs, top_info.unbonded_nbrs)

    params = get_default_params(t=t)

    # Simulate with the energy function via Nose-Hoover
    kT = get_kt(t=t) # 300 Kelvin = 0.1 kT
    dt = 5e-3

    if(sim_type == "langevin"):
       init_fn, step_fn = langevin.nvt_langevin(energy_fn, shift_fn, dt, kT, mgamma) 
       state = init_fn(key, body, mass=mass, seq=seq, params=params)
    elif(sim_type=="nose_hoover"):
       init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)
       state = init_fn(key, body, mass=mass, seq=seq, params=params)
       E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT, seq=seq, params=params)

    step_fn = jit(step_fn)

    trajectory = [state.position]

    for i in range(steps):
        state = step_fn(state, seq=seq, params=params)
        trajectory.append(state.position)

    #pdb.set_trace()

    # E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    # FIXME: store in TrajectoryInfo and write to file (potentially also write topology file)

    print("done")
    return trajectory

if __name__ == "__main__":

    """
    conf_path = "data/polyA_10bp/equilibrated.dat"
    top_path = "data/polyA_10bp/generated.top"
    """
    """
    conf_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/simple-helix/start.conf"
    top_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/simple-helix/generated.top"
    """
    
    conf_path = "/Users/megancengel/Research_apps/jaxmd-oxdna/data/simple-helix/start.conf"
    top_path = "/Users/megancengel/Research_apps/jaxmd-oxdna/data/simple-helix/generated.top"
    
    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)

    forward(top_info, config_info, steps=10, sim_type="langevin")

    """
    TODO:
    - make seq and potential_fns fixed with `functools.partial` (or jnp.Partial)
    - call this file `forward` and move the energy function to its own thing (maybe `energy.py`). And test.
    - start a `optimize_parameters.py` that only makes `seq` fixed and updates `params`, and therefore `potential_fns`
      - see if we can take grads w.r.t. to some dummy loss function
      - first see if we can get non-zero/nan grads, then see if we can optimize some subset of the params dictionary (e.g. we  can overwrite some subset of the dictionary with some parameters that we optimize over)
    """
