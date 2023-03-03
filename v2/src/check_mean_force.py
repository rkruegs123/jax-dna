import pdb
import jax
from jax import jit, vmap
import jax.numpy as jnp
from jax import random
from jax.tree_util import Partial
from tqdm import tqdm
import datetime
from pathlib import Path
import shutil
from functools import partial

from jax_md.rigid_body import RigidBody, Quaternion
from jax_md.rigid_body import _quaternion_multiply, _quaternion_conjugate
from jax_md import space, util, simulate, quantity

from utils import nucleotide_mass, get_kt, moment_of_inertia, get_one_hot, DEFAULT_TEMP
from utils import base_site, stack_site, back_site
from utils import bcolors
# import langevin
from energy import factory, ext_force
from loader import get_params
from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo

from jax.config import config
config.update("jax_enable_x64", True)

f64 = util.f64


if __name__ == "__main__":
    basedir = Path("/n/brenner_lab/User/rkrueger/jaxmd-oxdna/v2/data/output/langevin_2023-02-28_00-51-21_n20000000_k10")
    top_path = basedir / "generated.top"
    traj_path = basedir / "output.dat"

    traj_path1 = "/n/brenner_lab/User/rkrueger/jaxmd-oxdna/v2/data/output/langevin_2023-02-27_23-51-26_n20000000_k1/output.dat"
    traj_path2 = "/n/brenner_lab/User/rkrueger/jaxmd-oxdna/v2/data/output/langevin_2023-02-27_23-51-53_n20000000_k2/output.dat"
    traj_path3 = "/n/brenner_lab/User/rkrueger/jaxmd-oxdna/v2/data/output/langevin_2023-02-27_23-52-01_n20000000_k3/output.dat"
    traj_path4 = "/n/brenner_lab/User/rkrueger/jaxmd-oxdna/v2/data/output/langevin_2023-02-27_23-52-01_n20000000_k4/output.dat"

    # all_traj_paths = [traj_path, traj_path1, traj_path2, traj_path3, traj_path4]
    all_traj_paths = [traj_path2, traj_path3]
    # all_traj_paths = [traj_path]
    

    print("Loading trajectories...")
    top_info = TopologyInfo(top_path, reverse_direction=True)
    all_traj_info = [TrajectoryInfo(top_info, traj_path=t_path, reverse_direction=True) for t_path in tqdm(all_traj_paths)]
    # traj_info = TrajectoryInfo(top_info, traj_path=traj_path, reverse_direction=True) # traj_info.states is trajectory

    use_ext_force = True
    ext_force_magnitude = 0.025
    ext_force_bps1 = [5, 214]
    ext_force_bps2 = [104, 115]

    mass = RigidBody(center=jnp.array([nucleotide_mass], dtype=f64),
                     orientation=jnp.array([moment_of_inertia], dtype=f64))

    init_fene_params = [2.0, 0.25, 0.7525]
    init_stacking_params = [
        1.3448, 2.6568, 6.0, 0.4, 0.9, 0.32, 0.75, # f1(dr_stack)
        1.30, 0.0, 0.8, # f4(theta_4)
        0.90, 0.0, 0.95, # f4(theta_5p)
        0.90, 0.0, 0.95, # f4(theta_6p)
        2.0, -0.65, # f5(-cos(phi1))
        2.0, -0.65 # f5(-cos(phi2))
    ]
    params = init_fene_params + init_stacking_params
    displacement_fn, shift_fn = space.free()
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    T = DEFAULT_TEMP
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT
    
    gamma = RigidBody(center=jnp.array([kT/2.5], dtype=f64),
                      orientation=jnp.array([kT/7.5], dtype=f64))

    
    energy_fn, compute_subterms = factory.energy_fn_factory(displacement_fn,
                                                            back_site, stack_site, base_site,
                                                            top_info.bonded_nbrs, top_info.unbonded_nbrs)
    energy_fn = jit(Partial(energy_fn, seq=seq, params=params))
    compute_subterms = jit(Partial(compute_subterms, seq=seq, params=params))

    if use_ext_force:
        _, force_fn = ext_force.get_force_fn(energy_fn, top_info.n, displacement_fn,
                                             ext_force_bps1,
                                             [0, 0, ext_force_magnitude], [0, 0, 0, 0])
        _, force_fn = ext_force.get_force_fn(force_fn, top_info.n, displacement_fn,
                                             ext_force_bps2,
                                             [0, 0, -ext_force_magnitude], [0, 0, 0, 0])
    else:
        force_fn = quantity.force(energy_fn)
    force_fn = jit(force_fn)

    def force_adj(body):
        q = body.orientation.vec
        q_conj = _quaternion_conjugate(q)
        non_adj_force = force_fn(body)
        non_adj_force_q = non_adj_force.orientation
        orientation_force_adjustment = _quaternion_multiply(
            _quaternion_multiply(q_conj, non_adj_force_q.vec), q)
        adj_orientation_force = non_adj_force_q.vec - orientation_force_adjustment
        return RigidBody(center=non_adj_force.center, 
                         orientation=Quaternion(adj_orientation_force))
    batched_force_fn = vmap(force_adj)
    # batched_force_fn = vmap(force_fn)

    pdb.set_trace()

    equilibration_start = 200
    # eq_states = traj_info.states[equilibration_start:]
    eq_states = list()
    for t_info in tqdm(all_traj_info, desc="States"):
        eq_states += t_info.states[equilibration_start:]
    eq_rb_center = jnp.array([s.center for s in eq_states])
    eq_rb_quat_vec = jnp.array([s.orientation.vec for s in eq_states])
    eq_rb = RigidBody(eq_rb_center, Quaternion(eq_rb_quat_vec))

    print("Computing forces")
    # Now, we do the thing
    batch_forces = batched_force_fn(eq_rb)
    mean_center_force = jnp.mean(batch_forces.center, axis=0)
    mean_orientation_force = jnp.mean(batch_forces.orientation.vec, axis=0)


    pdb.set_trace()

    print("done")
        

