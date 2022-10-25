import pdb
import jax.numpy as jnp
from jax.tree_util import Partial

from jax_md import space
from jax_md import util

from energy import factory
from utils import base_site, stack_site, back_site, DEFAULT_TEMP, get_one_hot
from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo
from loader import get_params

import oxpy
from oxDNA_analysis_tools.UTILS.RyeReader import describe
from oxDNA_analysis_tools.output_bonds import output_bonds


f64 = util.f64

# `n` is the index into states to compute the subterms
def compute_subterms(top_path, traj_path, T=DEFAULT_TEMP, n=-1):
    top_info = TopologyInfo(top_path, reverse_direction=True)
    traj_info = TrajectoryInfo(top_info, traj_path=traj_path, reverse_direction=True)
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)

    displacement_fn, shift_fn = space.periodic(traj_info.box_size)
    params = get_params.get_default_params(t=T, no_smoothing=False)

    _, _compute_subterms = factory.energy_fn_factory(displacement_fn,
                                                     back_site, stack_site, base_site,
                                                     top_info.bonded_nbrs, top_info.unbonded_nbrs)
    _compute_subterms = Partial(_compute_subterms, seq=seq, params=params)
    return _compute_subterms(traj_info.states[-1])

def get_oxpy_subterms(top_path, traj_path, input_path):
    top_info, traj_info = describe(top_path, traj_path)
    output_bonds(traj_info, top_info, input_path, visualize=False)

def run(top_path, traj_path, input_path, T=DEFAULT_TEMP):
    # computed_subterms = compute_subterms(top_path, traj_path, T)
    # return computed_subterms
    get_oxpy_subterms(top_path, traj_path, input_path)
