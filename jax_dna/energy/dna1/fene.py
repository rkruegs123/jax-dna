import dataclasses as dc

import chex
import jax.numpy as jnp
import numpy as np

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1.base_smoothing_functions as bsf
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.energy.dna1.nucleotide as dna1_nucleotide
import jax_dna.energy.utils as je_utils
import jax_dna.input.dna1.bonded as config
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class FeneConfiguration(config.BaseConfiguration):
    # independent parameters
    eps_backbone: float | None = None
    r0_backbone: float | None = None
    delta_backbone: float | None = None
    fmax: float | None = None
    finf: float | None = None

    # override
    required_params: tuple[str] = ("eps_backbone", "r0_backbone", "delta_backbone", "fmax", "finf")

    @staticmethod
    def from_toml(file_path: str, params_to_optimize: tuple[str] = ()) -> "FeneConfiguration":
        dict_params = FeneConfiguration.parse_toml(file_path, "vfene")
        return FeneConfiguration.from_dict(dict_params, params_to_optimize)


@chex.dataclass(frozen=True)
class Fene(je_base.BaseEnergyFunction):
    params: config.FeneConfiguration

    def __call__(
        self,
        body: dna1_nucleotide.Nucleotide,
        seq: jnp.ndarray,
        bonded_neghbors: typ.Arr_Bonded_Neighbors,
        unbounded_neghbors: typ.Arr_Unbonded_Neighbors,
    ) -> typ.Scalar:
        nn_i = bonded_neghbors[:, 0]
        nn_j = bonded_neghbors[:, 1]

        dr_back_nn = self.displacement_mapped(body.back_sites[nn_i], body.back_sites[nn_j])
        r_back_nn = jnp.linalg.norm(dr_back_nn, axis=1)

        return dna1_interactions.v_fene_smooth(
            r_back_nn,
            self.params.eps_backbone,
            self.params.r0_backbone,
            self.params.delta_backbone,
            self.params.fmax,
            self.params.finf,
        ).sum()
