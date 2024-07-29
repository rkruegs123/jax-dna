import dataclasses as dc

import chex
import jax.numpy as jnp

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1.base_smoothing_functions as bsf
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.energy.dna1.nucleotide as dna1_nucleotide
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class ExcludedVolumeConfiguration(config.BaseConfiguration):
    # independent parameters
    eps_exc: float | None = None
    dr_star_base: float | None = None
    sigma_base: float | None = None
    sigma_back_base: float | None = None
    sigma_base_back: float | None = None
    dr_star_back_base: float | None = None
    dr_star_base_back: float | None = None

    # dependent parameters
    b_base: float | None = None
    dr_c_base: float | None = None
    b_back_base: float | None = None
    dr_c_back_base: float | None = None
    b_base_back: float | None = None
    dr_c_base_back: float | None = None

    # override
    required_params: tuple[str] = (
        "eps_exc",
        "dr_star_base",
        "sigma_base",
        "sigma_back_base",
        "sigma_base_back",
        "dr_star_back_base",
        "dr_star_base_back",
    )

    def init_params(self) -> "ExcludedVolumeConfiguration":
        b_base, dr_c_base = bsf.get_f3_smoothing_params(self.dr_star_base, self.eps_exc, self.sigma_base)

        ## f3(dr_back_base)
        b_back_base, dr_c_back_base = bsf.get_f3_smoothing_params(
            self.dr_star_back_base, self.eps_exc, self.sigma_back_base
        )

        ## f3(dr_base_back)
        b_base_back, dr_c_base_back = bsf.get_f3_smoothing_params(
            self.dr_star_base_back,
            self.eps_exc,
            self.sigma_base_back,
        )

        return dc.replace(
            self,
            b_base=b_base,
            dr_c_base=dr_c_base,
            b_back_base=b_back_base,
            dr_c_back_base=dr_c_back_base,
            b_base_back=b_base_back,
            dr_c_base_back=dr_c_base_back,
        )


@chex.dataclass(frozen=True)
class ExcludedVolume(je_base.BaseEnergyFunction):
    params: config.ExcludedVolumeConfiguration

    def __call__(
        self,
        body: dna1_nucleotide.Nucleotide,
        seq: jnp.ndarray,
        bonded_neighbors: typ.Arr_Bonded_Neighbors,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors,
    ) -> typ.Scalar:
        nn_i = bonded_neighbors[:, 0]
        nn_j = bonded_neighbors[:, 1]

        dr_base = self.displacement_mapped(body.base_sites[nn_i], body.base_sites[nn_j])
        dr_back_base = self.displacement_mapped(body.back_sites[nn_i], body.base_sites[nn_j])
        dr_base_back = self.displacement_mapped(body.base_sites[nn_i], body.back_sites[nn_j])

        exc_vol_bonded_dg = dna1_interactions.exc_vol_bonded(
            dr_base,
            dr_back_base,
            dr_base_back,
            self.params.eps_exc,
            self.params.dr_star_base,
            self.params.sigma_base,
            self.params.b_base,
            self.params.dr_c_base,
            self.params.dr_star_back_base,
            self.params.sigma_back_base,
            self.params.b_back_base,
            self.params.dr_c_back_base,
            self.params.dr_star_base_back,
            self.params.sigma_base_back,
            self.params.b_base_back,
            self.params.dr_c_base_back,
        ).sum()

        return exc_vol_bonded_dg
