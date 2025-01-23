"""Bonded excluded volume energy for DNA1 model."""

import chex
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1.base_smoothing_functions as bsf
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class BondedExcludedVolumeConfiguration(config.BaseConfiguration):
    """Configuration for the bonded excluded volume energy function."""

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

    # override
    dependent_params: tuple[str] = (
        "b_base",
        "dr_c_base",
        "b_back_base",
        "dr_c_back_base",
        "b_base_back",
        "dr_c_base_back",
    )

    @override
    def init_params(self) -> "BondedExcludedVolumeConfiguration":
        b_base, dr_c_base = bsf.get_f3_smoothing_params(self.dr_star_base, self.sigma_base)

        # reference to f3(dr_back_base)
        b_back_base, dr_c_back_base = bsf.get_f3_smoothing_params(self.dr_star_back_base, self.sigma_back_base)

        # reference to f3(dr_base_back)
        b_base_back, dr_c_base_back = bsf.get_f3_smoothing_params(
            self.dr_star_base_back,
            self.sigma_base_back,
        )

        return self.replace(
            b_base=b_base,
            dr_c_base=dr_c_base,
            b_back_base=b_back_base,
            dr_c_back_base=dr_c_back_base,
            b_base_back=b_base_back,
            dr_c_base_back=dr_c_base_back,
        )


@chex.dataclass(frozen=True)
class BondedExcludedVolume(je_base.BaseEnergyFunction):
    """Bonded excluded volume energy function for DNA1 model."""

    params: BondedExcludedVolumeConfiguration

    def pairwise_energies(
        self,
        body: je_base.BaseNucleotide,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
    ) -> typ.Arr_Bonded_Neighbors:
        """Computes the excluded volume energy for each bonded pair."""
        nn_i = bonded_neighbors[:, 0]
        nn_j = bonded_neighbors[:, 1]

        dr_base = self.displacement_mapped(body.base_sites[nn_i], body.base_sites[nn_j])
        dr_back_base = self.displacement_mapped(body.back_sites[nn_i], body.base_sites[nn_j])
        dr_base_back = self.displacement_mapped(body.base_sites[nn_i], body.back_sites[nn_j])

        return dna1_interactions.exc_vol_bonded(
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
        )


    @override
    def __call__(
        self,
        body: je_base.BaseNucleotide,
        seq: typ.Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Scalar:
        dgs = self.pairwise_energies(body, bonded_neighbors)
        return dgs.sum()
