"""Unbonded excluded volume energy function for DNA1 model."""

import dataclasses as dc

import chex
import jax.numpy as jnp
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1.base_smoothing_functions as bsf
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.energy.dna1.nucleotide as dna1_nucleotide
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class UnbondedExcludedVolumeConfiguration(config.BaseConfiguration):
    """Configuration for the unbonded excluded volume energy function."""

    # independent parameters
    eps_exc: float | None = None
    dr_star_base: float | None = None
    sigma_base: float | None = None
    dr_star_back_base: float | None = None
    sigma_back_base: float | None = None
    dr_star_base_back: float | None = None
    sigma_base_back: float | None = None
    dr_star_backbone: float | None = None
    sigma_backbone: float | None = None

    # dependent parameters
    b_base: float | None = None
    dr_c_base: float | None = None
    b_back_base: float | None = None
    dr_c_back_base: float | None = None
    b_base_back: float | None = None
    dr_c_base_back: float | None = None
    b_backbone: float | None = None
    dr_c_backbone: float | None = None

    # override
    required_params: tuple[str] = (
        "eps_exc",
        "dr_star_base",
        "sigma_base",
        "dr_star_back_base",
        "sigma_back_base",
        "dr_star_base_back",
        "sigma_base_back",
        "dr_star_backbone",
        "sigma_backbone",
    )

    # override
    dependent_params: tuple[str] = (
        "b_base",
        "dr_c_base",
        "b_back_base",
        "dr_c_back_base",
        "b_base_back",
        "dr_c_base_back",
        "b_backbone",
        "dr_c_backbone",
    )

    @override
    def init_params(self) -> "UnbondedExcludedVolumeConfiguration":
        # reference to f3(dr_base)
        b_base, dr_c_base = bsf.get_f3_smoothing_params(self.dr_star_base, self.sigma_base)

        # reference to f3(dr_back_base)
        b_back_base, dr_c_back_base = bsf.get_f3_smoothing_params(self.dr_star_back_base, self.sigma_back_base)

        # reference to f3(dr_base_back)
        b_base_back, dr_c_base_back = bsf.get_f3_smoothing_params(
            self.dr_star_base_back,
            self.sigma_base_back,
        )

        # reference to f3(dr_backbone)
        b_backbone, dr_c_backbone = bsf.get_f3_smoothing_params(
            self.dr_star_backbone,
            self.sigma_backbone,
        )

        return dc.replace(
            self,
            b_base=b_base,
            dr_c_base=dr_c_base,
            b_back_base=b_back_base,
            dr_c_back_base=dr_c_back_base,
            b_base_back=b_base_back,
            dr_c_base_back=dr_c_base_back,
            b_backbone=b_backbone,
            dr_c_backbone=dr_c_backbone,
        )


@chex.dataclass(frozen=True)
class UnbondedExcludedVolume(je_base.BaseEnergyFunction):
    """Unbonded excluded volume energy function for DNA1 model."""

    params: UnbondedExcludedVolumeConfiguration

    @override
    def __call__(
        self,
        body: dna1_nucleotide.Nucleotide,
        seq: typ.Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Scalar:
        op_i = unbonded_neighbors[0]
        op_j = unbonded_neighbors[1]

        mask = jnp.array(op_i < body.center.shape[0], dtype=jnp.float32)

        dr_base_op = self.displacement_mapped(body.base_sites[op_j], body.base_sites[op_i])  # Note the flip here
        dr_backbone_op = self.displacement_mapped(body.back_sites[op_j], body.back_sites[op_i])  # Note the flip here
        dr_back_base_op = self.displacement_mapped(
            body.back_sites[op_i], body.base_sites[op_j]
        )  # Note: didn't flip this one (and others) because no need, but should look into at some point
        dr_base_back_op = self.displacement_mapped(body.base_sites[op_i], body.back_sites[op_j])

        exc_vol_unbonded_dg = dna1_interactions.exc_vol_unbonded(
            dr_base_op,
            dr_backbone_op,
            dr_back_base_op,
            dr_base_back_op,
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
            self.params.dr_star_backbone,
            self.params.sigma_backbone,
            self.params.b_backbone,
            self.params.dr_c_backbone,
        )

        return jnp.where(mask, exc_vol_unbonded_dg, 0.0).sum()
