"""Cross-stacking energy term for NA1 model."""


import chex
import jax
import jax.numpy as jnp
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.energy.na1.nucleotide as na1_nucleotide
import jax_dna.energy.na1.utils as je_utils
import jax_dna.energy.rna2 as rna2_energy
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class CrossStackingConfiguration(config.BaseConfiguration):
    """Configuration for the cross-stacking energy function."""

    # independent parameters
    nt_type: typ.Arr_Nucleotide | None = None
    ## DNA2-specific
    dna_dr_low_cross: float | None = None
    dna_dr_high_cross: float | None = None
    dna_k_cross: float | None = None
    dna_r0_cross: float | None = None
    dna_dr_c_cross: float | None = None
    dna_theta0_cross_1: float | None = None
    dna_delta_theta_star_cross_1: float | None = None
    dna_a_cross_1: float | None = None
    dna_theta0_cross_2: float | None = None
    dna_delta_theta_star_cross_2: float | None = None
    dna_a_cross_2: float | None = None
    dna_theta0_cross_3: float | None = None
    dna_delta_theta_star_cross_3: float | None = None
    dna_a_cross_3: float | None = None
    dna_theta0_cross_4: float | None = None
    dna_delta_theta_star_cross_4: float | None = None
    dna_a_cross_4: float | None = None
    dna_theta0_cross_7: float | None = None
    dna_delta_theta_star_cross_7: float | None = None
    dna_a_cross_7: float | None = None
    dna_theta0_cross_8: float | None = None
    dna_delta_theta_star_cross_8: float | None = None
    dna_a_cross_8: float | None = None
    ## RNA2-specific
    rna_dr_low_cross: float | None = None
    rna_dr_high_cross: float | None = None
    rna_k_cross: float | None = None
    rna_r0_cross: float | None = None
    rna_dr_c_cross: float | None = None
    rna_theta0_cross_1: float | None = None
    rna_delta_theta_star_cross_1: float | None = None
    rna_a_cross_1: float | None = None
    rna_theta0_cross_2: float | None = None
    rna_delta_theta_star_cross_2: float | None = None
    rna_a_cross_2: float | None = None
    rna_theta0_cross_3: float | None = None
    rna_delta_theta_star_cross_3: float | None = None
    rna_a_cross_3: float | None = None
    rna_theta0_cross_7: float | None = None
    rna_delta_theta_star_cross_7: float | None = None
    rna_a_cross_7: float | None = None
    rna_theta0_cross_8: float | None = None
    rna_delta_theta_star_cross_8: float | None = None
    rna_a_cross_8: float | None = None
    ## DNA/RNA-hybrid-specific
    drh_dr_low_cross: float | None = None
    drh_dr_high_cross: float | None = None
    drh_k_cross: float | None = None
    drh_r0_cross: float | None = None
    drh_dr_c_cross: float | None = None
    drh_theta0_cross_1: float | None = None
    drh_delta_theta_star_cross_1: float | None = None
    drh_a_cross_1: float | None = None
    drh_theta0_cross_2: float | None = None
    drh_delta_theta_star_cross_2: float | None = None
    drh_a_cross_2: float | None = None
    drh_theta0_cross_3: float | None = None
    drh_delta_theta_star_cross_3: float | None = None
    drh_a_cross_3: float | None = None
    drh_theta0_cross_4: float | None = None
    drh_delta_theta_star_cross_4: float | None = None
    drh_a_cross_4: float | None = None
    drh_theta0_cross_7: float | None = None
    drh_delta_theta_star_cross_7: float | None = None
    drh_a_cross_7: float | None = None
    drh_theta0_cross_8: float | None = None
    drh_delta_theta_star_cross_8: float | None = None
    drh_a_cross_8: float | None = None

    # dependent parameters
    dna_config: dna1_energy.CrossStackingConfiguration | None = None
    rna_config: rna2_energy.CrossStackingConfiguration | None = None
    drh_config: dna1_energy.CrossStackingConfiguration | None = None

    # override
    required_params: tuple[str] = (
        "nt_type",
        # DNA2-specific
        "dna_dr_low_cross",
        "dna_dr_high_cross",
        "dna_k_cross",
        "dna_r0_cross",
        "dna_dr_c_cross",
        "dna_theta0_cross_1",
        "dna_delta_theta_star_cross_1",
        "dna_a_cross_1",
        "dna_theta0_cross_2",
        "dna_delta_theta_star_cross_2",
        "dna_a_cross_2",
        "dna_theta0_cross_3",
        "dna_delta_theta_star_cross_3",
        "dna_a_cross_3",
        "dna_theta0_cross_4",
        "dna_delta_theta_star_cross_4",
        "dna_a_cross_4",
        "dna_theta0_cross_7",
        "dna_delta_theta_star_cross_7",
        "dna_a_cross_7",
        "dna_theta0_cross_8",
        "dna_delta_theta_star_cross_8",
        "dna_a_cross_8",
        # RNA2-specific
        "rna_dr_low_cross",
        "rna_dr_high_cross",
        "rna_k_cross",
        "rna_r0_cross",
        "rna_dr_c_cross",
        "rna_theta0_cross_1",
        "rna_delta_theta_star_cross_1",
        "rna_a_cross_1",
        "rna_theta0_cross_2",
        "rna_delta_theta_star_cross_2",
        "rna_a_cross_2",
        "rna_theta0_cross_3",
        "rna_delta_theta_star_cross_3",
        "rna_a_cross_3",
        "rna_theta0_cross_7",
        "rna_delta_theta_star_cross_7",
        "rna_a_cross_7",
        "rna_theta0_cross_8",
        "rna_delta_theta_star_cross_8",
        "rna_a_cross_8",
        # DNA/RNA-hybrid-specific
        "drh_dr_low_cross",
        "drh_dr_high_cross",
        "drh_k_cross",
        "drh_r0_cross",
        "drh_dr_c_cross",
        "drh_theta0_cross_1",
        "drh_delta_theta_star_cross_1",
        "drh_a_cross_1",
        "drh_theta0_cross_2",
        "drh_delta_theta_star_cross_2",
        "drh_a_cross_2",
        "drh_theta0_cross_3",
        "drh_delta_theta_star_cross_3",
        "drh_a_cross_3",
        "drh_theta0_cross_4",
        "drh_delta_theta_star_cross_4",
        "drh_a_cross_4",
        "drh_theta0_cross_7",
        "drh_delta_theta_star_cross_7",
        "drh_a_cross_7",
        "drh_theta0_cross_8",
        "drh_delta_theta_star_cross_8",
        "drh_a_cross_8",
    )

    @override
    def init_params(self) -> "CrossStackingConfiguration":

        dna_config = dna1_energy.CrossStackingConfiguration(
            dr_low_cross=self.dna_dr_low_cross,
            dr_high_cross=self.dna_dr_high_cross,
            k_cross=self.dna_k_cross,
            r0_cross=self.dna_r0_cross,
            dr_c_cross=self.dna_dr_c_cross,
            theta0_cross_1=self.dna_theta0_cross_1,
            delta_theta_star_cross_1=self.dna_delta_theta_star_cross_1,
            a_cross_1=self.dna_a_cross_1,
            theta0_cross_2=self.dna_theta0_cross_2,
            delta_theta_star_cross_2=self.dna_delta_theta_star_cross_2,
            a_cross_2=self.dna_a_cross_2,
            theta0_cross_3=self.dna_theta0_cross_3,
            delta_theta_star_cross_3=self.dna_delta_theta_star_cross_3,
            a_cross_3=self.dna_a_cross_3,
            theta0_cross_4=self.dna_theta0_cross_4,
            delta_theta_star_cross_4=self.dna_delta_theta_star_cross_4,
            a_cross_4=self.dna_a_cross_4,
            theta0_cross_7=self.dna_theta0_cross_7,
            delta_theta_star_cross_7=self.dna_delta_theta_star_cross_7,
            a_cross_7=self.dna_a_cross_7,
            theta0_cross_8=self.dna_theta0_cross_8,
            delta_theta_star_cross_8=self.dna_delta_theta_star_cross_8,
            a_cross_8=self.dna_a_cross_8,
        ).init_params()

        rna_config = rna2_energy.CrossStackingConfiguration(
            dr_low_cross=self.rna_dr_low_cross,
            dr_high_cross=self.rna_dr_high_cross,
            k_cross=self.rna_k_cross,
            r0_cross=self.rna_r0_cross,
            dr_c_cross=self.rna_dr_c_cross,
            theta0_cross_1=self.rna_theta0_cross_1,
            delta_theta_star_cross_1=self.rna_delta_theta_star_cross_1,
            a_cross_1=self.rna_a_cross_1,
            theta0_cross_2=self.rna_theta0_cross_2,
            delta_theta_star_cross_2=self.rna_delta_theta_star_cross_2,
            a_cross_2=self.rna_a_cross_2,
            theta0_cross_3=self.rna_theta0_cross_3,
            delta_theta_star_cross_3=self.rna_delta_theta_star_cross_3,
            a_cross_3=self.rna_a_cross_3,
            theta0_cross_7=self.rna_theta0_cross_7,
            delta_theta_star_cross_7=self.rna_delta_theta_star_cross_7,
            a_cross_7=self.rna_a_cross_7,
            theta0_cross_8=self.rna_theta0_cross_8,
            delta_theta_star_cross_8=self.rna_delta_theta_star_cross_8,
            a_cross_8=self.rna_a_cross_8,
        ).init_params()

        drh_config = dna1_energy.CrossStackingConfiguration(
            dr_low_cross=self.drh_dr_low_cross,
            dr_high_cross=self.drh_dr_high_cross,
            k_cross=self.drh_k_cross,
            r0_cross=self.drh_r0_cross,
            dr_c_cross=self.drh_dr_c_cross,
            theta0_cross_1=self.drh_theta0_cross_1,
            delta_theta_star_cross_1=self.drh_delta_theta_star_cross_1,
            a_cross_1=self.drh_a_cross_1,
            theta0_cross_2=self.drh_theta0_cross_2,
            delta_theta_star_cross_2=self.drh_delta_theta_star_cross_2,
            a_cross_2=self.drh_a_cross_2,
            theta0_cross_3=self.drh_theta0_cross_3,
            delta_theta_star_cross_3=self.drh_delta_theta_star_cross_3,
            a_cross_3=self.drh_a_cross_3,
            theta0_cross_4=self.drh_theta0_cross_4,
            delta_theta_star_cross_4=self.drh_delta_theta_star_cross_4,
            a_cross_4=self.drh_a_cross_4,
            theta0_cross_7=self.drh_theta0_cross_7,
            delta_theta_star_cross_7=self.drh_delta_theta_star_cross_7,
            a_cross_7=self.drh_a_cross_7,
            theta0_cross_8=self.drh_theta0_cross_8,
            delta_theta_star_cross_8=self.drh_delta_theta_star_cross_8,
            a_cross_8=self.drh_a_cross_8,
        ).init_params()

        return self.replace(
            dna_config=dna_config,
            rna_config=rna_config,
            drh_config=drh_config,
        )


@chex.dataclass(frozen=True)
class CrossStacking(je_base.BaseEnergyFunction):
    """Cross-stacking energy function for NA1 model."""

    params: CrossStackingConfiguration

    @override
    def __call__(
        self,
        body: na1_nucleotide.HybridNucleotide,
        seq: typ.Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Scalar:
        op_i = unbonded_neighbors[0]
        op_j = unbonded_neighbors[1]

        is_rna_bond = jax.vmap(je_utils.is_rna_pair, (0, 0, None))(op_i, op_j, self.params.nt_type)
        is_drh_bond = jax.vmap(je_utils.is_dna_rna_pair, (0, 0, None))(op_i, op_j, self.params.nt_type)
        is_rdh_bond = jax.vmap(je_utils.is_dna_rna_pair, (0, 0, None))(op_j, op_i, self.params.nt_type)

        mask = jnp.array(op_i < body.dna.center.shape[0], dtype=jnp.float32)


        dna_dgs = dna1_energy.CrossStacking(
            displacement_fn=self.displacement_fn,
            params=self.params.dna_config
        ).pairwise_energies(
            body.dna,
            body.dna,
            unbonded_neighbors,
        )

        rna_dgs = rna2_energy.CrossStacking(
            displacement_fn=self.displacement_fn,
            params=self.params.rna_config
        ).pairwise_energies(
            body.rna,
            body.rna,
            unbonded_neighbors,
        )

        drh_dgs = dna1_energy.CrossStacking(
            displacement_fn=self.displacement_fn,
            params=self.params.drh_config
        ).pairwise_energies(
            body.dna,
            body.rna,
            unbonded_neighbors,
        )

        rdh_dgs = dna1_energy.CrossStacking(
            displacement_fn=self.displacement_fn,
            params=self.params.drh_config
        ).pairwise_energies(
            body.rna,
            body.dna,
            unbonded_neighbors,
        )


        dgs = jnp.where(is_rna_bond, rna_dgs,
                        jnp.where(is_drh_bond, drh_dgs,
                                  jnp.where(is_rdh_bond, rdh_dgs, dna_dgs)))
        dgs = jnp.where(mask, dgs, 0.0)

        return dgs.sum()
