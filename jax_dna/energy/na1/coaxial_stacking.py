"""Coaxial stacking energy term for NA1 model."""

import chex
import jax
import jax.numpy as jnp
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.energy.dna2 as dna2_energy
import jax_dna.energy.na1.nucleotide as na1_nucleotide
import jax_dna.energy.na1.utils as je_utils
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class CoaxialStackingConfiguration(config.BaseConfiguration):
    """Configuration for the cross-stacking energy function."""

    # independent parameters
    nt_type: typ.Arr_Nucleotide | None = None
    ## DNA2-specific
    dna_dr_low_coax: float | None = None
    dna_dr_high_coax: float | None = None
    dna_k_coax: float | None = None
    dna_dr0_coax: float | None = None
    dna_dr_c_coax: float | None = None
    dna_theta0_coax_4: float | None = None
    dna_delta_theta_star_coax_4: float | None = None
    dna_a_coax_4: float | None = None
    dna_theta0_coax_1: float | None = None
    dna_delta_theta_star_coax_1: float | None = None
    dna_a_coax_1: float | None = None
    dna_theta0_coax_5: float | None = None
    dna_delta_theta_star_coax_5: float | None = None
    dna_a_coax_5: float | None = None
    dna_theta0_coax_6: float | None = None
    dna_delta_theta_star_coax_6: float | None = None
    dna_a_coax_6: float | None = None
    dna_a_coax_1_f6: float | None = None
    dna_b_coax_1_f6: float | None = None

    ## RNA2-specific
    rna_dr_low_coax: float | None = None
    rna_dr_high_coax: float | None = None
    rna_k_coax: float | None = None
    rna_dr0_coax: float | None = None
    rna_dr_c_coax: float | None = None
    rna_theta0_coax_4: float | None = None
    rna_delta_theta_star_coax_4: float | None = None
    rna_a_coax_4: float | None = None
    rna_theta0_coax_1: float | None = None
    rna_delta_theta_star_coax_1: float | None = None
    rna_a_coax_1: float | None = None
    rna_theta0_coax_5: float | None = None
    rna_delta_theta_star_coax_5: float | None = None
    rna_a_coax_5: float | None = None
    rna_theta0_coax_6: float | None = None
    rna_delta_theta_star_coax_6: float | None = None
    rna_a_coax_6: float | None = None
    rna_cos_phi3_star_coax: float | None = None
    rna_a_coax_3p: float | None = None
    rna_cos_phi4_star_coax: float | None = None
    rna_a_coax_4p: float | None = None

    ## DNA/RNA-hybrid-specific
    drh_dr_low_coax: float | None = None
    drh_dr_high_coax: float | None = None
    drh_k_coax: float | None = None
    drh_dr0_coax: float | None = None
    drh_dr_c_coax: float | None = None
    drh_theta0_coax_4: float | None = None
    drh_delta_theta_star_coax_4: float | None = None
    drh_a_coax_4: float | None = None
    drh_theta0_coax_1: float | None = None
    drh_delta_theta_star_coax_1: float | None = None
    drh_a_coax_1: float | None = None
    drh_theta0_coax_5: float | None = None
    drh_delta_theta_star_coax_5: float | None = None
    drh_a_coax_5: float | None = None
    drh_theta0_coax_6: float | None = None
    drh_delta_theta_star_coax_6: float | None = None
    drh_a_coax_6: float | None = None
    drh_cos_phi3_star_coax: float | None = None
    drh_a_coax_3p: float | None = None
    drh_cos_phi4_star_coax: float | None = None
    drh_a_coax_4p: float | None = None

    # dependent parameters
    dna_config: dna2_energy.CoaxialStackingConfiguration | None = None
    rna_config: dna1_energy.CoaxialStackingConfiguration | None = None
    drh_config: dna1_energy.CoaxialStackingConfiguration | None = None

    # override
    required_params: tuple[str] = (
        "nt_type",
        # DNA2-specific
        "dna_dr_low_coax",
        "dna_dr_high_coax",
        "dna_k_coax",
        "dna_dr0_coax",
        "dna_dr_c_coax",
        "dna_theta0_coax_4",
        "dna_delta_theta_star_coax_4",
        "dna_a_coax_4",
        "dna_theta0_coax_1",
        "dna_delta_theta_star_coax_1",
        "dna_a_coax_1",
        "dna_theta0_coax_5",
        "dna_delta_theta_star_coax_5",
        "dna_a_coax_5",
        "dna_theta0_coax_6",
        "dna_delta_theta_star_coax_6",
        "dna_a_coax_6",
        "dna_a_coax_1_f6",
        "dna_b_coax_1_f6",
        # RNA2-specific
        "rna_dr_low_coax",
        "rna_dr_high_coax",
        "rna_k_coax",
        "rna_dr0_coax",
        "rna_dr_c_coax",
        "rna_theta0_coax_4",
        "rna_delta_theta_star_coax_4",
        "rna_a_coax_4",
        "rna_theta0_coax_1",
        "rna_delta_theta_star_coax_1",
        "rna_a_coax_1",
        "rna_theta0_coax_5",
        "rna_delta_theta_star_coax_5",
        "rna_a_coax_5",
        "rna_theta0_coax_6",
        "rna_delta_theta_star_coax_6",
        "rna_a_coax_6",
        "rna_cos_phi3_star_coax",
        "rna_a_coax_3p",
        "rna_cos_phi4_star_coax",
        "rna_a_coax_4p",
        # DNA/RNA-hybrid-specific
        "drh_dr_low_coax",
        "drh_dr_high_coax",
        "drh_k_coax",
        "drh_dr0_coax",
        "drh_dr_c_coax",
        "drh_theta0_coax_4",
        "drh_delta_theta_star_coax_4",
        "drh_a_coax_4",
        "drh_theta0_coax_1",
        "drh_delta_theta_star_coax_1",
        "drh_a_coax_1",
        "drh_theta0_coax_5",
        "drh_delta_theta_star_coax_5",
        "drh_a_coax_5",
        "drh_theta0_coax_6",
        "drh_delta_theta_star_coax_6",
        "drh_a_coax_6",
        "drh_cos_phi3_star_coax",
        "drh_a_coax_3p",
        "drh_cos_phi4_star_coax",
        "drh_a_coax_4p",
    )

    @override
    def init_params(self) -> "CoaxialStackingConfiguration":
        dna_config = dna2_energy.CoaxialStackingConfiguration(
            dr_low_coax=self.dna_dr_low_coax,
            dr_high_coax=self.dna_dr_high_coax,
            k_coax=self.dna_k_coax,
            dr0_coax=self.dna_dr0_coax,
            dr_c_coax=self.dna_dr_c_coax,
            theta0_coax_4=self.dna_theta0_coax_4,
            delta_theta_star_coax_4=self.dna_delta_theta_star_coax_4,
            a_coax_4=self.dna_a_coax_4,
            theta0_coax_1=self.dna_theta0_coax_1,
            delta_theta_star_coax_1=self.dna_delta_theta_star_coax_1,
            a_coax_1=self.dna_a_coax_1,
            theta0_coax_5=self.dna_theta0_coax_5,
            delta_theta_star_coax_5=self.dna_delta_theta_star_coax_5,
            a_coax_5=self.dna_a_coax_5,
            theta0_coax_6=self.dna_theta0_coax_6,
            delta_theta_star_coax_6=self.dna_delta_theta_star_coax_6,
            a_coax_6=self.dna_a_coax_6,
            a_coax_1_f6=self.dna_a_coax_1_f6,
            b_coax_1_f6=self.dna_b_coax_1_f6,
        ).init_params()

        rna_config = dna1_energy.CoaxialStackingConfiguration(
            dr_low_coax=self.rna_dr_low_coax,
            dr_high_coax=self.rna_dr_high_coax,
            k_coax=self.rna_k_coax,
            dr0_coax=self.rna_dr0_coax,
            dr_c_coax=self.rna_dr_c_coax,
            theta0_coax_4=self.rna_theta0_coax_4,
            delta_theta_star_coax_4=self.rna_delta_theta_star_coax_4,
            a_coax_4=self.rna_a_coax_4,
            theta0_coax_1=self.rna_theta0_coax_1,
            delta_theta_star_coax_1=self.rna_delta_theta_star_coax_1,
            a_coax_1=self.rna_a_coax_1,
            theta0_coax_5=self.rna_theta0_coax_5,
            delta_theta_star_coax_5=self.rna_delta_theta_star_coax_5,
            a_coax_5=self.rna_a_coax_5,
            theta0_coax_6=self.rna_theta0_coax_6,
            delta_theta_star_coax_6=self.rna_delta_theta_star_coax_6,
            a_coax_6=self.rna_a_coax_6,
            cos_phi3_star_coax=self.rna_cos_phi3_star_coax,
            a_coax_3p=self.rna_a_coax_3p,
            cos_phi4_star_coax=self.rna_cos_phi4_star_coax,
            a_coax_4p=self.rna_a_coax_4p,
        ).init_params()

        drh_config = dna1_energy.CoaxialStackingConfiguration(
            dr_low_coax=self.drh_dr_low_coax,
            dr_high_coax=self.drh_dr_high_coax,
            k_coax=self.drh_k_coax,
            dr0_coax=self.drh_dr0_coax,
            dr_c_coax=self.drh_dr_c_coax,
            theta0_coax_4=self.drh_theta0_coax_4,
            delta_theta_star_coax_4=self.drh_delta_theta_star_coax_4,
            a_coax_4=self.drh_a_coax_4,
            theta0_coax_1=self.drh_theta0_coax_1,
            delta_theta_star_coax_1=self.drh_delta_theta_star_coax_1,
            a_coax_1=self.drh_a_coax_1,
            theta0_coax_5=self.drh_theta0_coax_5,
            delta_theta_star_coax_5=self.drh_delta_theta_star_coax_5,
            a_coax_5=self.drh_a_coax_5,
            theta0_coax_6=self.drh_theta0_coax_6,
            delta_theta_star_coax_6=self.drh_delta_theta_star_coax_6,
            a_coax_6=self.drh_a_coax_6,
            cos_phi3_star_coax=self.drh_cos_phi3_star_coax,
            a_coax_3p=self.drh_a_coax_3p,
            cos_phi4_star_coax=self.drh_cos_phi4_star_coax,
            a_coax_4p=self.drh_a_coax_4p,
        ).init_params()

        return self.replace(
            dna_config=dna_config,
            rna_config=rna_config,
            drh_config=drh_config,
        )


@chex.dataclass(frozen=True)
class CoaxialStacking(je_base.BaseEnergyFunction):
    """Coaxial stacking energy function for NA1 model."""

    params: CoaxialStackingConfiguration

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

        dna_dgs = dna2_energy.CoaxialStacking(
            displacement_fn=self.displacement_fn, params=self.params.dna_config
        ).pairwise_energies(
            body.dna,
            body.dna,
            unbonded_neighbors,
        )

        rna_dgs = dna1_energy.CoaxialStacking(
            displacement_fn=self.displacement_fn, params=self.params.rna_config
        ).pairwise_energies(
            body.rna,
            body.rna,
            unbonded_neighbors,
        )

        drh_dgs = dna1_energy.CoaxialStacking(
            displacement_fn=self.displacement_fn, params=self.params.drh_config
        ).pairwise_energies(
            body.dna,
            body.rna,
            unbonded_neighbors,
        )

        rdh_dgs = dna1_energy.CoaxialStacking(
            displacement_fn=self.displacement_fn, params=self.params.drh_config
        ).pairwise_energies(
            body.rna,
            body.dna,
            unbonded_neighbors,
        )

        dgs = jnp.where(is_rna_bond, rna_dgs, jnp.where(is_drh_bond, drh_dgs, jnp.where(is_rdh_bond, rdh_dgs, dna_dgs)))
        dgs = jnp.where(mask, dgs, 0.0)

        return dgs.sum()
