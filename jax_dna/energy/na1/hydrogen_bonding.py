"""Hydrogen bonding energy function for NA1 model."""

import dataclasses as dc

import chex
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.energy.na1.nucleotide as na1_nucleotide
import jax_dna.energy.na1.utils as je_utils
import jax_dna.utils.types as typ
from jax_dna.energy.dna1.hydrogen_bonding import HB_WEIGHTS_SA


@chex.dataclass(frozen=True)
class HydrogenBondingConfiguration(config.BaseConfiguration):
    """Configuration for the cross-stacking energy function."""

    # independent parameters
    nt_type: typ.Arr_Nucleotide | None = None
    ## DNA2-specific
    dna_eps_hb: float | None = None
    dna_a_hb: float | None = None
    dna_dr0_hb: float | None = None
    dna_dr_c_hb: float | None = None
    dna_dr_low_hb: float | None = None
    dna_dr_high_hb: float | None = None

    dna_a_hb_1: float | None = None
    dna_theta0_hb_1: float | None = None
    dna_delta_theta_star_hb_1: float | None = None

    dna_a_hb_2: float | None = None
    dna_theta0_hb_2: float | None = None
    dna_delta_theta_star_hb_2: float | None = None

    dna_a_hb_3: float | None = None
    dna_theta0_hb_3: float | None = None
    dna_delta_theta_star_hb_3: float | None = None

    dna_a_hb_4: float | None = None
    dna_theta0_hb_4: float | None = None
    dna_delta_theta_star_hb_4: float | None = None

    dna_a_hb_7: float | None = None
    dna_theta0_hb_7: float | None = None
    dna_delta_theta_star_hb_7: float | None = None

    dna_a_hb_8: float | None = None
    dna_theta0_hb_8: float | None = None
    dna_delta_theta_star_hb_8: float | None = None

    dna_ss_hb_weights: np.ndarray | None = dc.field(default_factory=lambda: HB_WEIGHTS_SA)

    ## RNA2-specific
    rna_eps_hb: float | None = None
    rna_a_hb: float | None = None
    rna_dr0_hb: float | None = None
    rna_dr_c_hb: float | None = None
    rna_dr_low_hb: float | None = None
    rna_dr_high_hb: float | None = None

    rna_a_hb_1: float | None = None
    rna_theta0_hb_1: float | None = None
    rna_delta_theta_star_hb_1: float | None = None

    rna_a_hb_2: float | None = None
    rna_theta0_hb_2: float | None = None
    rna_delta_theta_star_hb_2: float | None = None

    rna_a_hb_3: float | None = None
    rna_theta0_hb_3: float | None = None
    rna_delta_theta_star_hb_3: float | None = None

    rna_a_hb_4: float | None = None
    rna_theta0_hb_4: float | None = None
    rna_delta_theta_star_hb_4: float | None = None

    rna_a_hb_7: float | None = None
    rna_theta0_hb_7: float | None = None
    rna_delta_theta_star_hb_7: float | None = None

    rna_a_hb_8: float | None = None
    rna_theta0_hb_8: float | None = None
    rna_delta_theta_star_hb_8: float | None = None

    rna_ss_hb_weights: np.ndarray | None = dc.field(default_factory=lambda: HB_WEIGHTS_SA)

    ## DNA/RNA-hybrid-specific
    drh_eps_hb: float | None = None
    drh_a_hb: float | None = None
    drh_dr0_hb: float | None = None
    drh_dr_c_hb: float | None = None
    drh_dr_low_hb: float | None = None
    drh_dr_high_hb: float | None = None

    drh_a_hb_1: float | None = None
    drh_theta0_hb_1: float | None = None
    drh_delta_theta_star_hb_1: float | None = None

    drh_a_hb_2: float | None = None
    drh_theta0_hb_2: float | None = None
    drh_delta_theta_star_hb_2: float | None = None

    drh_a_hb_3: float | None = None
    drh_theta0_hb_3: float | None = None
    drh_delta_theta_star_hb_3: float | None = None

    drh_a_hb_4: float | None = None
    drh_theta0_hb_4: float | None = None
    drh_delta_theta_star_hb_4: float | None = None

    drh_a_hb_7: float | None = None
    drh_theta0_hb_7: float | None = None
    drh_delta_theta_star_hb_7: float | None = None

    drh_a_hb_8: float | None = None
    drh_theta0_hb_8: float | None = None
    drh_delta_theta_star_hb_8: float | None = None

    drh_ss_hb_weights: np.ndarray | None = dc.field(default_factory=lambda: HB_WEIGHTS_SA)

    # dependent parameters
    dna_config: dna1_energy.HydrogenBondingConfiguration | None = None
    rna_config: dna1_energy.HydrogenBondingConfiguration | None = None
    drh_config: dna1_energy.HydrogenBondingConfiguration | None = None

    # override
    required_params: tuple[str] = (
        "nt_type",
        # DNA2-specific
        "dna_eps_hb",
        "dna_a_hb",
        "dna_dr0_hb",
        "dna_dr_c_hb",
        "dna_dr_low_hb",
        "dna_dr_high_hb",
        "dna_a_hb_1",
        "dna_theta0_hb_1",
        "dna_delta_theta_star_hb_1",
        "dna_a_hb_2",
        "dna_theta0_hb_2",
        "dna_delta_theta_star_hb_2",
        "dna_a_hb_3",
        "dna_theta0_hb_3",
        "dna_delta_theta_star_hb_3",
        "dna_a_hb_4",
        "dna_theta0_hb_4",
        "dna_delta_theta_star_hb_4",
        "dna_a_hb_7",
        "dna_theta0_hb_7",
        "dna_delta_theta_star_hb_7",
        "dna_a_hb_8",
        "dna_theta0_hb_8",
        "dna_delta_theta_star_hb_8",
        "dna_ss_hb_weights",
        # RNA2-specific
        "rna_eps_hb",
        "rna_a_hb",
        "rna_dr0_hb",
        "rna_dr_c_hb",
        "rna_dr_low_hb",
        "rna_dr_high_hb",
        "rna_a_hb_1",
        "rna_theta0_hb_1",
        "rna_delta_theta_star_hb_1",
        "rna_a_hb_2",
        "rna_theta0_hb_2",
        "rna_delta_theta_star_hb_2",
        "rna_a_hb_3",
        "rna_theta0_hb_3",
        "rna_delta_theta_star_hb_3",
        "rna_a_hb_4",
        "rna_theta0_hb_4",
        "rna_delta_theta_star_hb_4",
        "rna_a_hb_7",
        "rna_theta0_hb_7",
        "rna_delta_theta_star_hb_7",
        "rna_a_hb_8",
        "rna_theta0_hb_8",
        "rna_delta_theta_star_hb_8",
        "rna_ss_hb_weights",
        # DNA/RNA-hybrid-specific
        "drh_eps_hb",
        "drh_a_hb",
        "drh_dr0_hb",
        "drh_dr_c_hb",
        "drh_dr_low_hb",
        "drh_dr_high_hb",
        "drh_a_hb_1",
        "drh_theta0_hb_1",
        "drh_delta_theta_star_hb_1",
        "drh_a_hb_2",
        "drh_theta0_hb_2",
        "drh_delta_theta_star_hb_2",
        "drh_a_hb_3",
        "drh_theta0_hb_3",
        "drh_delta_theta_star_hb_3",
        "drh_a_hb_4",
        "drh_theta0_hb_4",
        "drh_delta_theta_star_hb_4",
        "drh_a_hb_7",
        "drh_theta0_hb_7",
        "drh_delta_theta_star_hb_7",
        "drh_a_hb_8",
        "drh_theta0_hb_8",
        "drh_delta_theta_star_hb_8",
        "drh_ss_hb_weights",
    )

    @override
    def init_params(self) -> "HydrogenBondingConfiguration":
        dna_config = dna1_energy.HydrogenBondingConfiguration(
            eps_hb=self.dna_eps_hb,
            a_hb=self.dna_a_hb,
            dr0_hb=self.dna_dr0_hb,
            dr_c_hb=self.dna_dr_c_hb,
            dr_low_hb=self.dna_dr_low_hb,
            dr_high_hb=self.dna_dr_high_hb,
            a_hb_1=self.dna_a_hb_1,
            theta0_hb_1=self.dna_theta0_hb_1,
            delta_theta_star_hb_1=self.dna_delta_theta_star_hb_1,
            a_hb_2=self.dna_a_hb_2,
            theta0_hb_2=self.dna_theta0_hb_2,
            delta_theta_star_hb_2=self.dna_delta_theta_star_hb_2,
            a_hb_3=self.dna_a_hb_3,
            theta0_hb_3=self.dna_theta0_hb_3,
            delta_theta_star_hb_3=self.dna_delta_theta_star_hb_3,
            a_hb_4=self.dna_a_hb_4,
            theta0_hb_4=self.dna_theta0_hb_4,
            delta_theta_star_hb_4=self.dna_delta_theta_star_hb_4,
            a_hb_7=self.dna_a_hb_7,
            theta0_hb_7=self.dna_theta0_hb_7,
            delta_theta_star_hb_7=self.dna_delta_theta_star_hb_7,
            a_hb_8=self.dna_a_hb_8,
            theta0_hb_8=self.dna_theta0_hb_8,
            delta_theta_star_hb_8=self.dna_delta_theta_star_hb_8,
            ss_hb_weights=self.dna_ss_hb_weights,
        ).init_params()

        rna_config = dna1_energy.HydrogenBondingConfiguration(
            eps_hb=self.rna_eps_hb,
            a_hb=self.rna_a_hb,
            dr0_hb=self.rna_dr0_hb,
            dr_c_hb=self.rna_dr_c_hb,
            dr_low_hb=self.rna_dr_low_hb,
            dr_high_hb=self.rna_dr_high_hb,
            a_hb_1=self.rna_a_hb_1,
            theta0_hb_1=self.rna_theta0_hb_1,
            delta_theta_star_hb_1=self.rna_delta_theta_star_hb_1,
            a_hb_2=self.rna_a_hb_2,
            theta0_hb_2=self.rna_theta0_hb_2,
            delta_theta_star_hb_2=self.rna_delta_theta_star_hb_2,
            a_hb_3=self.rna_a_hb_3,
            theta0_hb_3=self.rna_theta0_hb_3,
            delta_theta_star_hb_3=self.rna_delta_theta_star_hb_3,
            a_hb_4=self.rna_a_hb_4,
            theta0_hb_4=self.rna_theta0_hb_4,
            delta_theta_star_hb_4=self.rna_delta_theta_star_hb_4,
            a_hb_7=self.rna_a_hb_7,
            theta0_hb_7=self.rna_theta0_hb_7,
            delta_theta_star_hb_7=self.rna_delta_theta_star_hb_7,
            a_hb_8=self.rna_a_hb_8,
            theta0_hb_8=self.rna_theta0_hb_8,
            delta_theta_star_hb_8=self.rna_delta_theta_star_hb_8,
            ss_hb_weights=self.rna_ss_hb_weights,
        ).init_params()

        drh_config = dna1_energy.HydrogenBondingConfiguration(
            eps_hb=self.drh_eps_hb,
            a_hb=self.drh_a_hb,
            dr0_hb=self.drh_dr0_hb,
            dr_c_hb=self.drh_dr_c_hb,
            dr_low_hb=self.drh_dr_low_hb,
            dr_high_hb=self.drh_dr_high_hb,
            a_hb_1=self.drh_a_hb_1,
            theta0_hb_1=self.drh_theta0_hb_1,
            delta_theta_star_hb_1=self.drh_delta_theta_star_hb_1,
            a_hb_2=self.drh_a_hb_2,
            theta0_hb_2=self.drh_theta0_hb_2,
            delta_theta_star_hb_2=self.drh_delta_theta_star_hb_2,
            a_hb_3=self.drh_a_hb_3,
            theta0_hb_3=self.drh_theta0_hb_3,
            delta_theta_star_hb_3=self.drh_delta_theta_star_hb_3,
            a_hb_4=self.drh_a_hb_4,
            theta0_hb_4=self.drh_theta0_hb_4,
            delta_theta_star_hb_4=self.drh_delta_theta_star_hb_4,
            a_hb_7=self.drh_a_hb_7,
            theta0_hb_7=self.drh_theta0_hb_7,
            delta_theta_star_hb_7=self.drh_delta_theta_star_hb_7,
            a_hb_8=self.drh_a_hb_8,
            theta0_hb_8=self.drh_theta0_hb_8,
            delta_theta_star_hb_8=self.drh_delta_theta_star_hb_8,
            ss_hb_weights=self.drh_ss_hb_weights,
        ).init_params()

        return self.replace(
            dna_config=dna_config,
            rna_config=rna_config,
            drh_config=drh_config,
        )


@chex.dataclass(frozen=True)
class HydrogenBonding(je_base.BaseEnergyFunction):
    """Hydrogen bonding energy function for NA1 model."""

    params: HydrogenBondingConfiguration

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

        dna_dgs = dna1_energy.HydrogenBonding(
            displacement_fn=self.displacement_fn, params=self.params.dna_config
        ).pairwise_energies(
            body.dna,
            body.dna,
            seq,
            unbonded_neighbors,
        )

        rna_dgs = dna1_energy.HydrogenBonding(
            displacement_fn=self.displacement_fn, params=self.params.rna_config
        ).pairwise_energies(
            body.rna,
            body.rna,
            seq,
            unbonded_neighbors,
        )

        drh_dgs = dna1_energy.HydrogenBonding(
            displacement_fn=self.displacement_fn, params=self.params.drh_config
        ).pairwise_energies(
            body.dna,
            body.rna,
            seq,
            unbonded_neighbors,
        )

        rdh_dgs = dna1_energy.HydrogenBonding(
            displacement_fn=self.displacement_fn, params=self.params.drh_config
        ).pairwise_energies(
            body.rna,
            body.dna,
            seq,
            unbonded_neighbors,
        )

        dgs = jnp.where(is_rna_bond, rna_dgs, jnp.where(is_drh_bond, drh_dgs, jnp.where(is_rdh_bond, rdh_dgs, dna_dgs)))
        dgs = jnp.where(mask, dgs, 0.0)

        return dgs.sum()
