"""Hydrogen bonding energy function for DNA1 model."""

import dataclasses as dc

import chex
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1.base_smoothing_functions as bsf
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.energy.dna1.nucleotide as dna1_nucleotide
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as typ

HB_WEIGHTS_SA = jnp.array(
    [
        [0.0, 0.0, 0.0, 1.0],  # AX
        [0.0, 0.0, 1.0, 0.0],  # CX
        [0.0, 1.0, 0.0, 0.0],  # GX
        [1.0, 0.0, 0.0, 0.0],  # TX
    ]
)


@chex.dataclass(frozen=True)
class HydrogenBondingConfiguration(config.BaseConfiguration):
    """Configuration for the hydrogen bonding energy function."""

    # independent parameters ===================================================
    # reference to f1(dr_hb)
    eps_hb: float | None = None
    a_hb: float | None = None
    dr0_hb: float | None = None
    dr_c_hb: float | None = None
    dr_low_hb: float | None = None
    dr_high_hb: float | None = None

    # reference to f4(theta_1)
    a_hb_1: float | None = None
    theta0_hb_1: float | None = None
    delta_theta_star_hb_1: float | None = None

    # reference to f4(theta_2)
    a_hb_2: float | None = None
    theta0_hb_2: float | None = None
    delta_theta_star_hb_2: float | None = None

    # reference to f4(theta_3)
    a_hb_3: float | None = None
    theta0_hb_3: float | None = None
    delta_theta_star_hb_3: float | None = None

    # reference to f4(theta_4)
    a_hb_4: float | None = None
    theta0_hb_4: float | None = None
    delta_theta_star_hb_4: float | None = None

    # reference to f4(theta_7)
    a_hb_7: float | None = None
    theta0_hb_7: float | None = None
    delta_theta_star_hb_7: float | None = None

    # reference to f4(theta_8)
    a_hb_8: float | None = None
    theta0_hb_8: float | None = None
    delta_theta_star_hb_8: float | None = None

    # required but not optimizable
    ss_hb_weights: np.ndarray | None = dc.field(default_factory=lambda:HB_WEIGHTS_SA)

    # dependent parameters =====================================================
    b_low_hb: float | None = None
    dr_c_low_hb: float | None = None
    b_high_hb: float | None = None
    dr_c_high_hb: float | None = None
    b_hb_1: float | None = None
    delta_theta_hb_1_c: float | None = None
    b_hb_2: float | None = None
    delta_theta_hb_2_c: float | None = None
    b_hb_3: float | None = None
    delta_theta_hb_3_c: float | None = None
    b_hb_4: float | None = None
    delta_theta_hb_4_c: float | None = None
    b_hb_7: float | None = None
    delta_theta_hb_7_c: float | None = None
    b_hb_8: float | None = None
    delta_theta_hb_8_c: float | None = None

    # override
    required_params: tuple[str] = (
        # Sequence-independence
        "eps_hb",
        "a_hb",
        "dr0_hb",
        "dr_c_hb",
        "dr_low_hb",
        "dr_high_hb",
        "a_hb_1",
        "theta0_hb_1",
        "delta_theta_star_hb_1",
        "a_hb_2",
        "theta0_hb_2",
        "delta_theta_star_hb_2",
        "a_hb_3",
        "theta0_hb_3",
        "delta_theta_star_hb_3",
        "a_hb_4",
        "theta0_hb_4",
        "delta_theta_star_hb_4",
        "a_hb_7",
        "theta0_hb_7",
        "delta_theta_star_hb_7",
        "a_hb_8",
        "theta0_hb_8",
        "delta_theta_star_hb_8",
        # Sequence-dependence
        "ss_hb_weights",
    )

    @override
    def init_params(self) -> "HydrogenBondingConfiguration":
        # reference to f1(dr_hb)
        b_low_hb, dr_c_low_hb, b_high_hb, dr_c_high_hb = bsf.get_f1_smoothing_params(
            self.dr0_hb,
            self.a_hb,
            self.dr_c_hb,
            self.dr_low_hb,
            self.dr_high_hb,
        )

        # reference to f4(theta_1)
        b_hb_1, delta_theta_hb_1_c = bsf.get_f4_smoothing_params(
            self.a_hb_1,
            self.theta0_hb_1,
            self.delta_theta_star_hb_1,
        )

        # reference to f4(theta_2)
        b_hb_2, delta_theta_hb_2_c = bsf.get_f4_smoothing_params(
            self.a_hb_2,
            self.theta0_hb_2,
            self.delta_theta_star_hb_2,
        )

        # reference to f4(theta_3)
        b_hb_3, delta_theta_hb_3_c = bsf.get_f4_smoothing_params(
            self.a_hb_3,
            self.theta0_hb_3,
            self.delta_theta_star_hb_3,
        )

        # reference to f4(theta_4)
        b_hb_4, delta_theta_hb_4_c = bsf.get_f4_smoothing_params(
            self.a_hb_4,
            self.theta0_hb_4,
            self.delta_theta_star_hb_4,
        )

        # reference to f4(theta_7)
        b_hb_7, delta_theta_hb_7_c = bsf.get_f4_smoothing_params(
            self.a_hb_7,
            self.theta0_hb_7,
            self.delta_theta_star_hb_7,
        )

        # reference to f4(theta_8)
        b_hb_8, delta_theta_hb_8_c = bsf.get_f4_smoothing_params(
            self.a_hb_8,
            self.theta0_hb_8,
            self.delta_theta_star_hb_8,
        )

        return self.replace(
            b_low_hb=b_low_hb,
            dr_c_low_hb=dr_c_low_hb,
            b_high_hb=b_high_hb,
            dr_c_high_hb=dr_c_high_hb,
            b_hb_1=b_hb_1,
            delta_theta_hb_1_c=delta_theta_hb_1_c,
            b_hb_2=b_hb_2,
            delta_theta_hb_2_c=delta_theta_hb_2_c,
            b_hb_3=b_hb_3,
            delta_theta_hb_3_c=delta_theta_hb_3_c,
            b_hb_4=b_hb_4,
            delta_theta_hb_4_c=delta_theta_hb_4_c,
            b_hb_7=b_hb_7,
            delta_theta_hb_7_c=delta_theta_hb_7_c,
            b_hb_8=b_hb_8,
            delta_theta_hb_8_c=delta_theta_hb_8_c,
        )


@chex.dataclass(frozen=True)
class HydrogenBonding(je_base.BaseEnergyFunction):
    """Hydrogen bonding energy function for DNA1 model."""

    params: HydrogenBondingConfiguration

    def compute_v_hb(
        self,
        body: dna1_nucleotide.Nucleotide,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors,
    ) -> typ.Arr_Unbonded_Neighbors:
        """Computes the sequence-independent energy for each unbonded pair."""
        op_i = unbonded_neighbors[0]
        op_j = unbonded_neighbors[1]
        mask = jnp.array(op_i < body.center.shape[0], dtype=jnp.float64)

        dr_base_op = self.displacement_mapped(body.base_sites[op_j], body.base_sites[op_i])  # Note the flip here
        r_base_op = jnp.linalg.norm(dr_base_op, axis=1)

        theta1_op = jnp.arccos(jd_math.clamp(jd_math.mult(-body.back_base_vectors[op_i], body.back_base_vectors[op_j])))
        theta2_op = jnp.arccos(jd_math.clamp(jd_math.mult(-body.back_base_vectors[op_j], dr_base_op) / r_base_op))
        theta3_op = jnp.arccos(jd_math.clamp(jd_math.mult(body.back_base_vectors[op_i], dr_base_op) / r_base_op))
        theta4_op = jnp.arccos(jd_math.clamp(jd_math.mult(body.base_normals[op_i], body.base_normals[op_j])))
        # note: are these swapped in Lorenzo's code?
        theta7_op = jnp.arccos(jd_math.clamp(jd_math.mult(-body.base_normals[op_j], dr_base_op) / r_base_op))
        theta8_op = jnp.pi - jnp.arccos(jd_math.clamp(jd_math.mult(body.base_normals[op_i], dr_base_op) / r_base_op))

        v_hb = dna1_interactions.hydrogen_bonding(
            dr_base_op,
            theta1_op,
            theta2_op,
            theta3_op,
            theta4_op,
            theta7_op,
            theta8_op,
            self.params.dr_low_hb,
            self.params.dr_high_hb,
            self.params.dr_c_low_hb,
            self.params.dr_c_high_hb,
            self.params.eps_hb,
            self.params.a_hb,
            self.params.dr0_hb,
            self.params.dr_c_hb,
            self.params.b_low_hb,
            self.params.b_high_hb,
            self.params.theta0_hb_1,
            self.params.delta_theta_star_hb_1,
            self.params.a_hb_1,
            self.params.delta_theta_hb_1_c,
            self.params.b_hb_1,
            self.params.theta0_hb_2,
            self.params.delta_theta_star_hb_2,
            self.params.a_hb_2,
            self.params.delta_theta_hb_2_c,
            self.params.b_hb_2,
            self.params.theta0_hb_3,
            self.params.delta_theta_star_hb_3,
            self.params.a_hb_3,
            self.params.delta_theta_hb_3_c,
            self.params.b_hb_3,
            self.params.theta0_hb_4,
            self.params.delta_theta_star_hb_4,
            self.params.a_hb_4,
            self.params.delta_theta_hb_4_c,
            self.params.b_hb_4,
            self.params.theta0_hb_7,
            self.params.delta_theta_star_hb_7,
            self.params.a_hb_7,
            self.params.delta_theta_hb_7_c,
            self.params.b_hb_7,
            self.params.theta0_hb_8,
            self.params.delta_theta_star_hb_8,
            self.params.a_hb_8,
            self.params.delta_theta_hb_8_c,
            self.params.b_hb_8,
        )

        return jnp.where(mask, v_hb, 0.0)  # Mask for neighbors

    @override
    def __call__(
        self,
        body: dna1_nucleotide.Nucleotide,
        seq: typ.Discrete_Sequence,
        bonded_neighbors: typ.Arr_Bonded_Neighbors_2,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Scalar:
        # Compute sequence-independent energy for each unbonded pair
        v_hb = self.compute_v_hb(body, unbonded_neighbors)

        # Compute sequence-dependent weight for each unbonded pair
        op_i = unbonded_neighbors[0]
        op_j = unbonded_neighbors[1]
        hb_weights = self.params.ss_hb_weights[seq[op_i], seq[op_j]]

        # Return the weighted sum
        return jnp.dot(hb_weights, v_hb)
