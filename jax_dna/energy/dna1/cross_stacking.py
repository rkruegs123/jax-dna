"""Cross-stacking energy term for DNA1 model."""

import chex
import jax.numpy as jnp
from typing_extensions import override

import jax_dna.energy.base as je_base
import jax_dna.energy.configuration as config
import jax_dna.energy.dna1.base_smoothing_functions as bsf
import jax_dna.energy.dna1.interactions as dna1_interactions
import jax_dna.energy.dna1.nucleotide as dna1_nucleotide
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class CrossStackingConfiguration(config.BaseConfiguration):
    """Configuration for the cross-stacking energy function."""

    # independent parameters ===================================================
    dr_low_cross: float | None = None
    dr_high_cross: float | None = None
    k_cross: float | None = None
    r0_cross: float | None = None
    dr_c_cross: float | None = None
    theta0_cross_1: float | None = None
    delta_theta_star_cross_1: float | None = None
    a_cross_1: float | None = None
    theta0_cross_2: float | None = None
    delta_theta_star_cross_2: float | None = None
    a_cross_2: float | None = None
    theta0_cross_3: float | None = None
    delta_theta_star_cross_3: float | None = None
    a_cross_3: float | None = None
    theta0_cross_4: float | None = None
    delta_theta_star_cross_4: float | None = None
    a_cross_4: float | None = None
    theta0_cross_7: float | None = None
    delta_theta_star_cross_7: float | None = None
    a_cross_7: float | None = None
    theta0_cross_8: float | None = None
    delta_theta_star_cross_8: float | None = None
    a_cross_8: float | None = None

    # dependent parameters =====================================================
    b_low_cross: float | None = None
    dr_c_low_cross: float | None = None
    b_high_cross: float | None = None
    dr_c_high_cross: float | None = None
    b_cross_1: float | None = None
    delta_theta_cross_1_c: float | None = None
    b_cross_2: float | None = None
    delta_theta_cross_2_c: float | None = None
    b_cross_3: float | None = None
    delta_theta_cross_3_c: float | None = None
    b_cross_4: float | None = None
    delta_theta_cross_4_c: float | None = None
    b_cross_7: float | None = None
    delta_theta_cross_7_c: float | None = None
    b_cross_8: float | None = None
    delta_theta_cross_8_c: float | None = None

    # override
    required_params: tuple[str] = (
        "dr_low_cross",
        "dr_high_cross",
        "k_cross",
        "r0_cross",
        "dr_c_cross",
        "theta0_cross_1",
        "delta_theta_star_cross_1",
        "a_cross_1",
        "theta0_cross_2",
        "delta_theta_star_cross_2",
        "a_cross_2",
        "theta0_cross_3",
        "delta_theta_star_cross_3",
        "a_cross_3",
        "theta0_cross_4",
        "delta_theta_star_cross_4",
        "a_cross_4",
        "theta0_cross_7",
        "delta_theta_star_cross_7",
        "a_cross_7",
        "theta0_cross_8",
        "delta_theta_star_cross_8",
        "a_cross_8",
    )

    @override
    def init_params(self) -> "CrossStackingConfiguration":
        # reference to f2(dr_hb)
        (
            b_low_cross,
            dr_c_low_cross,
            b_high_cross,
            dr_c_high_cross,
        ) = bsf.get_f2_smoothing_params(
            self.r0_cross,
            self.dr_c_cross,
            self.dr_low_cross,
            self.dr_high_cross,
        )

        # reference to f4(theta_1)
        b_cross_1, delta_theta_cross_1_c = bsf.get_f4_smoothing_params(
            self.a_cross_1,
            self.theta0_cross_1,
            self.delta_theta_star_cross_1,
        )

        # reference to f4(theta_2)
        b_cross_2, delta_theta_cross_2_c = bsf.get_f4_smoothing_params(
            self.a_cross_2,
            self.theta0_cross_2,
            self.delta_theta_star_cross_2,
        )

        # reference to f4(theta_3)
        b_cross_3, delta_theta_cross_3_c = bsf.get_f4_smoothing_params(
            self.a_cross_3,
            self.theta0_cross_3,
            self.delta_theta_star_cross_3,
        )

        # reference to f4(theta_4) + f4(pi - theta_4)
        b_cross_4, delta_theta_cross_4_c = bsf.get_f4_smoothing_params(
            self.a_cross_4,
            self.theta0_cross_4,
            self.delta_theta_star_cross_4,
        )

        # reference to f4(theta_7) + f4(pi - theta_7)
        b_cross_7, delta_theta_cross_7_c = bsf.get_f4_smoothing_params(
            self.a_cross_7,
            self.theta0_cross_7,
            self.delta_theta_star_cross_7,
        )

        # reference to f4(theta_8) + f4(pi - theta_8)
        b_cross_8, delta_theta_cross_8_c = bsf.get_f4_smoothing_params(
            self.a_cross_8,
            self.theta0_cross_8,
            self.delta_theta_star_cross_8,
        )

        return self.replace(
            b_low_cross=b_low_cross,
            dr_c_low_cross=dr_c_low_cross,
            b_high_cross=b_high_cross,
            dr_c_high_cross=dr_c_high_cross,
            b_cross_1=b_cross_1,
            delta_theta_cross_1_c=delta_theta_cross_1_c,
            b_cross_2=b_cross_2,
            delta_theta_cross_2_c=delta_theta_cross_2_c,
            b_cross_3=b_cross_3,
            delta_theta_cross_3_c=delta_theta_cross_3_c,
            b_cross_4=b_cross_4,
            delta_theta_cross_4_c=delta_theta_cross_4_c,
            b_cross_7=b_cross_7,
            delta_theta_cross_7_c=delta_theta_cross_7_c,
            b_cross_8=b_cross_8,
            delta_theta_cross_8_c=delta_theta_cross_8_c,
        )


@chex.dataclass(frozen=True)
class CrossStacking(je_base.BaseEnergyFunction):
    """Cross-stacking energy function for DNA1 model."""

    params: CrossStackingConfiguration

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
        r_base_op = jnp.linalg.norm(dr_base_op, axis=1)

        theta1_op = jnp.arccos(jd_math.clamp(jd_math.mult(-body.back_base_vectors[op_i], body.back_base_vectors[op_j])))
        theta2_op = jnp.arccos(jd_math.clamp(jd_math.mult(-body.back_base_vectors[op_j], dr_base_op) / r_base_op))
        theta3_op = jnp.arccos(jd_math.clamp(jd_math.mult(body.back_base_vectors[op_i], dr_base_op) / r_base_op))
        theta4_op = jnp.arccos(jd_math.clamp(jd_math.mult(body.base_normals[op_i], body.base_normals[op_j])))
        # note: are these swapped in Lorenzo's code?
        theta7_op = jnp.arccos(jd_math.clamp(jd_math.mult(-body.base_normals[op_j], dr_base_op) / r_base_op))
        theta8_op = jnp.pi - jnp.arccos(jd_math.clamp(jd_math.mult(body.base_normals[op_i], dr_base_op) / r_base_op))

        cr_stack_dg = dna1_interactions.cross_stacking(
            r_base_op,
            theta1_op,
            theta2_op,
            theta3_op,
            theta4_op,
            theta7_op,
            theta8_op,
            self.params.dr_low_cross,
            self.params.dr_high_cross,
            self.params.dr_c_low_cross,
            self.params.dr_c_high_cross,
            self.params.k_cross,
            self.params.r0_cross,
            self.params.dr_c_cross,
            self.params.b_low_cross,
            self.params.b_high_cross,
            self.params.theta0_cross_1,
            self.params.delta_theta_star_cross_1,
            self.params.delta_theta_cross_1_c,
            self.params.a_cross_1,
            self.params.b_cross_1,
            self.params.theta0_cross_2,
            self.params.delta_theta_star_cross_2,
            self.params.delta_theta_cross_2_c,
            self.params.a_cross_2,
            self.params.b_cross_2,
            self.params.theta0_cross_3,
            self.params.delta_theta_star_cross_3,
            self.params.delta_theta_cross_3_c,
            self.params.a_cross_3,
            self.params.b_cross_3,
            self.params.theta0_cross_4,
            self.params.delta_theta_star_cross_4,
            self.params.delta_theta_cross_4_c,
            self.params.a_cross_4,
            self.params.b_cross_4,
            self.params.theta0_cross_7,
            self.params.delta_theta_star_cross_7,
            self.params.delta_theta_cross_7_c,
            self.params.a_cross_7,
            self.params.b_cross_7,
            self.params.theta0_cross_8,
            self.params.delta_theta_star_cross_8,
            self.params.delta_theta_cross_8_c,
            self.params.a_cross_8,
            self.params.b_cross_8,
        )

        return jnp.where(mask, cr_stack_dg, 0.0).sum()  # Mask for neighbors
