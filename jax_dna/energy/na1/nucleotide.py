"""Extends `jax_md.rigid_body.RigidBody` for NA1 nucleotide."""

import chex
import jax_md

import jax_dna.utils.types as typ
from jax_dna.energy.dna2 import nucleotide as dna2_nucleotide
from jax_dna.energy.rna2 import nucleotide as rna2_nucleotide


@chex.dataclass(frozen=True)
class HybridNucleotide:
    """Nucleotide rigid body with additional sites for NA1.

    This class is inteneded to be used as a dataclass for a nucleotide rigid body
    as a `rigid_body_transform_fn` in `jax_md.energy.ComposedEnergyFunction`.
    """

    dna: dna2_nucleotide.Nucleotide
    rna: rna2_nucleotide.Nucleotide

    @staticmethod
    def from_rigid_body(
        rigid_body: jax_md.rigid_body.RigidBody,
        # DNA2-specific
        dna_com_to_backbone_x: typ.Scalar,
        dna_com_to_backbone_y: typ.Scalar,
        dna_com_to_backbone_dna1: typ.Scalar,
        dna_com_to_hb: typ.Scalar,
        dna_com_to_stacking: typ.Scalar,
        # RNA2-specific
        rna_com_to_backbone_x: typ.Scalar,
        rna_com_to_backbone_y: typ.Scalar,
        rna_com_to_stacking: typ.Scalar,
        rna_com_to_hb: typ.Scalar,
        rna_p3_x: typ.Scalar,
        rna_p3_y: typ.Scalar,
        rna_p3_z: typ.Scalar,
        rna_p5_x: typ.Scalar,
        rna_p5_y: typ.Scalar,
        rna_p5_z: typ.Scalar,
        rna_pos_stack_3_a1: typ.Scalar,
        rna_pos_stack_3_a2: typ.Scalar,
        rna_pos_stack_5_a1: typ.Scalar,
        rna_pos_stack_5_a2: typ.Scalar,
    ) -> "HybridNucleotide":
        """Class method to precompute nucleotide sites from a rigid body."""
        dna = dna2_nucleotide.Nucleotide.from_rigid_body(
            rigid_body,
            dna_com_to_backbone_x,
            dna_com_to_backbone_y,
            dna_com_to_backbone_dna1,
            dna_com_to_hb,
            dna_com_to_stacking,
        )

        rna = rna2_nucleotide.Nucleotide.from_rigid_body(
            rigid_body,
            rna_com_to_backbone_x,
            rna_com_to_backbone_y,
            rna_com_to_stacking,
            rna_com_to_hb,
            rna_p3_x,
            rna_p3_y,
            rna_p3_z,
            rna_p5_x,
            rna_p5_y,
            rna_p5_z,
            rna_pos_stack_3_a1,
            rna_pos_stack_3_a2,
            rna_pos_stack_5_a1,
            rna_pos_stack_5_a2,
        )

        return HybridNucleotide(
            dna=dna,
            rna=rna,
        )
