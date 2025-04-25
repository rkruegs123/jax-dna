"""Extends `jax_md.rigid_body.RigidBody` for RNA2 nucleotide."""

import chex
import jax_md

import jax_dna.energy.base as je_base
import jax_dna.energy.utils as je_utils
import jax_dna.utils.types as typ


@chex.dataclass(frozen=True)
class Nucleotide(je_base.BaseNucleotide):
    """Nucleotide rigid body with additional sites for RNA2.

    This class is inteneded to be used as a dataclass for a nucleotide rigid body
    as a `rigid_body_transform_fn` in `jax_md.energy.ComposedEnergyFunction`.
    """

    center: typ.Arr_Nucleotide_3
    orientation: typ.Arr_Nucleotide_3 | jax_md.rigid_body.Quaternion
    stack_sites: typ.Arr_Nucleotide_3
    back_sites: typ.Arr_Nucleotide_3
    base_sites: typ.Arr_Nucleotide_3
    back_base_vectors: typ.Arr_Nucleotide_3
    base_normals: typ.Arr_Nucleotide_3
    cross_prods: typ.Arr_Nucleotide_3
    bb_p3_sites: typ.Arr_Nucleotide_3
    bb_p5_sites: typ.Arr_Nucleotide_3
    stack3_sites: typ.Arr_Nucleotide_3
    stack5_sites: typ.Arr_Nucleotide_3

    @staticmethod
    def from_rigid_body(
        rigid_body: jax_md.rigid_body.RigidBody,
        com_to_backbone_x: typ.Scalar,
        com_to_backbone_y: typ.Scalar,
        com_to_stacking: typ.Scalar,
        com_to_hb: typ.Scalar,
        p3_x: typ.Scalar,
        p3_y: typ.Scalar,
        p3_z: typ.Scalar,
        p5_x: typ.Scalar,
        p5_y: typ.Scalar,
        p5_z: typ.Scalar,
        pos_stack_3_a1: typ.Scalar,
        pos_stack_3_a2: typ.Scalar,
        pos_stack_5_a1: typ.Scalar,
        pos_stack_5_a2: typ.Scalar,
    ) -> "Nucleotide":
        """Class method to precompute nucleotide sites from a rigid body."""
        back_base_vectors = je_utils.q_to_back_base(rigid_body.orientation)
        base_normals = je_utils.q_to_base_normal(rigid_body.orientation)
        cross_prods = je_utils.q_to_cross_prod(rigid_body.orientation)

        back_sites = rigid_body.center + com_to_backbone_x * back_base_vectors + com_to_backbone_y * base_normals
        stack_sites = rigid_body.center + com_to_stacking * back_base_vectors
        base_sites = rigid_body.center + com_to_hb * back_base_vectors

        bb_p3_sites = p3_x * back_base_vectors + p3_y * cross_prods + p3_z * base_normals
        bb_p5_sites = p5_x * back_base_vectors + p5_y * cross_prods + p5_z * base_normals

        stack3_sites = rigid_body.center + pos_stack_3_a1 * back_base_vectors + pos_stack_3_a2 * cross_prods
        stack5_sites = rigid_body.center + pos_stack_5_a1 * back_base_vectors + pos_stack_5_a2 * cross_prods

        return Nucleotide(
            center=rigid_body.center,
            orientation=rigid_body.orientation,
            back_base_vectors=back_base_vectors,
            base_normals=base_normals,
            cross_prods=cross_prods,
            stack_sites=stack_sites,
            back_sites=back_sites,
            base_sites=base_sites,
            bb_p3_sites=bb_p3_sites,
            bb_p5_sites=bb_p5_sites,
            stack3_sites=stack3_sites,
            stack5_sites=stack5_sites,
        )
