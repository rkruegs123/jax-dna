
import dataclasses
from typing import Union

import jax_md

import jax_dna.energy.utils as je_utils
import jax_dna.utils.types as typ

class Nucleotide(jax_md.rigid_body.RigidBody):

    def __init__(
        self,
        center: typ.Arr_Nucleotide_3,
        orientation: Union[typ.Arr_Nucleotide_3, jax_md.rigid_body.Quaternion],
        com_to_backbone: typ.Scalar,
        com_to_hb: typ.Scalar,
        com_to_stacking: typ.Scalar,
    ):
        super().__init__(center, orientation)

        # These are space frame, normalized
        self.back_base_vectors = je_utils.q_to_back_base(self.orientation)
        self.base_normals = je_utils.q_to_base_normal(self.orientation)
        self.cross_prods = je_utils.q_to_cross_prod(self.orientation)

        self.stack_sites = self.center + com_to_stacking * self.back_base_vectors
        self.back_sites = self.center + com_to_backbone * self.back_base_vectors
        self.base_sites = self.center + com_to_hb * self.back_base_vectors

