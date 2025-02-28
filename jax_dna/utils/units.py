"""Units for the oxDNA model."""

import jax_dna.utils.types as jd_types

ANGSTROMS_PER_OXDNA_LENGTH = 8.518
ANGSTROMS_PER_NM = 10
NM_PER_OXDNA_LENGTH = ANGSTROMS_PER_OXDNA_LENGTH / ANGSTROMS_PER_NM
PN_PER_OXDNA_FORCE = 48.63
JOULES_PER_OXDNA_ENERGY = 4.142e-20


def get_kt(t_kelvin: jd_types.ARR_OR_SCALAR) -> jd_types.ARR_OR_SCALAR:
    """Converts a temperature in Kelvin to kT in simulation units."""
    return 0.1 * t_kelvin / 300.0
