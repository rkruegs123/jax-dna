"""Units for the oxDNA model."""

ANGSTROMS_PER_OXDNA_LENGTH = 8.518
ANGSTROMS_PER_NM = 10
NM_PER_OXDNA_LENGTH = ANGSTROMS_PER_OXDNA_LENGTH / ANGSTROMS_PER_NM
PN_PER_OXDNA_FORCE = 48.63

def get_kt(t_kelvin):
    """Converts a temperature in Kelvin to kT in simulation units."""
    return 0.1 * t_kelvin / 300.0
