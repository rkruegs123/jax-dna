"""oxDNA1 energy implementation in jax_dna."""

import os
from pathlib import Path

import jax_dna.input.toml as toml
from jax_dna.energy.dna1.bonded_excluded_volume import BondedExcludedVolume, BondedExcludedVolumeConfiguration
from jax_dna.energy.dna1.coaxial_stacking import CoaxialStacking, CoaxialStackingConfiguration
from jax_dna.energy.dna1.cross_stacking import CrossStacking, CrossStackingConfiguration
from jax_dna.energy.dna1.fene import Fene, FeneConfiguration
from jax_dna.energy.dna1.hydrogen_bonding import HydrogenBonding, HydrogenBondingConfiguration
from jax_dna.energy.dna1.nucleotide import Nucleotide
from jax_dna.energy.dna1.stacking import Stacking, StackingConfiguration
from jax_dna.energy.dna1.unbonded_excluded_volume import UnbondedExcludedVolume, UnbondedExcludedVolumeConfiguration


def default_configs(overrides: dict = {}, opts: dict = {}):
    """Return the default configurations for the energy functions."""

    config_dir = (
        # jax_dna/energy/dna1/__init__.py
        Path(os.path.abspath(__file__))
        # jax_dna/
        .parent.parent.parent
        # jax_dna/input/dna1
        .joinpath("input")
        .joinpath("dna1")
    )

    default_sim_config = toml.parse_toml(config_dir.joinpath("default_simulation.toml"))
    default_config = toml.parse_toml(config_dir.joinpath("default_energy.toml"))

    get_param = lambda x: default_config[x] | overrides.get(x, {})
    get_opts = lambda x: opts.get(x, ("*",))

    return [
        FeneConfiguration.from_dict(get_param("fene"), get_opts("fene")),
        BondedExcludedVolumeConfiguration.from_dict(
            get_param("bonded_excluded_volume"), get_opts("bonded_excluded_volume")
        ),
        StackingConfiguration.from_dict(
            get_param("stacking") | {"kt": overrides.get("kT", default_sim_config["kT"])}, get_opts("stacking")
        ),
        UnbondedExcludedVolumeConfiguration.from_dict(get_param("unbonded_excluded_volume")),
        HydrogenBondingConfiguration.from_dict(get_param("hydrogen_bonding"), get_opts("hydrogen_bonding")),
        CrossStackingConfiguration.from_dict(get_param("cross_stacking"), get_opts("cross_stacking")),
        CoaxialStackingConfiguration.from_dict(get_param("coaxial_stacking"), get_opts("coaxial_stacking")),
    ]


def default_energy_fns():
    """Return the default energy functions."""
    return [
        Fene,
        BondedExcludedVolume,
        Stacking,
        UnbondedExcludedVolume,
        HydrogenBonding,
        CrossStacking,
        CoaxialStacking,
    ]


__all__ = [
    "Nucleotide",
    "Fene",
    "FeneConfiguration",
    "HydrogenBonding",
    "HydrogenBondingConfiguration",
    "Stacking",
    "StackingConfiguration",
    "CoaxialStacking",
    "CoaxialStackingConfiguration",
    "CrossStacking",
    "CrossStackingConfiguration",
    "BondedExcludedVolume",
    "BondedExcludedVolumeConfiguration",
    "UnbondedExcludedVolume",
    "UnbondedExcludedVolumeConfiguration",
    "default_configs",
    "default_energy_fns",
]
