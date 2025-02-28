"""oxDNA1 energy implementation in jax_dna."""

from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp

from jax_dna.energy.base import BaseEnergyFunction
from jax_dna.energy.configuration import BaseConfiguration
from jax_dna.energy.dna1.bonded_excluded_volume import BondedExcludedVolume, BondedExcludedVolumeConfiguration
from jax_dna.energy.dna1.coaxial_stacking import CoaxialStacking, CoaxialStackingConfiguration
from jax_dna.energy.dna1.cross_stacking import CrossStacking, CrossStackingConfiguration
from jax_dna.energy.dna1.expected_hydrogen_bonding import ExpectedHydrogenBonding, ExpectedHydrogenBondingConfiguration
from jax_dna.energy.dna1.expected_stacking import ExpectedStacking, ExpectedStackingConfiguration
from jax_dna.energy.dna1.fene import Fene, FeneConfiguration
from jax_dna.energy.dna1.hydrogen_bonding import HydrogenBonding, HydrogenBondingConfiguration
from jax_dna.energy.dna1.nucleotide import Nucleotide
from jax_dna.energy.dna1.stacking import Stacking, StackingConfiguration
from jax_dna.energy.dna1.unbonded_excluded_volume import UnbondedExcludedVolume, UnbondedExcludedVolumeConfiguration
from jax_dna.input import toml
from jax_dna.utils.types import PyTree


def default_configs() -> tuple[PyTree, PyTree]:
    """Return the default simulation and energy configuration files for dna1 simulations."""
    config_dir = (
        # jax_dna/energy/dna1/__init__.py
        Path(__file__)
        .resolve()
        # jax_dna/
        .parent.parent.parent
        # jax_dna/input/dna1
        .joinpath("input")
        .joinpath("dna1")
    )

    def cast_f(x: float | list[float]) -> jnp.ndarray:
        return jnp.array(x, dtype=jnp.float64)

    return (
        jax.tree.map(cast_f, toml.parse_toml(config_dir.joinpath("default_simulation.toml"))),
        jax.tree.map(cast_f, toml.parse_toml(config_dir.joinpath("default_energy.toml"))),
    )


def default_energy_configs(
    overrides: dict = MappingProxyType({}), opts: dict = MappingProxyType({})
) -> list[BaseConfiguration]:
    """Return the default configurations for the energy functions."""
    default_sim_config, default_config = default_configs()

    def get_param(x: str) -> dict:
        return default_config[x] | overrides.get(x, {})

    def get_opts(x: str, defaults: tuple[str] = BaseConfiguration.OPT_ALL) -> tuple[str]:
        return opts.get(x, defaults)

    default_stacking_opts = tuple(set(default_config["stacking"].keys()) - {"kT", "ss_stack_weights"})

    return [
        FeneConfiguration.from_dict(get_param("fene"), get_opts("fene")),
        BondedExcludedVolumeConfiguration.from_dict(
            get_param("bonded_excluded_volume"), get_opts("bonded_excluded_volume")
        ),
        StackingConfiguration.from_dict(
            get_param("stacking") | {"kt": overrides.get("kT", default_sim_config["kT"])},
            get_opts("stacking", default_stacking_opts),
        ),
        UnbondedExcludedVolumeConfiguration.from_dict(get_param("unbonded_excluded_volume")),
        HydrogenBondingConfiguration.from_dict(get_param("hydrogen_bonding"), get_opts("hydrogen_bonding")),
        CrossStackingConfiguration.from_dict(get_param("cross_stacking"), get_opts("cross_stacking")),
        CoaxialStackingConfiguration.from_dict(get_param("coaxial_stacking"), get_opts("coaxial_stacking")),
    ]


def default_energy_fns() -> list[BaseEnergyFunction]:
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
    "ExpectedHydrogenBondingConfiguration",
    "ExpectedHydrogenBonding",
    "ExpectedStackingConfiguration",
    "ExpectedStacking",
    "default_configs",
    "default_energy_fns",
]
