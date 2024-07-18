import dataclasses as dc

import chex
import numpy as np

import jax_dna.input.configuration as config
import jax_dna.input.dna1.base_smoothing_functions as bsf


@chex.dataclass(frozen=True)
class ExcludedVolumeConfiguration(config.BaseConfiguration):
    # independent parameters
    eps_exc: float | None = None
    dr_star_base: float | None = None
    sigma_base: float | None = None
    sigma_back_base: float | None = None
    sigma_base_back: float | None = None
    dr_star_back_base: float | None = None
    dr_star_base_back: float | None = None

    # dependent parameters
    b_base: float | None = None
    dr_c_base: float | None = None
    b_back_base: float | None = None
    dr_c_back_base: float | None = None
    b_base_back: float | None = None
    dr_c_base_back: float | None = None

    # override
    required_params: list[str] = [
        "eps_exc",
        "dr_star_base",
        "sigma_base",
        "sigma_back_base",
        "sigma_base_back",
        "dr_star_back_base",
        "dr_star_base_back",
    ]

    def init_params(self) -> "ExcludedVolumeConfiguration":
        b_base, dr_c_base = bsf.get_f3_smoothing_params(self.dr_star_base, self.eps_exc, self.sigma_base)

        ## f3(dr_back_base)
        b_back_base, dr_c_back_base = bsf.get_f3_smoothing_params(
            self.dr_star_back_base, self.eps_exc, self.sigma_back_base
        )

        ## f3(dr_base_back)
        b_base_back, dr_c_base_back = bsf.get_f3_smoothing_params(
            self.dr_star_base_back,
            self.eps_exc,
            self.sigma_base_back,
        )

        return dc.replace(
            self,
            b_base=b_base,
            dr_c_base=dr_c_base,
            b_back_base=b_back_base,
            dr_c_back_base=dr_c_back_base,
            b_base_back=b_base_back,
            dr_c_base_back=dr_c_base_back,
        )

    @staticmethod
    def from_toml(file_path: str) -> "ExcludedVolumeConfiguration":
        return dc.replace(
            ExcludedVolumeConfiguration(), **ExcludedVolumeConfiguration.parse_toml(file_path, "excluded_volume")
        ).init_params()

    @staticmethod
    def from_dict(params: dict[str, float]) -> "ExcludedVolumeConfiguration":
        return dc.replace(ExcludedVolumeConfiguration(), **params).init_params()


@chex.dataclass(frozen=True)
class VFeneConfiguration(config.BaseConfiguration):
    # independent parameters
    eps_backbone: float | None = None
    r0_backbone: float | None = None
    delta_backbone: float | None = None
    fmax: float | None = None
    finf: float | None = None

    # override
    required_params: list[str] = ["eps_backbone", "r0_backbone", "delta_backbone", "fmax", "finf"]

    @staticmethod
    def from_toml(file_path: str) -> "VFeneConfiguration":
        return dc.replace(VFeneConfiguration(), **VFeneConfiguration.parse_toml(file_path, "vfene")).init_params()

    @staticmethod
    def from_dict(params: dict[str, float]) -> "VFeneConfiguration":
        return dc.replace(VFeneConfiguration(), **params).init_params()
