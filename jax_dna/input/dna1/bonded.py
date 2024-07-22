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
    required_params: tuple[str] = (
        "eps_exc",
        "dr_star_base",
        "sigma_base",
        "sigma_back_base",
        "sigma_base_back",
        "dr_star_back_base",
        "dr_star_base_back",
    )

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
    def from_toml(file_path: str, params_to_optimize: tuple[str] = ()) -> "ExcludedVolumeConfiguration":
        dict_params = ExcludedVolumeConfiguration.parse_toml(file_path, "bonded_excluded_volume")
        return ExcludedVolumeConfiguration.from_dict(dict_params, params_to_optimize)

    @staticmethod
    def from_dict(params: dict[str, float], params_to_optimize: tuple[str] = ()) -> "ExcludedVolumeConfiguration":
        return dc.replace(
            ExcludedVolumeConfiguration(), **(params | {"params_to_optimize": params_to_optimize})
        ).init_params()


@chex.dataclass(frozen=True)
class FeneConfiguration(config.BaseConfiguration):
    # independent parameters
    eps_backbone: float | None = None
    r0_backbone: float | None = None
    delta_backbone: float | None = None
    fmax: float | None = None
    finf: float | None = None

    # override
    required_params: tuple[str] = ("eps_backbone", "r0_backbone", "delta_backbone", "fmax", "finf")

    @staticmethod
    def from_toml(file_path: str, params_to_optimize: tuple[str] = ()) -> "FeneConfiguration":
        dict_params = FeneConfiguration.parse_toml(file_path, "vfene")
        FeneConfiguration.from_dict(dict_params, params_to_optimize)

    @staticmethod
    def from_dict(params: dict[str, float], params_to_optimize: tuple[str] = ()) -> "FeneConfiguration":
        return dc.replace(FeneConfiguration(), **(params | {"params_to_optimize": params_to_optimize})).init_params()


@chex.dataclass(frozen=True)
class StackingConfiguration(config.BaseConfiguration):
    # independent parameters
    eps_stack_base: float | None = None
    eps_stack_kt_coeff: float | None = None
    dr_low_stack: float | None = None
    dr_high_stack: float | None = None
    a_stack: float | None = None
    dr0_stack: float | None = None
    dr_c_stack: float | None = None
    theta0_stack_4: float | None = None
    delta_theta_star_stack_4: float | None = None
    a_stack_4: float | None = None
    theta0_stack_5: float | None = None
    delta_theta_star_stack_5: float | None = None
    a_stack_5: float | None = None
    theta0_stack_6: float | None = None
    delta_theta_star_stack_6: float | None = None
    a_stack_6: float | None = None
    neg_cos_phi1_star_stack: float | None = None
    a_stack_1: float | None = None
    neg_cos_phi2_star_stack: float | None = None
    a_stack_2: float | None = None
    kt: float | None = None
    ss_stack_weights: np.ndarray | None = None

    # dependent parameters
    b_low_stack: float | None = None
    dr_c_low_stack: float | None = None
    b_high_stack: float | None = None
    dr_c_high_stack: float | None = None
    b_stack_4: float | None = None
    delta_theta_stack_4_c: float | None = None
    b_stack_5: float | None = None
    delta_theta_stack_5_c: float | None = None
    b_stack_6: float | None = None
    delta_theta_stack_6_c: float | None = None
    b_neg_cos_phi1_stack: float | None = None
    neg_cos_phi1_c_stack: float | None = None
    b_neg_cos_phi2_stack: float | None = None
    neg_cos_phi2_c_stack: float | None = None
    eps_stack: float | None = None

    required_params: tuple[str] = (
        "eps_stack_base",
        "eps_stack_kt_coeff",
        "dr_low_stack",
        "dr_high_stack",
        "a_stack",
        "dr0_stack",
        "dr_c_stack",
        "theta0_stack_4",
        "delta_theta_star_stack_4",
        "a_stack_4",
        "theta0_stack_5",
        "delta_theta_star_stack_5",
        "a_stack_5",
        "theta0_stack_6",
        "delta_theta_star_stack_6",
        "a_stack_6",
        "neg_cos_phi1_star_stack",
        "a_stack_1",
        "neg_cos_phi2_star_stack",
        "a_stack_2",
        "ss_stack_weights",
    )

    @staticmethod
    def from_toml(file_path: str, params_to_optimize: tuple[str] = ()) -> "StackingConfiguration":
        full_toml = config.BaseConfiguration.parse_toml(file_path, "")

        dict_params = full_toml["stacking"].update(
            {
                "kt": full_toml["t_kelvin"],
                "ss_stack_weights": full_toml["stack_weights_sa"],
            }
        )

        return StackingConfiguration.from_dict(dict_params, params_to_optimize)

    @staticmethod
    def from_dict(params: dict[str, float], params_to_optimize: tuple[str] = ()) -> "StackingConfiguration":
        return dc.replace(
            StackingConfiguration(), **(params | {"params_to_optimize": params_to_optimize})
        ).init_params()

    def init_params(self) -> "StackingConfiguration":
        eps_stack = self.eps_stack_base + self.eps_stack_kt_coeff * self.kt

        b_low_stack, dr_c_low_stack, b_high_stack, dr_c_high_stack = bsf.get_f1_smoothing_params(
            self.eps_stack,
            self.dr0_stack,
            self.a_stack,
            self.dr_c_stack,
            self.dr_low_stack,
            self.dr_high_stack,
        )

        b_stack_4, delta_theta_stack_4_c = bsf.get_f4_smoothing_params(
            self.a_stack_4,
            self.theta0_stack_4,
            self.delta_theta_star_stack_4,
        )

        b_stack_5, delta_theta_stack_5_c = bsf.get_f4_smoothing_params(
            self.a_stack_5,
            self.theta0_stack_5,
            self.delta_theta_star_stack_5,
        )

        b_stack_6, delta_theta_stack_6_c = bsf.get_f4_smoothing_params(
            self.a_stack_6,
            self.theta0_stack_6,
            self.delta_theta_star_stack_6,
        )

        b_neg_cos_phi1_stack, neg_cos_phi1_c_stack = bsf.get_f5_smoothing_params(
            self.a_stack_1,
            self.neg_cos_phi1_star_stack,
        )

        b_neg_cos_phi2_stack, neg_cos_phi2_c_stack = bsf.get_f5_smoothing_params(
            self.a_stack_2,
            self.neg_cos_phi2_star_stack,
        )

        return dc.replace(
            self,
            b_low_stack=b_low_stack,
            dr_c_low_stack=dr_c_low_stack,
            b_high_stack=b_high_stack,
            dr_c_high_stack=dr_c_high_stack,
            b_stack_4=b_stack_4,
            delta_theta_stack_4_c=delta_theta_stack_4_c,
            b_stack_5=b_stack_5,
            delta_theta_stack_5_c=delta_theta_stack_5_c,
            b_stack_6=b_stack_6,
            delta_theta_stack_6_c=delta_theta_stack_6_c,
            b_neg_cos_phi1_stack=b_neg_cos_phi1_stack,
            neg_cos_phi1_c_stack=neg_cos_phi1_c_stack,
            b_neg_cos_phi2_stack=b_neg_cos_phi2_stack,
            neg_cos_phi2_c_stack=neg_cos_phi2_c_stack,
            eps_stack=eps_stack,
        )
