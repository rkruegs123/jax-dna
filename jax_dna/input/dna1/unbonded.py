import dataclasses as dc

import chex

import jax_dna.input.configuration as config
import jax_dna.input.dna1.base_smoothing_functions as bsf


@chex.dataclass(frozen=True)
class ExcludedVolumeConfiguration(config.BaseConfiguration):
    # independent parameters
    eps_exc: float | None = None
    dr_star_base: float | None = None
    sigma_base: float | None = None
    dr_star_back_base: float | None = None
    sigma_back_base: float | None = None
    dr_star_base_back: float | None = None
    sigma_base_back: float | None = None
    dr_star_backbone: float | None = None
    sigma_backbone: float | None = None

    # dependent parameters
    b_base: float | None = None
    dr_c_base: float | None = None
    b_back_base: float | None = None
    dr_c_back_base: float | None = None
    b_base_back: float | None = None
    dr_c_base_back: float | None = None
    b_backbone: float | None = None
    dr_c_backbone: float | None = None

    # override
    required_params: tuple[str] = (
        "eps_exc",
        "dr_star_base",
        "sigma_base",
        "dr_star_back_base",
        "sigma_back_base",
        "dr_star_base_back",
        "sigma_base_back",
        "dr_star_backbone",
        "sigma_backbone",
    )

    def init_params(self) -> "ExcludedVolumeConfiguration":
        ## f3(dr_base)
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

        ## f3(dr_backbone)
        b_backbone, dr_c_backbone = bsf.get_f3_smoothing_params(
            self.dr_star_backbone,
            self.eps_exc,
            self.sigma_backbone,
        )

        return dc.replace(
            self,
            b_base=b_base,
            dr_c_base=dr_c_base,
            b_back_base=b_back_base,
            dr_c_back_base=dr_c_back_base,
            b_base_back=b_base_back,
            dr_c_base_back=dr_c_base_back,
            b_backbone=b_backbone,
            dr_c_backbone=dr_c_backbone,
        )

    @staticmethod
    def from_toml(file_path: str, params_to_optimize: tuple[str] = ()) -> "ExcludedVolumeConfiguration":
        dict_params = ExcludedVolumeConfiguration.parse_toml(file_path, "unbonded_excluded_volume")
        return ExcludedVolumeConfiguration.from_dict(dict_params, params_to_optimize)


@chex.dataclass(frozen=True)
class HydrogenBondingConfiguration(config.BaseConfiguration):
    # independent parameters ===================================================
    ## f1(dr_hb)
    eps_hb: float | None = None
    a_hb: float | None = None
    dr0_hb: float | None = None
    dr_c_hb: float | None = None
    dr_low_hb: float | None = None
    dr_high_hb: float | None = None

    ## f4(theta_1)
    a_hb_1: float | None = None
    theta0_hb_1: float | None = None
    delta_theta_star_hb_1: float | None = None

    ## f4(theta_2)
    a_hb_2: float | None = None
    theta0_hb_2: float | None = None
    delta_theta_star_hb_2: float | None = None

    ## f4(theta_3)
    a_hb_3: float | None = None
    theta0_hb_3: float | None = None
    delta_theta_star_hb_3: float | None = None

    ## f4(theta_4)
    a_hb_4: float | None = None
    theta0_hb_4: float | None = None
    delta_theta_star_hb_4: float | None = None

    ## f4(theta_7)
    a_hb_7: float | None = None
    theta0_hb_7: float | None = None
    delta_theta_star_hb_7: float | None = None

    ## f4(theta_8)
    a_hb_8: float | None = None
    theta0_hb_8: float | None = None
    delta_theta_star_hb_8: float | None = None

    # dependent parameters =====================================================
    b_low_hb: float | None = None
    dr_c_low_hb: float | None = None
    b_high_hb: float | None = None
    dr_c_high_hb: float | None = None
    b_hb_1: float | None = None
    delta_theta_hb_1_c: float | None = None
    b_hb_2: float | None = None
    delta_theta_hb_2_c: float | None = None
    b_hb_3: float | None = None
    delta_theta_hb_3_c: float | None = None
    b_hb_4: float | None = None
    delta_theta_hb_4_c: float | None = None
    b_hb_7: float | None = None
    delta_theta_hb_7_c: float | None = None
    b_hb_8: float | None = None
    delta_theta_hb_8_c: float | None = None

    # override
    required_params: tuple[str] = (
        "eps_hb",
        "a_hb",
        "dr0_hb",
        "dr_c_hb",
        "dr_low_hb",
        "dr_high_hb",
        "a_hb_1",
        "theta0_hb_1",
        "delta_theta_star_hb_1",
        "a_hb_2",
        "theta0_hb_2",
        "delta_theta_star_hb_2",
        "a_hb_3",
        "theta0_hb_3",
        "delta_theta_star_hb_3",
        "a_hb_4",
        "theta0_hb_4",
        "delta_theta_star_hb_4",
        "a_hb_7",
        "theta0_hb_7",
        "delta_theta_star_hb_7",
        "a_hb_8",
        "theta0_hb_8",
        "delta_theta_star_hb_8",
    )

    def init_params(self) -> "HydrogenBondingConfiguration":
        ## f1(dr_hb)
        b_low_hb, dr_c_low_hb, b_high_hb, dr_c_high_hb = bsf.get_f1_smoothing_params(
            self.eps_hb,
            self.dr0_hb,
            self.a_hb,
            self.dr_c_hb,
            self.dr_low_hb,
            self.dr_high_hb,
        )

        ## f4(theta_1)
        b_hb_1, delta_theta_hb_1_c = bsf.get_f4_smoothing_params(
            self.a_hb_1,
            self.theta0_hb_1,
            self.delta_theta_star_hb_1,
        )

        ## f4(theta_2)
        b_hb_2, delta_theta_hb_2_c = bsf.get_f4_smoothing_params(
            self.a_hb_2,
            self.theta0_hb_2,
            self.delta_theta_star_hb_2,
        )

        ## f4(theta_3)
        b_hb_3, delta_theta_hb_3_c = bsf.get_f4_smoothing_params(
            self.a_hb_3,
            self.theta0_hb_3,
            self.delta_theta_star_hb_3,
        )

        ## f4(theta_4)
        b_hb_4, delta_theta_hb_4_c = bsf.get_f4_smoothing_params(
            self.a_hb_4,
            self.theta0_hb_4,
            self.delta_theta_star_hb_4,
        )

        ## f4(theta_7)
        b_hb_7, delta_theta_hb_7_c = bsf.get_f4_smoothing_params(
            self.a_hb_7,
            self.theta0_hb_7,
            self.delta_theta_star_hb_7,
        )

        ## f4(theta_8)
        b_hb_8, delta_theta_hb_8_c = bsf.get_f4_smoothing_params(
            self.a_hb_8,
            self.theta0_hb_8,
            self.delta_theta_star_hb_8,
        )

        return dc.replace(
            self,
            b_low_hb=b_low_hb,
            dr_c_low_hb=dr_c_low_hb,
            b_high_hb=b_high_hb,
            dr_c_high_hb=dr_c_high_hb,
            b_hb_1=b_hb_1,
            delta_theta_hb_1_c=delta_theta_hb_1_c,
            b_hb_2=b_hb_2,
            delta_theta_hb_2_c=delta_theta_hb_2_c,
            b_hb_3=b_hb_3,
            delta_theta_hb_3_c=delta_theta_hb_3_c,
            b_hb_4=b_hb_4,
            delta_theta_hb_4_c=delta_theta_hb_4_c,
            b_hb_7=b_hb_7,
            delta_theta_hb_7_c=delta_theta_hb_7_c,
            b_hb_8=b_hb_8,
            delta_theta_hb_8_c=delta_theta_hb_8_c,
        )

    @staticmethod
    def from_toml(file_path: str, params_to_optimize: tuple[str] = ()) -> "HydrogenBondingConfiguration":
        dict_params = HydrogenBondingConfiguration.parse_toml(file_path, "hydrogen_bonding")
        return HydrogenBondingConfiguration.from_dict(dict_params, params_to_optimize)


@chex.dataclass(frozen=True)
class CrossStackingConfiguration(config.BaseConfiguration):
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

    def init_params(self) -> "CrossStackingConfiguration":
        ## f2(dr_hb)
        (
            b_low_cross,
            dr_c_low_cross,
            b_high_cross,
            dr_c_high_cross,
        ) = bsf.get_f2_smoothing_params(
            self.k_cross,
            self.r0_cross,
            self.dr_c_cross,
            self.dr_low_cross,
            self.dr_high_cross,
        )

        ## f4(theta_1)
        b_cross_1, delta_theta_cross_1_c = bsf.get_f4_smoothing_params(
            self.a_cross_1,
            self.theta0_cross_1,
            self.delta_theta_star_cross_1,
        )

        ## f4(theta_2)
        b_cross_2, delta_theta_cross_2_c = bsf.get_f4_smoothing_params(
            self.a_cross_2,
            self.theta0_cross_2,
            self.delta_theta_star_cross_2,
        )

        ## f4(theta_3)
        b_cross_3, delta_theta_cross_3_c = bsf.get_f4_smoothing_params(
            self.a_cross_3,
            self.theta0_cross_3,
            self.delta_theta_star_cross_3,
        )

        ## f4(theta_4) + f4(pi - theta_4)
        b_cross_4, delta_theta_cross_4_c = bsf.get_f4_smoothing_params(
            self.a_cross_4,
            self.theta0_cross_4,
            self.delta_theta_star_cross_4,
        )

        ## f4(theta_7) + f4(pi - theta_7)
        b_cross_7, delta_theta_cross_7_c = bsf.get_f4_smoothing_params(
            self.a_cross_7,
            self.theta0_cross_7,
            self.delta_theta_star_cross_7,
        )

        ## f4(theta_8) + f4(pi - theta_8)
        b_cross_8, delta_theta_cross_8_c = bsf.get_f4_smoothing_params(
            self.a_cross_8,
            self.theta0_cross_8,
            self.delta_theta_star_cross_8,
        )

        return dc.replace(
            self,
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

    @staticmethod
    def from_toml(file_path: str, params_to_optimize: tuple[str] = ()) -> "CrossStackingConfiguration":
        dict_params = CrossStackingConfiguration.parse_toml(file_path, "cross_stacking")
        return CrossStackingConfiguration.from_dict(dict_params, params_to_optimize)


@chex.dataclass(frozen=True)
class CoaxialStackingConfiguration(config.BaseConfiguration):
    # independent parameters ===================================================
    dr_low_coax: float | None = None
    dr_high_coax: float | None = None
    k_coax: float | None = None
    dr0_coax: float | None = None
    dr_c_coax: float | None = None
    theta0_coax_4: float | None = None
    delta_theta_star_coax_4: float | None = None
    a_coax_4: float | None = None
    theta0_coax_1: float | None = None
    delta_theta_star_coax_1: float | None = None
    a_coax_1: float | None = None
    theta0_coax_5: float | None = None
    delta_theta_star_coax_5: float | None = None
    a_coax_5: float | None = None
    theta0_coax_6: float | None = None
    delta_theta_star_coax_6: float | None = None
    a_coax_6: float | None = None
    cos_phi3_star_coax: float | None = None
    a_coax_3p: float | None = None
    cos_phi4_star_coax: float | None = None
    a_coax_4p: float | None = None

    # dependent parameters =====================================================
    b_low_coax: float | None = None
    dr_c_low_coax: float | None = None
    b_high_coax: float | None = None
    dr_c_high_coax: float | None = None
    b_coax_4: float | None = None
    delta_theta_coax_4_c: float | None = None
    b_coax_1: float | None = None
    delta_theta_coax_1_c: float | None = None
    b_coax_5: float | None = None
    delta_theta_coax_5_c: float | None = None
    b_coax_6: float | None = None
    delta_theta_coax_6_c: float | None = None
    b_cos_phi3_coax: float | None = None
    cos_phi3_c_coax: float | None = None
    b_cos_phi4_coax: float | None = None
    cos_phi4_c_coax: float | None = None

    # override
    required_params: tuple[str] = (
        "dr_low_coax",
        "dr_high_coax",
        "k_coax",
        "dr0_coax",
        "dr_c_coax",
        "theta0_coax_4",
        "delta_theta_star_coax_4",
        "a_coax_4",
        "theta0_coax_1",
        "delta_theta_star_coax_1",
        "a_coax_1",
        "theta0_coax_5",
        "delta_theta_star_coax_5",
        "a_coax_5",
        "theta0_coax_6",
        "delta_theta_star_coax_6",
        "a_coax_6",
        "cos_phi3_star_coax",
        "a_coax_3p",
        "cos_phi4_star_coax",
        "a_coax_4p",
    )

    def init_params(self) -> "CoaxialStackingConfiguration":
        ## f2(dr_coax)
        b_low_coax, dr_c_low_coax, b_high_coax, dr_c_high_coax = bsf.get_f2_smoothing_params(
            self.k_coax,
            self.dr0_coax,
            self.dr_c_coax,
            self.dr_low_coax,
            self.dr_high_coax,
        )

        ## f4(theta_4)
        b_coax_4, delta_theta_coax_4_c = bsf.get_f4_smoothing_params(
            self.a_coax_4,
            self.theta0_coax_4,
            self.delta_theta_star_coax_4,
        )

        ## f4(theta_1) + f4(2*pi - theta_1)
        b_coax_1, delta_theta_coax_1_c = bsf.get_f4_smoothing_params(
            self.a_coax_1,
            self.theta0_coax_1,
            self.delta_theta_star_coax_1,
        )

        ## f4(theta_5) + f4(pi - theta_5)
        b_coax_5, delta_theta_coax_5_c = bsf.get_f4_smoothing_params(
            self.a_coax_5,
            self.theta0_coax_5,
            self.delta_theta_star_coax_5,
        )

        ## f4(theta_6) + f4(pi - theta_6)
        b_coax_6, delta_theta_coax_6_c = bsf.get_f4_smoothing_params(
            self.a_coax_6,
            self.theta0_coax_6,
            self.delta_theta_star_coax_6,
        )

        ## f5(cos(phi3))
        b_cos_phi3_coax, cos_phi3_c_coax = bsf.get_f5_smoothing_params(
            self.a_coax_3p,
            self.cos_phi3_star_coax,
        )

        ## f5(cos(phi4))
        b_cos_phi4_coax, cos_phi4_c_coax = bsf.get_f5_smoothing_params(
            self.a_coax_4p,
            self.cos_phi4_star_coax,
        )

        return dc.replace(
            self,
            b_low_coax=b_low_coax,
            dr_c_low_coax=dr_c_low_coax,
            b_high_coax=b_high_coax,
            dr_c_high_coax=dr_c_high_coax,
            b_coax_4=b_coax_4,
            delta_theta_coax_4_c=delta_theta_coax_4_c,
            b_coax_1=b_coax_1,
            delta_theta_coax_1_c=delta_theta_coax_1_c,
            b_coax_5=b_coax_5,
            delta_theta_coax_5_c=delta_theta_coax_5_c,
            b_coax_6=b_coax_6,
            delta_theta_coax_6_c=delta_theta_coax_6_c,
            b_cos_phi3_coax=b_cos_phi3_coax,
            cos_phi3_c_coax=cos_phi3_c_coax,
            b_cos_phi4_coax=b_cos_phi4_coax,
            cos_phi4_c_coax=cos_phi4_c_coax,
        )

    @staticmethod
    def from_toml(file_path: str, params_to_optimize: tuple[str] = ()) -> "CoaxialStackingConfiguration":
        dict_params = CoaxialStackingConfiguration.parse_toml(file_path, "coaxial_stacking")
        return CoaxialStackingConfiguration.from_dict(dict_params, params_to_optimize)
