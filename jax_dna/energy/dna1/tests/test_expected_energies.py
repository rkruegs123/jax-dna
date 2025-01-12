import functools
import itertools

import jax
import jax.numpy as jnp
import jax_md
import numpy as np
import pytest

import jax_dna.energy.configuration as jd_config
import jax_dna.energy.dna1 as jd_energy
import jax_dna.input.sequence_constraints as jd_sc
import jax_dna.input.toml as jd_toml
import jax_dna.input.topology as jd_top
import jax_dna.input.trajectory as jd_traj
import jax_dna.utils.constants as jd_const
import jax_dna.utils.types as typ

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - ignore boolean positional value
# this is a common jax practice


default_params = jd_toml.parse_toml("jax_dna/input/dna1/default_energy.toml")


@pytest.mark.parametrize(
    ("params", "expected_error"),
    [
        (
            default_params,
            jd_config.ERR_MISSING_REQUIRED_PARAMS.format(props="sequence_constraints"),
        ),
    ],
)
def test_expected_hb_config_raises_value_error(params: dict, expected_error: str):
    with pytest.raises(ValueError, match=expected_error):
        jd_energy.ExpectedHydrogenBondingConfiguration(**params["hydrogen_bonding"])


def test_initialize_expected_hb_config():
    sequence_constraints = jd_sc.SequenceConstraints(
        n_nucleotides=4,
        n_unpaired=2,
        n_bp=1,
        is_unpaired=np.array([0, 1, 1, 0]),
        unpaired=np.array([1, 2]),
        bps=np.array([[0, 3]]),
        idx_to_unpaired_idx=np.array([-1, 0, 1, -1]),
        idx_to_bp_idx=np.array([[0, 0], [-1, -1], [-1, -1], [0, 1]]),
    )

    jd_energy.ExpectedHydrogenBondingConfiguration(
        **default_params["hydrogen_bonding"], sequence_constraints=sequence_constraints
    )


COLUMN_NAMES = [
    "t",
    "fene",
    "bonded_excluded_volume",
    "stacking",
    "unbonded_excluded_volume",
    "hydrogen_bonding",
    "cross_stacking",
    "coaxial_stacking",
]


def get_energy_terms(base_dir: str, term: str) -> np.ndarray:
    energy_terms = np.loadtxt(base_dir + "/split_energy.dat", skiprows=1)
    return energy_terms[:, COLUMN_NAMES.index(term)]


def get_topology(base_dir: str, top_fname: str = "generated.top") -> jd_top.Topology:
    return jd_top.from_oxdna_file(base_dir + f"/{top_fname}")


def get_trajectory(base_dir: str, topology: jd_top.Topology) -> jd_traj.Trajectory:
    return jd_traj.from_file(
        base_dir + "/output.dat",
        topology.strand_counts,
        is_oxdna=False,
    )


def get_setup_data(base_dir: str, top_fname: str = "generated.top"):
    topology = get_topology(base_dir, top_fname)
    trajectory = get_trajectory(base_dir, topology)
    default_params = jd_toml.parse_toml("jax_dna/input/dna1/default_energy.toml")

    transform_fn = functools.partial(
        jd_energy.Nucleotide.from_rigid_body,
        com_to_backbone=default_params["geometry"]["com_to_backbone"],
        com_to_hb=default_params["geometry"]["com_to_hb"],
        com_to_stacking=default_params["geometry"]["com_to_stacking"],
    )

    displacement_fn, _ = jax_md.space.periodic(20.0)

    return (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    )


@pytest.mark.parametrize("base_dir", ["data/test-data/simple-helix"])
def test_hydrogen_bonding_discrete(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    sc = jd_sc.from_bps(16, np.array([[0, 15]]))
    energy_config = jd_energy.ExpectedHydrogenBondingConfiguration(
        **default_params["hydrogen_bonding"], sequence_constraints=sc
    )

    pseq = jd_sc.dseq_to_pseq(topology.seq, sc)

    terms = get_energy_terms(base_dir, "hydrogen_bonding")

    # compute energy terms
    energy_fn = jd_energy.ExpectedHydrogenBonding(displacement_fn=displacement_fn, params=energy_config.init_params())

    states = trajectory.state_rigid_body

    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            pseq,
            topology.bonded_neighbors,
            topology.unbonded_neighbors.T,
        )
    )(states)

    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-3)


def sequence_probability(
    sequence: str,
    unpaired: typ.Arr_Unpaired,
    bps: typ.Arr_Bp,
    unpaired_pseq: typ.Arr_Unpaired_Pseq,
    bp_pseq: typ.Arr_Bp_Pseq,
):
    # Initialize probability to 1 (neutral for multiplication)
    probability = 1.0

    for n_up_idx, up_idx in enumerate(unpaired):
        up_nt_idx = jd_const.DNA_ALPHA.index(sequence[up_idx])
        probability *= unpaired_pseq[n_up_idx, up_nt_idx]

    for bp_idx, (nt1, nt2) in enumerate(bps):
        bp_type_idx = jd_const.BP_TYPES.index(sequence[nt1] + sequence[nt2])
        probability *= bp_pseq[bp_idx, bp_type_idx]

    return probability


def test_hydrogen_bonding_brute_force():
    base_dir = "data/test-data/helix-4bp"
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir, "sys.top")

    sc = jd_sc.from_bps(8, np.array([[0, 7], [1, 6], [2, 5]]))

    rng = np.random.default_rng()
    ss_hb_weights = rng.random((4, 4))
    ss_hb_weights = ss_hb_weights / ss_hb_weights.sum(axis=1, keepdims=True)
    ss_hb_weights = jnp.array(ss_hb_weights)

    energy_config = jd_energy.ExpectedHydrogenBondingConfiguration(
        **default_params["hydrogen_bonding"], ss_hb_weights=ss_hb_weights, sequence_constraints=sc
    )

    energy_config_base = jd_energy.HydrogenBondingConfiguration(
        **default_params["hydrogen_bonding"],
        ss_hb_weights=ss_hb_weights,
    )

    bp_pseq = rng.random((sc.n_bp, 4))
    bp_pseq = bp_pseq / bp_pseq.sum(axis=1, keepdims=True)
    bp_pseq = jnp.array(bp_pseq)

    up_pseq = rng.random((sc.n_unpaired, 4))
    up_pseq = up_pseq / up_pseq.sum(axis=1, keepdims=True)
    up_pseq = jnp.array(up_pseq)

    pseq = (up_pseq, bp_pseq)

    # compute energy terms
    energy_fn = jd_energy.ExpectedHydrogenBonding(displacement_fn=displacement_fn, params=energy_config.init_params())

    energy_fn_base = jd_energy.HydrogenBonding(displacement_fn=displacement_fn, params=energy_config_base.init_params())

    states = trajectory.state_rigid_body

    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            pseq,
            topology.bonded_neighbors,
            topology.unbonded_neighbors.T,
        )
    )(states)

    @jax.jit
    def compute_base_vals(dseq):
        """Compute the per-state energies given a discrete sequence."""
        return jax.vmap(
            lambda s: energy_fn_base(
                transform_fn(s),
                dseq,
                topology.bonded_neighbors,
                topology.unbonded_neighbors.T,
            )
        )(states)

    # Brute force calculation
    assert len(jd_const.BP_TYPES) == len(jd_const.DNA_ALPHA)
    all_seq_idxs = itertools.product(np.arange(4), repeat=sc.n_unpaired + sc.n_bp)

    expected_energy_brute = 0.0
    for seq_idxs in all_seq_idxs:
        sampled_unpaired_seq_idxs = seq_idxs[: sc.n_unpaired]
        sampled_bp_type_idxs = seq_idxs[sc.n_unpaired :]

        seq = ["X"] * sc.n_nucleotides
        for unpaired_idx, nt_idx in zip(sc.unpaired, sampled_unpaired_seq_idxs, strict=False):
            seq[unpaired_idx] = jd_const.DNA_ALPHA[nt_idx]

        for (nt1_idx, nt2_idx), bp_type_idx in zip(sc.bps, sampled_bp_type_idxs, strict=False):
            bp1, bp2 = jd_const.BP_TYPES[bp_type_idx]
            seq[nt1_idx] = bp1
            seq[nt2_idx] = bp2
        dseq = jnp.array([jd_const.NUCLEOTIDES_IDX[s] for s in seq], dtype=jnp.int32)

        seq_energy_calc = compute_base_vals(dseq)

        seq_prob = sequence_probability(seq, sc.unpaired, sc.bps, up_pseq, bp_pseq)

        expected_energy_brute += seq_prob * seq_energy_calc

    np.testing.assert_allclose(energy, expected_energy_brute, atol=1e-4)


@pytest.mark.parametrize(
    ("params", "expected_error"),
    [
        (
            default_params,
            jd_config.ERR_MISSING_REQUIRED_PARAMS.format(props="sequence_constraints"),
        ),
    ],
)
def test_expected_stacking_config_raises_value_error(params: dict, expected_error: str):
    with pytest.raises(ValueError, match=expected_error):
        jd_energy.ExpectedStackingConfiguration(**(params["stacking"] | {"kt": 296.15 * 0.1 / 300.0}))


def test_stacking_brute_force():
    base_dir = "data/test-data/helix-4bp"
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir, "sys.top")

    sc = jd_sc.from_bps(8, np.array([[0, 7], [1, 6], [2, 5]]))

    rng = np.random.default_rng()
    ss_stack_weights = rng.random((4, 4))
    ss_stack_weights = ss_stack_weights / ss_stack_weights.sum(axis=1, keepdims=True)
    ss_stack_weights = jnp.array(ss_stack_weights)

    energy_config = jd_energy.ExpectedStackingConfiguration(
        **(default_params["stacking"] | {"kt": 296.15 * 0.1 / 300.0}),
        ss_stack_weights=ss_stack_weights,
        sequence_constraints=sc,
    )

    energy_config_base = jd_energy.StackingConfiguration(
        **(default_params["stacking"] | {"kt": 296.15 * 0.1 / 300.0}),
        ss_stack_weights=ss_stack_weights,
    )

    bp_pseq = rng.random((sc.n_bp, 4))
    bp_pseq = bp_pseq / bp_pseq.sum(axis=1, keepdims=True)
    bp_pseq = jnp.array(bp_pseq)

    up_pseq = rng.random((sc.n_unpaired, 4))
    up_pseq = up_pseq / up_pseq.sum(axis=1, keepdims=True)
    up_pseq = jnp.array(up_pseq)

    pseq = (up_pseq, bp_pseq)

    # compute energy terms
    energy_fn = jd_energy.ExpectedStacking(displacement_fn=displacement_fn, params=energy_config.init_params())

    energy_fn_base = jd_energy.Stacking(displacement_fn=displacement_fn, params=energy_config_base.init_params())

    states = trajectory.state_rigid_body

    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            pseq,
            topology.bonded_neighbors,
            topology.unbonded_neighbors.T,
        )
    )(states)

    @jax.jit
    def compute_base_vals(dseq):
        """Compute the per-state energies given a discrete sequence."""
        return jax.vmap(
            lambda s: energy_fn_base(
                transform_fn(s),
                dseq,
                topology.bonded_neighbors,
                topology.unbonded_neighbors.T,
            )
        )(states)

    # Brute force calculation
    assert len(jd_const.BP_TYPES) == len(jd_const.DNA_ALPHA)
    all_seq_idxs = itertools.product(np.arange(4), repeat=sc.n_unpaired + sc.n_bp)

    expected_energy_brute = 0.0
    for seq_idxs in all_seq_idxs:
        sampled_unpaired_seq_idxs = seq_idxs[: sc.n_unpaired]
        sampled_bp_type_idxs = seq_idxs[sc.n_unpaired :]

        seq = ["X"] * sc.n_nucleotides
        for unpaired_idx, nt_idx in zip(sc.unpaired, sampled_unpaired_seq_idxs, strict=False):
            seq[unpaired_idx] = jd_const.DNA_ALPHA[nt_idx]

        for (nt1_idx, nt2_idx), bp_type_idx in zip(sc.bps, sampled_bp_type_idxs, strict=False):
            bp1, bp2 = jd_const.BP_TYPES[bp_type_idx]
            seq[nt1_idx] = bp1
            seq[nt2_idx] = bp2
        dseq = jnp.array([jd_const.NUCLEOTIDES_IDX[s] for s in seq], dtype=jnp.int32)

        seq_energy_calc = compute_base_vals(dseq)

        seq_prob = sequence_probability(seq, sc.unpaired, sc.bps, up_pseq, bp_pseq)

        expected_energy_brute += seq_prob * seq_energy_calc

    np.testing.assert_allclose(energy, expected_energy_brute, atol=1e-4)
