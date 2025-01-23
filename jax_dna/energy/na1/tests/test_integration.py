import functools
import itertools

import jax
import jax.numpy as jnp
import jax_md
import numpy as np
import pytest

import jax_dna.energy.na1 as jd_energy
import jax_dna.input.toml as jd_toml
import jax_dna.input.topology as jd_top
import jax_dna.input.trajectory as jd_traj
import jax_dna.utils.helpers as jd_help

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - ignore boolean positional value
# this is a common jax practice


COLUMN_NAMES = [
    "t",
    "fene",
    "bonded_excluded_volume",
    "stacking",
    "unbonded_excluded_volume",
    "hydrogen_bonding",
    "cross_stacking",
    "coaxial_stacking",
    "debye"
]

def flip_state(state, strand_bounds):
    # TODO (rkruegs123): use `is_oxdna=True` in trajectory reader instead of flipping states
    # https://github.com/ssec-jhu/jax-dna/issues/23
    return jd_help.tree_concatenate([state[s:e][::-1] for s, e in strand_bounds])

def get_energy_terms(base_dir: str, term: str) -> np.ndarray:
    energy_terms = np.loadtxt(base_dir + "/split_energy.dat", skiprows=1)
    return energy_terms[:, COLUMN_NAMES.index(term)]


def get_topology(base_dir: str) -> jd_top.Topology:
    return jd_top.from_oxdna_file(base_dir + "/generated.top")


def get_trajectory(base_dir: str, topology: jd_top.Topology) -> jd_traj.Trajectory:
    return jd_traj.from_file(
        base_dir + "/output.dat",
        topology.strand_counts,
        is_oxdna=False,
    )


def add_prefix_to_leaf_keys(data, prefix):
    """
    Recursively add a prefix to all leaf keys in a nested dictionary.

    Args:
        data (dict): The nested dictionary representing the TOML content.
        prefix (str): The prefix to add to the leaf keys.

    Returns:
        dict: A new dictionary with updated keys.
    """
    if isinstance(data, dict):
        # Traverse the dictionary and process each key and value
        return {
            (prefix + key if not isinstance(value, dict | list) else key): add_prefix_to_leaf_keys(value, prefix)
            for key, value in data.items()
        }

    if isinstance(data, list):
        # Process each item in the list
        return [add_prefix_to_leaf_keys(item, prefix) for item in data]

    # Return data as-is (not a dict or list)
    return data



def merge_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries. If a key exists in both:
    - If the values are dictionaries, merge them recursively.
    - Otherwise, the value from dict2 overwrites dict1.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        dict: A new dictionary with merged contents.
    """
    merged = dict1.copy()  # Start with a copy of dict1
    for key, value in dict2.items():
        if key in merged:
            if isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                merged[key] = merge_dicts(merged[key], value)
            else:
                # Overwrite or combine values if not dictionaries
                merged[key] = value
        else:
            # Key is only in dict2
            merged[key] = value
    return merged

def get_setup_data(base_dir: str):
    topology = get_topology(base_dir)
    trajectory = get_trajectory(base_dir, topology)
    default_rna2_params = jd_toml.parse_toml("jax_dna/input/rna2/default_energy.toml")
    default_dna2_params = jd_toml.parse_toml("jax_dna/input/dna2/default_energy.toml")
    default_na1_params = jd_toml.parse_toml("jax_dna/input/na1/default_energy.toml")

    transform_fn = functools.partial(
        jd_energy.HybridNucleotide.from_rigid_body,
        # DNA2-specific
        dna_com_to_backbone_x=default_dna2_params["geometry"]["com_to_backbone_x"],
        dna_com_to_backbone_y=default_dna2_params["geometry"]["com_to_backbone_y"],
        dna_com_to_backbone_dna1=default_dna2_params["geometry"]["com_to_backbone_dna1"],
        dna_com_to_hb=default_dna2_params["geometry"]["com_to_hb"],
        dna_com_to_stacking=default_dna2_params["geometry"]["com_to_stacking"],
        # RNA2-specific
        rna_com_to_backbone_x=default_rna2_params["geometry"]["pos_back_a1"],
        rna_com_to_backbone_y=default_rna2_params["geometry"]["pos_back_a3"],
        rna_com_to_hb=default_rna2_params["geometry"]["pos_base"],
        rna_com_to_stacking=default_rna2_params["geometry"]["pos_stack"],
        rna_p3_x=default_rna2_params["geometry"]["p3_x"],
        rna_p3_y=default_rna2_params["geometry"]["p3_y"],
        rna_p3_z=default_rna2_params["geometry"]["p3_z"],
        rna_p5_x=default_rna2_params["geometry"]["p5_x"],
        rna_p5_y=default_rna2_params["geometry"]["p5_y"],
        rna_p5_z=default_rna2_params["geometry"]["p5_z"],
        rna_pos_stack_3_a1=default_rna2_params["geometry"]["pos_stack_3_a1"],
        rna_pos_stack_3_a2=default_rna2_params["geometry"]["pos_stack_3_a2"],
        rna_pos_stack_5_a1=default_rna2_params["geometry"]["pos_stack_5_a1"],
        rna_pos_stack_5_a2=default_rna2_params["geometry"]["pos_stack_5_a2"],
    )

    displacement_fn, _ = jax_md.space.periodic(20.0)

    default_rna2_params = add_prefix_to_leaf_keys(default_rna2_params, "rna_")
    default_dna2_params = add_prefix_to_leaf_keys(default_dna2_params, "dna_")
    default_na1_params = add_prefix_to_leaf_keys(default_na1_params, "drh_")
    merged_params = merge_dicts(default_rna2_params, default_dna2_params)
    merged_params = merge_dicts(merged_params, default_na1_params)

    return (
        topology,
        trajectory,
        merged_params,
        transform_fn,
        displacement_fn,
    )



@pytest.mark.parametrize(
    "base_dir",
    [
        "data/test-data/na1/simple-helix-dna-dna",
        "data/test-data/na1/simple-helix-rna-rna",
        "data/test-data/na1/simple-helix-dna-rna",
    ]
)
def test_fene(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    strand_counts = topology.strand_counts
    strand_bounds = list(itertools.pairwise([0, *itertools.accumulate(strand_counts)]))
    strand_bounds = jnp.array(strand_bounds)

    terms = get_energy_terms(base_dir, "fene")
    # compute energy terms
    energy_config = jd_energy.FeneConfiguration(**(default_params["fene"] | {"nt_type": topology.nt_type}))
    energy_fn = jd_energy.Fene(displacement_fn=displacement_fn, params=energy_config.init_params())

    states = trajectory.state_rigid_body

    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            topology.seq,
            topology.bonded_neighbors,
            topology.unbonded_neighbors.T,
        )
    )(states)

    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-6)



@pytest.mark.parametrize(
    "base_dir",
    [
        "data/test-data/na1/simple-helix-dna-dna",
        "data/test-data/na1/simple-helix-rna-rna",
        "data/test-data/na1/simple-helix-dna-rna",
    ]
)
def test_bonded_excluded_volume(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    strand_counts = topology.strand_counts
    strand_bounds = list(itertools.pairwise([0, *itertools.accumulate(strand_counts)]))
    strand_bounds = jnp.array(strand_bounds)

    terms = get_energy_terms(base_dir, "bonded_excluded_volume")
    # compute energy terms
    energy_config = jd_energy.BondedExcludedVolumeConfiguration(
        **(default_params["bonded_excluded_volume"] | {"nt_type": topology.nt_type})
    )
    energy_fn = jd_energy.BondedExcludedVolume(displacement_fn=displacement_fn, params=energy_config.init_params())

    states = trajectory.state_rigid_body

    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            topology.seq,
            topology.bonded_neighbors,
            topology.unbonded_neighbors.T,
        )
    )(states)

    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-6)


@pytest.mark.parametrize(
    (
        "base_dir",
        "t_kelvin"
    ),
    [
        ("data/test-data/na1/simple-helix-dna-dna", 296.15),
        ("data/test-data/na1/simple-helix-dna-rna", 296.15),
        ("data/test-data/na1/simple-helix-rna-rna", 296.15),
    ]
)
def test_stacking(base_dir: str, t_kelvin: float):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    strand_counts = topology.strand_counts
    strand_bounds = list(itertools.pairwise([0, *itertools.accumulate(strand_counts)]))
    strand_bounds = jnp.array(strand_bounds)

    seq = jnp.concat([topology.seq[:8][::-1], topology.seq[8:][::-1]])
    nt_type = jnp.concat([topology.nt_type[:8][::-1], topology.nt_type[8:][::-1]])

    terms = get_energy_terms(base_dir, "stacking")
    # compute energy terms
    energy_config = jd_energy.StackingConfiguration(
        **(default_params["stacking"] | {"kt": t_kelvin * 0.1 / 300.0, "nt_type": nt_type})
    )
    energy_fn = jd_energy.Stacking(displacement_fn=displacement_fn, params=(energy_config).init_params())

    states = trajectory.state_rigid_body

    rev_states = jax.vmap(flip_state, (0, None))(states, strand_bounds)

    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            seq,
            topology.bonded_neighbors,
            topology.unbonded_neighbors.T,
        )
    )(rev_states)

    energy = np.around(energy / topology.n_nucleotides, 6)

    np.testing.assert_allclose(energy, terms, atol=1e-3) # using a higher tolerance here



@pytest.mark.parametrize(
    "base_dir",
    [
        "data/test-data/na1/simple-helix-dna-dna",
        "data/test-data/na1/simple-helix-rna-rna",
        "data/test-data/na1/simple-helix-dna-rna",
        "data/test-data/na1/simple-helix-rna-dna",
    ]
)
def test_unbonded_excluded_volume(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    strand_counts = topology.strand_counts
    strand_bounds = list(itertools.pairwise([0, *itertools.accumulate(strand_counts)]))
    strand_bounds = jnp.array(strand_bounds)

    terms = get_energy_terms(base_dir, "unbonded_excluded_volume")
    # compute energy terms
    energy_config = jd_energy.UnbondedExcludedVolumeConfiguration(
        **(default_params["unbonded_excluded_volume"] | {"nt_type": topology.nt_type})
    )
    energy_fn = jd_energy.UnbondedExcludedVolume(displacement_fn=displacement_fn, params=energy_config.init_params())

    states = trajectory.state_rigid_body

    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            topology.seq,
            topology.bonded_neighbors,
            topology.unbonded_neighbors.T,
        )
    )(states)

    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-6)



@pytest.mark.parametrize(
    "base_dir",
    [
        "data/test-data/na1/simple-helix-dna-dna",
        "data/test-data/na1/simple-helix-rna-rna",
        "data/test-data/na1/simple-helix-dna-rna",
        "data/test-data/na1/simple-helix-rna-dna",
    ]
)
def test_cross_stacking(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    strand_counts = topology.strand_counts
    strand_bounds = list(itertools.pairwise([0, *itertools.accumulate(strand_counts)]))
    strand_bounds = jnp.array(strand_bounds)

    terms = get_energy_terms(base_dir, "cross_stacking")
    # compute energy terms
    energy_config = jd_energy.CrossStackingConfiguration(
        **(default_params["cross_stacking"] | {"nt_type": topology.nt_type})
    )
    energy_fn = jd_energy.CrossStacking(displacement_fn=displacement_fn, params=energy_config.init_params())

    states = trajectory.state_rigid_body

    rev_states = jax.vmap(flip_state, (0, None))(states, strand_bounds)

    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            topology.seq,
            topology.bonded_neighbors,
            topology.unbonded_neighbors.T,
        )
    )(rev_states)

    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-4) # using a higher tolerance here


@pytest.mark.parametrize(
    "base_dir",
    [
        "data/test-data/na1/simple-helix-dna-dna",
        "data/test-data/na1/simple-helix-rna-rna",
        "data/test-data/na1/simple-helix-dna-rna",
        "data/test-data/na1/simple-helix-rna-dna",
    ]
)
def test_hydrogen_bonding(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    strand_counts = topology.strand_counts
    strand_bounds = list(itertools.pairwise([0, *itertools.accumulate(strand_counts)]))
    strand_bounds = jnp.array(strand_bounds)

    terms = get_energy_terms(base_dir, "hydrogen_bonding")
    # compute energy terms
    energy_config = jd_energy.HydrogenBondingConfiguration(
        **(default_params["hydrogen_bonding"] | {"nt_type": topology.nt_type})
    )
    energy_fn = jd_energy.HydrogenBonding(displacement_fn=displacement_fn, params=energy_config.init_params())

    states = trajectory.state_rigid_body

    rev_states = jax.vmap(flip_state, (0, None))(states, strand_bounds)

    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            topology.seq,
            topology.bonded_neighbors,
            topology.unbonded_neighbors.T,
        )
    )(rev_states)

    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-4) # using a higher tolerance here




@pytest.mark.parametrize(
    "base_dir",
    [
        "data/test-data/na1/simple-coax-dna-dna-dna",
        "data/test-data/na1/simple-coax-rna-rna-rna",
        # TODO (rkruegs123): there's a bug in oxNA standalone code! spring constant for DRH is read as 0.0
        # https://github.com/ssec-jhu/jax-dna/issues/22
        # "data/test-data/na1/simple-coax-dna-dna-rna",
    ]
)
def test_coaxial_stacking(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    strand_counts = topology.strand_counts
    strand_bounds = list(itertools.pairwise([0, *itertools.accumulate(strand_counts)]))
    strand_bounds = jnp.array(strand_bounds)

    terms = get_energy_terms(base_dir, "coaxial_stacking")
    # compute energy terms
    energy_config = jd_energy.CoaxialStackingConfiguration(
        **(default_params["coaxial_stacking"] | {"nt_type": topology.nt_type})
    )
    energy_fn = jd_energy.CoaxialStacking(displacement_fn=displacement_fn, params=energy_config.init_params())

    states = trajectory.state_rigid_body

    rev_states = jax.vmap(flip_state, (0, None))(states, strand_bounds)

    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            topology.seq,
            topology.bonded_neighbors,
            topology.unbonded_neighbors.T,
        )
    )(rev_states)

    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-6)



@pytest.mark.parametrize(
    (
        "base_dir",
        "t_kelvin",
        "salt_conc",
        "half_charged_ends",
    ),
    [
        ("data/test-data/na1/simple-helix-dna-dna", 296.15, 0.5, False),
        ("data/test-data/na1/simple-helix-rna-rna", 296.15, 0.5, False),
        ("data/test-data/na1/simple-helix-dna-rna", 296.15, 0.5, False),
        ("data/test-data/na1/simple-helix-rna-dna", 296.15, 0.5, False),
    ]
)
def test_debye(base_dir: str, t_kelvin: float, salt_conc: float, *, half_charged_ends: bool):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    strand_counts = topology.strand_counts
    strand_bounds = list(itertools.pairwise([0, *itertools.accumulate(strand_counts)]))
    strand_bounds = jnp.array(strand_bounds)

    terms = get_energy_terms(base_dir, "debye")
    # compute energy terms
    kt = t_kelvin * 0.1 / 300.0
    energy_config = jd_energy.DebyeConfiguration(
        **(
            default_params["debye"] | \
            {
                "kt": kt,
                "salt_conc": salt_conc,
                "is_end": topology.is_end,
                "half_charged_ends": half_charged_ends,
                "nt_type": topology.nt_type
            }
        )
    )
    energy_fn = jd_energy.Debye(displacement_fn=displacement_fn, params=energy_config.init_params())

    states = trajectory.state_rigid_body

    rev_states = jax.vmap(flip_state, (0, None))(states, strand_bounds)

    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            topology.seq,
            topology.bonded_neighbors,
            topology.unbonded_neighbors.T,
        )
    )(rev_states)

    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-5)



if __name__ == "__main__":
    pytest.main()
