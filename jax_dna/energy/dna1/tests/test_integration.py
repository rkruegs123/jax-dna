import functools

import jax
import jax.numpy as jnp
import jax_md
import numpy as np
import pytest

import jax_dna.energy.base as jd_energy_base
import jax_dna.energy.dna1 as jd_energy
import jax_dna.input.toml as jd_toml
import jax_dna.input.topology as jd_top
import jax_dna.input.trajectory as jd_traj
import jax_dna.utils.neighbors as jd_nb

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
]


def get_energy_terms(base_dir: str, term: str) -> np.ndarray:
    energy_terms = np.loadtxt(base_dir + "/split_energy.dat", skiprows=1)
    return energy_terms[:, COLUMN_NAMES.index(term)]

def get_potential_energy(base_dir: str) -> np.ndarray:
    # Columns are: time, potential_energy, kinetic_energy, total_energy
    energies = np.loadtxt(base_dir + "/energy.dat")
    potential_energies = energies[:, 1]
    return potential_energies[1:] # ignore the initial state


def get_topology(base_dir: str) -> jd_top.Topology:
    return jd_top.from_oxdna_file(base_dir + "/generated.top")


def get_trajectory(base_dir: str, topology: jd_top.Topology) -> jd_traj.Trajectory:
    return jd_traj.from_file(
        base_dir + "/output.dat",
        topology.strand_counts,
        is_oxdna=False,
    )


def get_setup_data(base_dir: str, box_size: float = 20.0):
    topology = get_topology(base_dir)
    trajectory = get_trajectory(base_dir, topology)
    default_params = jd_toml.parse_toml("jax_dna/input/dna1/default_energy.toml")

    transform_fn = functools.partial(
        jd_energy.Nucleotide.from_rigid_body,
        com_to_backbone=default_params["geometry"]["com_to_backbone"],
        com_to_hb=default_params["geometry"]["com_to_hb"],
        com_to_stacking=default_params["geometry"]["com_to_stacking"],
    )

    displacement_fn, _ = jax_md.space.periodic(box_size)

    return (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    )


@pytest.mark.parametrize("base_dir", ["data/test-data/dna1/simple-helix"])
def test_bonded_excluded_volume(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "bonded_excluded_volume")
    # compute energy terms
    energy_config = jd_energy.BondedExcludedVolumeConfiguration(**default_params["bonded_excluded_volume"])
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
    "base_dir",
    [
        "data/test-data/dna1/simple-helix",
        "data/test-data/dna1/simple-coax",
    ],
)
def test_coaxial_stacking(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "coaxial_stacking")
    # compute energy terms
    energy_config = jd_energy.CoaxialStackingConfiguration(**default_params["coaxial_stacking"])
    energy_fn = jd_energy.CoaxialStacking(displacement_fn=displacement_fn, params=energy_config.init_params())

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


@pytest.mark.parametrize("base_dir", ["data/test-data/dna1/simple-helix"])
def test_cross_stacking(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "cross_stacking")

    default_params = jax.tree_util.tree_map(lambda arr: jnp.array(arr, dtype=jnp.float64), default_params)
    energy_config = jd_energy.CrossStackingConfiguration(**default_params["cross_stacking"])
    energy_fn = jd_energy.CrossStacking(displacement_fn=displacement_fn, params=energy_config.init_params())

    seq = jnp.concat([topology.seq[:8][::-1], topology.seq[8:][::-1]])
    bonded_neighbors = topology.bonded_neighbors
    unbonded_neighbors = topology.unbonded_neighbors

    states = trajectory.state_rigid_body
    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            seq,
            bonded_neighbors,
            unbonded_neighbors.T,
        )
    )(states)
    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-3)


@pytest.mark.parametrize("base_dir", ["data/test-data/dna1/simple-helix"])
def test_fene(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "fene")
    # compute energy terms
    energy_config = jd_energy.FeneConfiguration(**default_params["fene"])
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


@pytest.mark.parametrize("base_dir", ["data/test-data/dna1/simple-helix"])
def test_hydrogen_bonding(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "hydrogen_bonding")
    # compute energy terms
    energy_config = jd_energy.HydrogenBondingConfiguration(**default_params["hydrogen_bonding"])
    energy_fn = jd_energy.HydrogenBonding(displacement_fn=displacement_fn, params=energy_config.init_params())

    states = trajectory.state_rigid_body

    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            jnp.array(topology.seq),
            topology.bonded_neighbors,
            topology.unbonded_neighbors.T,
        )
    )(states)

    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-3)


# mismatch 1/100
@pytest.mark.parametrize("base_dir", ["data/test-data/dna1/simple-helix"])
def test_stacking(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "stacking")
    # compute energy terms
    energy_config = jd_energy.StackingConfiguration(**(default_params["stacking"] | {"kt": 296.15 * 0.1 / 300.0}))
    energy_fn = jd_energy.Stacking(displacement_fn=displacement_fn, params=(energy_config).init_params())
    seq = jnp.concat([topology.seq[:8][::-1], topology.seq[8:][::-1]])

    states = trajectory.state_rigid_body
    energy = jax.vmap(
        lambda s: energy_fn(
            transform_fn(s),
            seq,
            topology.bonded_neighbors,
            topology.unbonded_neighbors.T,
        )
    )(states)

    energy = np.around(energy / topology.n_nucleotides, 6)

    np.testing.assert_allclose(energy, terms, atol=1e-6)


@pytest.mark.parametrize("base_dir", ["data/test-data/dna1/simple-helix"])
def test_unbonded_excluded_volume(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "unbonded_excluded_volume")
    # compute energy terms
    energy_config = jd_energy.UnbondedExcludedVolumeConfiguration(**default_params["unbonded_excluded_volume"])
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
    (
        "base_dir",
        "t_kelvin",
        "use_neighbors"
    ),
    [
        ("data/test-data/simple-helix", 296.15, False),
        ("data/test-data/simple-helix", 296.15, True),
    ]
)
def test_total_energy(base_dir: str, t_kelvin: float, *, use_neighbors: bool):
    box_size = 20.0
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir, box_size=box_size)
    states = trajectory.state_rigid_body

    terms = get_potential_energy(base_dir)


    kt = t_kelvin * 0.1 / 300.0

    configs = [
        jd_energy.FeneConfiguration.from_dict(default_params["fene"], ("eps_backbone",)),
        jd_energy.BondedExcludedVolumeConfiguration.from_dict(
            default_params["bonded_excluded_volume"],
            ("eps_exc","dr_star_base")
        ),
        jd_energy.StackingConfiguration.from_dict(default_params["stacking"] | {"kt": kt}, ("*",)),
        jd_energy.UnbondedExcludedVolumeConfiguration.from_dict(default_params["unbonded_excluded_volume"]),
        jd_energy.HydrogenBondingConfiguration.from_dict(default_params["hydrogen_bonding"]),
        jd_energy.CrossStackingConfiguration.from_dict(default_params["cross_stacking"]),
        jd_energy.CoaxialStackingConfiguration.from_dict(default_params["coaxial_stacking"]),
    ]

    energy_fns = [
        jd_energy.Fene,
        jd_energy.BondedExcludedVolume,
        jd_energy.Stacking,
        jd_energy.UnbondedExcludedVolume,
        jd_energy.HydrogenBonding,
        jd_energy.CrossStacking,
        jd_energy.CoaxialStacking,
    ]

    transformed_fns = [
        energy_fn(
            displacement_fn=displacement_fn,
            params=config.init_params(),
        )
        for config, energy_fn in zip(configs, energy_fns, strict=True)
    ]

    transform_fn = functools.partial(
        jd_energy.Nucleotide.from_rigid_body,
        com_to_backbone=default_params["geometry"]["com_to_backbone"],
        com_to_hb=default_params["geometry"]["com_to_hb"],
        com_to_stacking=default_params["geometry"]["com_to_stacking"],
    )

    energy_fn = jd_energy_base.ComposedEnergyFunction(
        energy_fns=transformed_fns,
        rigid_body_transform_fn=transform_fn,
    )

    if use_neighbors:

        @jax.jit
        def state_energy_fn(state, neighbors):
            neighbors = neighbors.update(state.center)
            neighbors_idx = neighbors.idx
            return energy_fn(
                transform_fn(state),
                topology.seq,
                topology.bonded_neighbors,
                neighbors_idx
            )

        neighbor_fn = jd_nb.get_neighbor_list_fn(
            topology.bonded_neighbors,
            topology.n_nucleotides,
            displacement_fn,
            box_size
        )
        neighbor_fn = jax.jit(neighbor_fn)
        neighbors = neighbor_fn.allocate(states[0].center) # We use the COMs to allocate neighbors

        energies = jax.vmap(state_energy_fn, (0, None))(states, neighbors)

    else:
        energies = jax.vmap(
            lambda s: energy_fn(
                transform_fn(s),
                topology.seq,
                topology.bonded_neighbors,
                topology.unbonded_neighbors.T,
            )
        )(states)

    energies = np.around(energies / topology.n_nucleotides, 6)

    np.testing.assert_allclose(energies, terms, rtol=1e-5, atol=1e-6)

if __name__ == "__main__":
    pytest.main()
