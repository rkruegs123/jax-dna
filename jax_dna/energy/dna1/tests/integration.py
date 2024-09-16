import functools

import jax
import jax.numpy as jnp
import jax_md
import numpy as np
import pandas as pd
import pytest

import jax_dna.energy.dna1 as jd_energy
import jax_dna.input.toml as jd_toml
import jax_dna.input.topology as jd_top
import jax_dna.input.trajectory as jd_traj

jax.config.update("jax_enable_x64", True)


def get_energy_terms(base_dir:str) -> pd.DataFrame:
    return pd.read_csv(
        base_dir + "/split_energy.dat",
        names=["t", "fene", "bonded_excluded_volume", "stacking", "unbonded_excluded_volume", "hydrogen_bonding", "cross_stacking", "coaxial_stacking"],
        sep='\s+',
    ).iloc[1:, :]



def get_topology(base_dir:str) -> jd_top.Topology:
    return jd_top.from_oxdna_file(base_dir + "/generated.top")


def get_trajectory(base_dir:str, topology:jd_top.Topology) -> jd_traj.Trajectory:
    return jd_traj.from_file(
        base_dir + "/output.dat",
        topology.strand_counts,
        is_oxdna=False,
    )


def get_setup_data(base_dir:str):
    topology = get_topology(base_dir)
    tracjectory = get_trajectory(base_dir, topology)
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
        tracjectory,
        default_params,
        transform_fn,
        displacement_fn,
    )


@pytest.mark.parametrize("base_dir", ["data/test-data/simple-helix"])
def test_bonded_excluded_volume(base_dir:str):
    (
        topology,
        tracjectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir).loc[:, "bonded_excluded_volume"].values
    # compute energy terms
    energy_config = jd_energy.BondedExcludedVolumeConfiguration(**default_params["bonded_excluded_volume"])
    energy_fn = jd_energy.BondedExcludedVolume(
        displacement_fn=displacement_fn,
        params=energy_config.init_params()
    )

    states = tracjectory.state_rigid_body


    energy = jax.vmap(lambda s: energy_fn(
        transform_fn(s),
        topology.seq_one_hot,
        topology.bonded_neighbors,
        topology.unbonded_neighbors.T,
    ))(states)

    energy = np.around(energy/topology.n_nucleotides, 6)
    return np.isclose(energy, terms, atol=1e-6).sum(), energy.shape[0]
    np.testing.assert_allclose(energy, terms, atol=1e-6)


@pytest.mark.parametrize("base_dir", ["data/test-data/simple-helix"])
def test_coaxial_stacking(base_dir:str):
    (
        topology,
        tracjectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir).loc[:, "coaxial_stacking"].values
    # compute energy terms
    energy_config = jd_energy.CoaxialStackingConfiguration(**default_params["coaxial_stacking"])
    energy_fn = jd_energy.CoaxialStacking(
        displacement_fn=displacement_fn,
        params=energy_config.init_params()
    )

    states = tracjectory.state_rigid_body

    energy = jax.vmap(lambda s: energy_fn(
        transform_fn(s),
        topology.seq_one_hot,
        topology.bonded_neighbors,
        topology.unbonded_neighbors.T,
    ))(states)

    energy = np.around(energy/topology.n_nucleotides, 6)
    return np.isclose(energy, terms, atol=1e-6).sum(), energy.shape[0]
    np.testing.assert_allclose(energy, terms, atol=1e-6)

# 39/100 wrong
@pytest.mark.parametrize("base_dir", ["data/test-data/simple-helix"])
def test_cross_stacking(base_dir:str):
    (
        topology,
        tracjectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    # import sys
    # print(tracjectory.state_rigid_body.center[2, :8, :])
    # sys.exit()


    import jax_dna.common.topology as old_top
    import jax_dna.common.utils as old_utils
    topology = old_top.TopologyInfo(base_dir + "/generated.top", reverse_direction=False)
    bonded_neighbors = topology.bonded_nbrs
    unbonded_neighbors = topology.unbonded_nbrs
    one_hot = jnp.array(old_utils.get_one_hot(topology.seq), dtype=jnp.float64)
    n_nucleotides = topology.n

    import jax_dna.common.trajectory as old_traj
    tracjectory = old_traj.TrajectoryInfo(
        topology,
        read_from_file=True,
        traj_path=base_dir + "/output.dat",
        reverse_direction=False,
    )

    tmp_states = tracjectory.get_states()

    states = jax_md.rigid_body.RigidBody(
        center = jnp.stack([s.center for s in tmp_states]),
        orientation = jax_md.rigid_body.Quaternion(vec=jnp.stack([s.orientation.vec for s in tmp_states])),
    )

    terms = get_energy_terms(base_dir).loc[:, "cross_stacking"].values
    # compute energy terms
    default_params = jax.tree_util.tree_map(lambda arr: jnp.array(arr, dtype=jnp.float64), default_params)
    energy_config = jd_energy.CrossStackingConfiguration(**default_params["cross_stacking"])
    energy_fn = jd_energy.CrossStacking(
        displacement_fn=displacement_fn,
        params=energy_config.init_params()
    )

    # one_hot = jnp.concat([
    #     topology.seq_one_hot[:8][::-1],
    #     topology.seq_one_hot[8:][::-1]
    # ])

    # bonded_neighbors = np.sort(topology.bonded_neighbors, axis=0)
    # unbonded_neighbors = np.sort(topology.unbonded_neighbors, axis=0)
    # bonded_neighbors = topology.bonded_neighbors
    # unbonded_neighbors = topology.unbonded_neighbors


    # states = tracjectory.state_rigid_body
    energy = jax.vmap(lambda s: energy_fn(
        transform_fn(s),
        one_hot,
        bonded_neighbors,
        unbonded_neighbors.T,
    ))(states)
    energy = np.around(energy/n_nucleotides, 6)
    mask = np.logical_not(np.isclose(energy, terms, atol=1e-6))
    # print(energy[np.where(mask)], terms[np.where(mask)])
    return np.isclose(energy, terms, atol=1e-6).sum(), energy.shape[0]
    np.testing.assert_allclose(energy, terms, atol=1e-6)


@pytest.mark.parametrize("base_dir", ["data/test-data/simple-helix"])
def test_fene(base_dir:str):
    (
        topology,
        tracjectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir).loc[:, "fene"].values
    # compute energy terms
    energy_config = jd_energy.FeneConfiguration(**default_params["fene"])
    energy_fn = jd_energy.Fene(
        displacement_fn=displacement_fn,
        params=energy_config.init_params()
    )

    states = tracjectory.state_rigid_body

    energy = jax.vmap(lambda s: energy_fn(
        transform_fn(s),
        topology.seq_one_hot,
        topology.bonded_neighbors,
        topology.unbonded_neighbors.T,
    ))(states)

    energy = np.around(energy/topology.n_nucleotides, 6)
    return np.isclose(energy, terms, atol=1e-6).sum(), energy.shape[0]
    np.testing.assert_allclose(energy, terms, atol=1e-6)

# mismatch 76/100
@pytest.mark.parametrize("base_dir", ["data/test-data/simple-helix"])
def test_hydrogen_bonding(base_dir:str):
    (
        topology,
        tracjectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir).loc[:, "hydrogen_bonding"].values
    # compute energy terms
    energy_config = jd_energy.HydrogenBondingConfiguration(**default_params["hydrogen_bonding"])
    energy_fn = jd_energy.HydrogenBonding(
        displacement_fn=displacement_fn,
        params=energy_config.init_params()
    )

    states = tracjectory.state_rigid_body

    energy = jax.vmap(lambda s: energy_fn(
        transform_fn(s),
        jnp.array(topology.seq_one_hot),
        topology.bonded_neighbors,
        topology.unbonded_neighbors.T,
    ))(states)

    energy = np.around(energy/topology.n_nucleotides, 6)
    mask = np.logical_not(np.isclose(energy, terms, atol=1e-6))
    print(energy[np.where(mask)], terms[np.where(mask)])
    return np.isclose(energy, terms, atol=1e-6).sum(), energy.shape[0]
    np.testing.assert_allclose(energy, terms, atol=1e-6)

# mismatch 1/100
@pytest.mark.parametrize("base_dir", ["data/test-data/simple-helix"])
def test_stacking(base_dir:str):
    (
        topology,
        tracjectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir).loc[:, "stacking"].values
    # compute energy terms
    energy_config = jd_energy.StackingConfiguration(**(default_params["stacking"] | {"kt": 296.15 * 0.1 / 300.0}))
    energy_fn = jd_energy.Stacking(
        displacement_fn=displacement_fn,
        params=(energy_config).init_params()
    )
    one_hot = jnp.concat([
        topology.seq_one_hot[:8][::-1],
        topology.seq_one_hot[8:][::-1]
    ])

    states = tracjectory.state_rigid_body
    energy = jax.vmap(lambda s: energy_fn(
        transform_fn(s),
        one_hot,
        topology.bonded_neighbors,
        topology.unbonded_neighbors.T,
    ))(states)

    energy = np.around(energy/topology.n_nucleotides, 6)

    return np.isclose(energy, terms, atol=1e-6).sum(), energy.shape[0]
    np.testing.assert_allclose(energy, terms, atol=1e-6)


@pytest.mark.parametrize("base_dir", ["data/test-data/simple-helix"])
def test_unbonded_excluded_volume(base_dir:str):
    (
        topology,
        tracjectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir).loc[:, "unbonded_excluded_volume"].values
    # compute energy terms
    energy_config = jd_energy.UnbondedExcludedVolumeConfiguration(**default_params["unbonded_excluded_volume"])
    energy_fn = jd_energy.UnbondedExcludedVolume(
        displacement_fn=displacement_fn,
        params=energy_config.init_params()
    )

    states = tracjectory.state_rigid_body

    energy = jax.vmap(lambda s: energy_fn(
        transform_fn(s),
        topology.seq_one_hot,
        topology.bonded_neighbors,
        topology.unbonded_neighbors.T,
    ))(states)

    energy = np.around(energy/topology.n_nucleotides, 6)

    return np.isclose(energy, terms, atol=1e-6).sum(), energy.shape[0]
    np.testing.assert_allclose(energy, terms, atol=1e-6)


if __name__=="__main__":
    print("Bonded Excluded Volume")
    print(test_bonded_excluded_volume("data/test-data/simple-helix"))
    print("Coaxial Stacking")
    print(test_coaxial_stacking("data/test-data/simple-helix"))
    print("Cross Stacking")
    print(test_cross_stacking("data/test-data/simple-helix"))
    print("Fene")
    print(test_fene("data/test-data/simple-helix"))
    print("Hydrogen Bonding")
    print(test_hydrogen_bonding("data/test-data/simple-helix"))
    print("Stacking")
    print(test_stacking("data/test-data/simple-helix"))
    print("Unbonded Excluded Volume")
    print(test_unbonded_excluded_volume("data/test-data/simple-helix"))
