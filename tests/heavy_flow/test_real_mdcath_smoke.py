from __future__ import annotations

from pathlib import Path

import pytest

from force_md.data.adapters.mdcath import MdCathConfig, MdCathDataset
from force_md.heavy.domain_topology import load_domain_topology
from force_md.heavy_flow import (
    ESMCConfig,
    HeavyFlowAtomEncoder,
    HeavyFlowGeometryConfig,
    HeavyFlowGraphConfig,
    bonds_from_topology,
    from_mdcath_example,
)


@pytest.mark.mdcath
def test_one_real_mdcath_heavy_batch_forward_backward():
    shards = sorted(Path("data").glob("mdcath_dataset_*.h5"))
    if not shards:
        pytest.skip("no mdCATH shard is present")
    domain = shards[0].stem[len("mdcath_dataset_"):]
    dataset = MdCathDataset(
        MdCathConfig(
            data_dir="data", temperatures=(320,), replicas=(0,),
            frames_per_trajectory=1, max_domains=1, allow_fake_plm=True,
        ),
        domains=[domain],
    )
    example = dataset[0]
    topology = load_domain_topology("data", domain, represented_scope="heavy_atom")
    bonds, bond_type = bonds_from_topology(topology, topology.raw_to_batch)
    sample = from_mdcath_example(
        example, bond_index=bonds, bond_type=bond_type, allow_identity_future=True,
    )
    # Bounded smoke configuration: real ordering/masks/direct heavy forces/PSF
    # edges are exercised without starting a training run.
    encoder = HeavyFlowAtomEncoder(
        geometry_config=HeavyFlowGeometryConfig(
            num_blocks=1, hidden_irreps="4x0e + 1x1o + 1x1e + 1x2e", radial_basis=4,
        ),
        graph_config=HeavyFlowGraphConfig(max_spatial_neighbors=1),
        esmc_config=ESMCConfig(backend="stub", stub_dim=4, projected_dim=2),
    )
    output = encoder(sample.condition)
    output.atom_features.square().mean().backward()
    assert output.atom_features.shape[0] == int(sample.condition.atom_mask.sum())
    assert output.graph.stats.num_covalent_edges > 0
    assert sample.targets.force_current.shape == (1, output.graph.num_nodes, 3)
