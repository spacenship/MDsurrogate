"""Regression: residue labels are not sequence positions."""
from pathlib import Path
import numpy as np
import pytest
import torch
from force_md.data.residue_order import residue_order
from force_md.data.adapters.mdcath import MdCathConfig, MdCathDataset
from force_md.heavy.domain_topology import load_domain_topology


def test_stable_chain_aware_mapping():
    labels, first, mapping = residue_order([72, 860, 72, 73, 72], ['A','A','A','A','B'])
    assert labels.tolist() == [72,860,73,72]
    assert first.tolist() == [0,1,3,4]
    assert mapping.tolist() == [0,1,0,2,3]


def test_real_nonmonotonic_residue_order():
    root = Path(__file__).resolve().parents[1] / 'data_rotation_v2/chunk_0000'
    domain = '1fwxA01'
    if not (root / f'mdcath_dataset_{domain}.h5').exists():
        pytest.skip('regression shard not present')
    reader = MdCathDataset(MdCathConfig(data_dir=str(root), load_plm=False), domains=[domain], build_index=False)
    try:
        topo = reader.topology_for(domain)
        assert topo.atom_order is None
        g = reader._open(domain)[domain]
        first = residue_order(g['resid'][:], g['chain'][:])[1]
        assert topo.resid_original.tolist() == g['resid'][:][first].tolist()
        np.testing.assert_array_equal(reader.sequence_tokens_for(domain), topo.residue_type)
        ht = load_domain_topology(str(root), domain)
        np.testing.assert_array_equal(ht.residue_index_raw.numpy(), topo.atom_to_residue)
        for frame in (0,146):
            x, f, valid = reader.load_frame_arrays(domain, '320', '0', frame)
            rawx = g['320']['0']['coords'][frame].astype(np.float64)
            np.testing.assert_allclose(x, rawx - rawx.mean(0))
            np.testing.assert_array_equal(f, g['320']['0']['forces'][frame])
            ff, mask = reader.load_force_arrays(domain, '320', '0', frame)
            np.testing.assert_array_equal(ff, f[topo.represented])
            bonds = ht.topology.bonds.numpy()
            assert np.linalg.norm(x[bonds[0]]-x[bonds[1]], axis=1).max() < 2
    finally:
        reader.close()


def test_esm2_rejects_same_length_wrong_sequence(tmp_path):
    from force_md.conditioning.esm2 import Esm2EmbeddingCache, Esm2Config
    cache = Esm2EmbeddingCache(tmp_path)
    cache.save('example', 'ACD', torch.zeros(3, 2), Esm2Config())
    assert cache.load('example', expect_sequence='ACD').shape == (3, 2)
    with pytest.raises(ValueError, match='sequence/order mismatch'):
        cache.load('example', expect_sequence='ADC')


def test_sequence_change_changes_esmc_cache_key():
    from force_md.heavy_flow.esmc_encoder import _sequence_hash
    assert _sequence_hash(torch.tensor([0,1,2])) != _sequence_hash(torch.tensor([0,2,1]))
