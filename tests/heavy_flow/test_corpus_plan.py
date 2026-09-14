import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from force_md.heavy_flow.corpus_plan import build_plan, validate_plan, fixed_frame_split, chunk_entries


def plan(**kwargs):
    return build_plan([(f'data/mdcath_dataset_d{i}.h5', 20) for i in range(20)],
                      repo_id='test/data', revision='fixed-sha', chunk_size=5, **kwargs)


def test_seeded_disjoint_complete_plan():
    p = plan()
    assert p == plan()
    assert p != plan(seed=1)
    assert validate_plan(p) == p
    assert len(p['splits']['train']) == 16
    assert len(chunk_entries(p, 0)) == 5
    broken = copy.deepcopy(p)
    broken['seed'] = 2
    with pytest.raises(ValueError):
        validate_plan(broken)


def test_legacy_train_never_becomes_heldout():
    key = lambda domain, frame: dict(domain=domain, temperature='320', replica='0', frame=frame)
    ckpt = {'chunk_rotation': {'step_in_chunk': 0, 'next_chunk': 1, 'total_chunks': 1,
                             'chunk_size': 5, 'domain_order': [f'd{i}' for i in range(5)]},
            'split_manifest': {'train': [key('d0', 0), key('d1', 0)],
                               'same_domain_validation': [key('d0', 1)],
                               'unseen_domain_validation': [key('d2', 0)]},
            'step': 10, 'normalizer': {'scale': 2}}
    p = plan(checkpoint=ckpt)
    assert [e['domain'] for e in chunk_entries(p, 0)] == [f'd{i}' for i in range(5)]
    assert {'d0', 'd1'} <= set(p['splits']['train'])
    assert 'd2' in p['splits']['validation']
    split = fixed_frame_split([key('d0', 0), key('d0', 1), key('d2', 0)], p)
    assert len(split.train) == 1
    assert len(split.same_domain_validation) == 2


def runner():
    path = Path(__file__).resolve().parents[2] / 'scripts/run_stage3_corpus.py'
    spec = importlib.util.spec_from_file_location('corpus_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eviction_only_exact_managed_shards(tmp_path):
    m, p = runner(), plan()
    directory = m.slot(tmp_path, 0)
    directory.mkdir()
    shard = directory / Path(chunk_entries(p, 0)[0]['path']).name
    shard.write_bytes(b'raw')
    retained = directory / 'audit.json'
    retained.write_text('{}')
    m.evict(tmp_path, p, 0)
    assert not shard.exists() and retained.exists()
    outside = tmp_path / 'outside.h5'
    outside.write_bytes(b'keep')
    shard.symlink_to(outside)
    with pytest.raises(ValueError):
        m.evict(tmp_path, p, 0)
    assert outside.read_bytes() == b'keep'


def test_prepare_overlaps_training_and_eviction_follows_checkpoint(tmp_path, monkeypatch):
    import threading
    m, p = runner(), plan()
    # Two slots sufficient to test the pipeline.
    p['shards'] = p['shards'][:10]
    args = SimpleNamespace(work_dir=str(tmp_path / 'data'), output_dir=str(tmp_path / 'out'),
                           initial_checkpoint=None, plan=str(tmp_path / 'plan'), config='config')
    Path(args.work_dir).mkdir()
    next_ready = threading.Event()
    events = []
    commands = []
    def prepare(args, plan, index):
        if index == 1:
            next_ready.set()
        return m.slot(args.work_dir, index)
    def train(*a, **kw):
        assert next_ready.wait(3)
        commands.append(a[0])
        events.append('train')
    monkeypatch.setattr(m, 'prepare', prepare)
    monkeypatch.setattr(m.subprocess, 'run', train)
    monkeypatch.setattr(m, 'completed', lambda *a: events.append('verified'))
    monkeypatch.setattr(m, 'evict', lambda *a: events.append('evicted'))
    m.run_pipeline(args, p)
    assert events == ['train', 'verified', 'evicted'] * 2
    assert commands[0][commands[0].index('--corpus-chunk-index') + 1] == '0'
    assert '--reuse-normalizer' in commands[0] and '--fit-normalizer' not in commands[0]
    assert '--initialize-from' in commands[1] and '--reuse-normalizer' not in commands[1]
    assert commands[1][commands[1].index('--initialize-from') + 1].endswith('corpus_0000_latest.pt')


def test_v2_rejects_old_state_before_preparation(tmp_path, monkeypatch):
    import json
    m, p = runner(), plan()
    (tmp_path / 'state.json').write_text(json.dumps({'digest': p['digest'], 'next_chunk': 2, 'checkpoint': 'old'}))
    args = SimpleNamespace(work_dir=str(tmp_path), initial_checkpoint=None)
    monkeypatch.setattr(m, 'prepare', lambda *a: pytest.fail('must not download'))
    with pytest.raises(ValueError, match='legacy corpus state'):
        m.run_pipeline(args, p)


def test_download_manifest_survives_eviction(tmp_path, monkeypatch):
    import io
    import json
    m, p = runner(), plan()
    monkeypatch.setattr(m.time, 'sleep', lambda seconds: None)
    sys.path.insert(0, str(Path(m.__file__).parent))
    try:
        import download_mdcath
    finally:
        sys.path.pop(0)
    args = SimpleNamespace(work_dir=str(tmp_path), reserve_gb=0, config='config')
    def download(*a, **kw):
        # Manifest must exist before the first network request.
        manifest = json.loads((m.slot(tmp_path, 0) / 'mdcath_manifest.json').read_text())
        assert manifest['revision'] == p['revision']
        assert manifest['corpus_plan_digest'] == p['digest']
        return io.BytesIO(b'x' * 20)
    monkeypatch.setattr(m.urllib.request, 'urlopen', download)
    monkeypatch.setattr(download_mdcath, 'run_audits', lambda *a, **kw: 0)
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: None)
    directory = m.prepare(args, p, 0)
    assert len(list(directory.glob('*.h5'))) == 5
    monkeypatch.setattr(m.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('completed files must be skipped'))
    m.prepare(args, p, 0)
    m.evict(tmp_path, p, 0)
    assert not list(directory.glob('*.h5'))
    assert len(json.loads((directory / 'mdcath_manifest.json').read_text())['shards']) == 5


def test_legacy_checkpoint_never_authorizes_eviction(monkeypatch):
    m = runner()
    monkeypatch.setattr(m, 'checkpoint', lambda path: {'stage': 'stage3_atom_physics'})
    with pytest.raises(ValueError, match='incompatible architecture'):
        m.completed('old.pt', plan(), 0)


def test_failed_training_never_evicts(tmp_path, monkeypatch):
    m, p = runner(), plan()
    args = SimpleNamespace(work_dir=str(tmp_path), output_dir=str(tmp_path / 'out'),
                           initial_checkpoint=None, plan='plan', config='config')
    monkeypatch.setattr(m, 'prepare', lambda a, p, i: m.slot(tmp_path, i))
    def fail(*a, **kw):
        raise RuntimeError('training failure')
    monkeypatch.setattr(m.subprocess, 'run', fail)
    monkeypatch.setattr(m, 'evict', lambda *a: pytest.fail('must retain failed chunk'))
    with pytest.raises(RuntimeError, match='training failure'):
        m.run_pipeline(args, p)
    assert not (tmp_path / 'state.json').exists()
