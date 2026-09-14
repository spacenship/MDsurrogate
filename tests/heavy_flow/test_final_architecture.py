"""v2 risk checks. Tiny optimizer/checkpoint fixtures are not pretrained models."""
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
import torch
from e3nn import o3

from experiments.heavy_flow.train_physics import build_stage3_model, save_stage3_checkpoint, train_physics_steps
from experiments.heavy_flow.train_flow import build_stage4_model, load_stage4_checkpoint, save_stage4_checkpoint, train_flow_steps
from force_md.heavy_flow import ForceNormalizer, HeavyFlowSample, HeavyFlowTargets, force_loss
from force_md.heavy_flow.checkpoint import load_normalizer_artifact, warm_start_stage3
from force_md.heavy_flow.history import HistoryConfig
from force_md.heavy_flow.physics_dataset import HeavyFlowPhysicsFrameDataset, HeavyFlowTemporalPairDataset, PhysicsSplitManifest
from force_md.heavy_flow.physics_edges import attach_physics_edges
from tests.heavy_flow.test_stage2_seq_geo import condition
from tests.heavy_flow.test_stage3_stage4_handoff import _stage3_config, _stage4_config
from tests.heavy_flow.test_stage4_flow import stage4_fixture


def sample():
    c = condition()
    # Non-rigid, non-planar past geometry so axial/history channels are exercised.
    torch.manual_seed(120)
    c.x_history[:, :-1] += torch.randn_like(c.x_history[:, :-1]) * .08
    return HeavyFlowSample(c, HeavyFlowTargets(torch.randn_like(c.current_positions),
                           c.current_positions + torch.randn_like(c.current_positions) * .04,
                           c.atom_mask, c.atom_mask))


def test_force_only_gradients_pair_atom_axial_projection_and_lag_independence():
    torch.manual_seed(100)
    model = build_stage3_model(_stage3_config()).eval()
    s = sample()
    first = model(s.condition)
    second = model(replace(s.condition, lag=s.condition.lag * 4))
    torch.testing.assert_close(first.context.atom_features, second.context.atom_features, atol=0, rtol=0)
    for name in ('atom_scalar', 'atom_vector', 'atom_axial', 'edge_scalar'):
        torch.testing.assert_close(getattr(first.physics_state, name), getattr(second.physics_state, name), atol=0, rtol=0)
    changed_history = replace(s.condition, x_history=s.condition.current_positions[:, None].expand_as(s.condition.x_history).clone())
    assert not torch.allclose(first.context.atom_features, model(changed_history).context.atom_features)
    first.physics_state.edge_scalar.retain_grad()
    first.physics_state.atom_vector.retain_grad()
    force_loss(first.force_distribution, s.targets, s.condition).backward()
    groups = ['predictor.pair_projection.', 'predictor.scalar_projection.', 'predictor.vector_projection.',
              'predictor.axial_projection.', 'context_encoder.geometry_encoder.sequence_encoder.projection.',
              'context_encoder.fusion.', 'context_encoder.refiner.']
    for prefix in groups:
        grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), prefix
        assert sum(float(g.abs().sum()) for g in grads) > 1e-12, prefix
    assert first.physics_state.edge_scalar.grad.abs().sum() > 1e-12
    assert first.physics_state.atom_vector.grad.abs().sum() > 1e-12
    assert all(p.grad is None for p in model.context_encoder.geometry_encoder.sequence_encoder.backend.parameters())


def test_decoder_uses_pair_and_atom_fields_and_preserves_moved_edges(stage4_fixture):
    c, bundle, model = stage4_fixture
    x = c.current_positions.clone()
    x[:, -1] += 15.0  # Original physics edges can move out of cutoff.
    s = torch.tensor([.37])
    first = model.velocity(bundle, x, s)
    for fields in [('edge_scalar',), ('atom_scalar', 'atom_vector', 'atom_axial')]:
        changed = replace(bundle.physics_state, **{name: torch.zeros_like(getattr(bundle.physics_state, name)) for name in fields})
        other = model.velocity(replace(bundle, physics_state=changed), x, s)
        assert (other - first).abs().max() > 1e-7, fields
    # Edge order is arbitrary, but its paired index/type/mask travels with it.
    state = bundle.physics_state
    perm = torch.randperm(state.edge_scalar.shape[0])
    reordered = replace(state, edge_scalar=state.edge_scalar[perm], edge_index=state.edge_index[:, perm],
                        edge_kind=state.edge_kind[perm], bond_type=state.bond_type[perm], edge_mask=state.edge_mask[perm])
    torch.testing.assert_close(model.velocity(replace(bundle, physics_state=reordered), x, s), first, atol=1e-6, rtol=1e-6)
    graph = bundle.atom_topology.graph_for(x)
    merged, features = attach_physics_edges(graph, state, 5)
    assert int(features[:, -1].sum()) == state.edge_scalar.shape[0]
    assert torch.equal(merged.state.current, x[merged.state.batch, merged.state.local_atom])
    with pytest.raises(ValueError, match='mapping'):
        model.velocity(replace(bundle, physics_state=replace(state, atom_local=state.atom_local.flip(0))), x, s)


@pytest.mark.parametrize('reflection', [False, True])
def test_complete_current_history_generated_path_equivariance(stage4_fixture, reflection):
    _, _, model = stage4_fixture
    model.eval()
    c = sample().condition
    torch.manual_seed(123)
    q = o3.rand_matrix()
    if reflection:
        q = -q
    translation = torch.tensor([2.4, -3.1, .7])
    rotate = lambda v: v @ q.T
    transformed = replace(c, x_history=rotate(c.x_history) + translation)
    base = model.encode_condition(c)
    moved = model.encode_condition(transformed)
    torch.testing.assert_close(moved.atom_context, base.atom_context @ model.context_encoder.output_irreps.D_from_matrix(q).T, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(moved.residue_context, base.residue_context, atol=3e-4, rtol=3e-4)
    for name in ('atom_scalar', 'edge_scalar', 'force_logvar'):
        torch.testing.assert_close(getattr(moved.physics_state, name), getattr(base.physics_state, name), atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(moved.physics_state.atom_vector, rotate(base.physics_state.atom_vector), atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(moved.physics_state.atom_axial, rotate(base.physics_state.atom_axial) * torch.det(q), atol=3e-4, rtol=3e-4)
    x = c.current_positions + torch.randn_like(c.current_positions) * .03
    v = model.velocity(base, x, torch.tensor([.42]))
    v2 = model.velocity(moved, rotate(x) + translation, torch.tensor([.42]))
    torch.testing.assert_close(v2, rotate(v), atol=3e-4, rtol=3e-4)


def test_optimizer_fixture_handoff_roundtrip_frozen_upstream(tmp_path):
    torch.manual_seed(125)
    config = _stage3_config()
    model = build_stage3_model(config)
    s = sample()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    losses = train_physics_steps(model, [s], opt, steps=3)
    assert len(losses) == 3 and torch.isfinite(torch.tensor(losses)).all()
    # This is explicitly a three-step verification fixture, not a completed training run.
    model.training_provenance = {'purpose': 'three_step_verification_fixture'}
    path = tmp_path / 'stage3_fixture.pt'
    save_stage3_checkpoint(path, model, config=config, normalizer=ForceNormalizer(scale=2., count=6),
                           split_manifest=PhysicsSplitManifest([], [], []), optimizer=opt, step=None)
    with pytest.raises(ValueError, match='fixture'):
        build_stage4_model(_stage4_config(config), stage3_checkpoint=path)
    flow = build_stage4_model(_stage4_config(config), stage3_checkpoint=path, allow_untrained_fixture=True)
    model.eval()
    before = model(s.condition)
    bundle = flow.encode_condition(s.condition)
    torch.testing.assert_close(bundle.atom_context, before.context.atom_features, atol=0, rtol=0)
    for name in ('atom_scalar', 'atom_vector', 'atom_axial', 'edge_scalar'):
        torch.testing.assert_close(getattr(bundle.physics_state, name), getattr(before.physics_state, name), atol=0, rtol=0)
    upstream = {k: v.clone() for k, v in flow.state_dict().items() if not k.startswith('decoder.')}
    decoder_before = {k: v.clone() for k, v in flow.decoder.state_dict().items()}
    opt2 = torch.optim.AdamW(flow.trainable_parameters, lr=1e-4)
    logs = train_flow_steps(flow, [s], opt2, steps=3, report_gradient_norms=False, seed=12)
    assert all(torch.isfinite(torch.tensor(row['loss'])) for row in logs)
    for key, tensor in upstream.items():
        assert torch.equal(flow.state_dict()[key], tensor), key
    assert any(not torch.equal(flow.decoder.state_dict()[k], v) for k, v in decoder_before.items())
    output = tmp_path / 'stage4_fixture.pt'
    save_stage4_checkpoint(output, flow, config=_stage4_config(config), optimizer=opt2, step=3)
    restored, payload = load_stage4_checkpoint(output, allow_untrained_fixture=True)
    after = restored.encode_condition(s.condition)
    torch.testing.assert_close(after.atom_context, bundle.atom_context, atol=0, rtol=0)
    torch.testing.assert_close(after.physics_state.edge_scalar, bundle.physics_state.edge_scalar, atol=0, rtol=0)
    assert restored.upstream_provenance == flow.upstream_provenance
    assert payload['normalizer']['scale'] == 2.
    bad = torch.load(path, weights_only=False)
    bad.pop('architecture_version')
    with pytest.raises(ValueError, match='incompatible architecture'):
        build_stage4_model(_stage4_config(config), stage3_checkpoint=bad, allow_untrained_fixture=True)
    # Explicit legacy warm-start lists missing/new weights and keeps them trainable.
    legacy = dict(bad)
    legacy['upstream'] = dict(legacy['upstream'])
    key = next(k for k in legacy['upstream'] if k.startswith('predictor.pair_projection.') and k.endswith('weight'))
    del legacy['upstream'][key]
    torch.save(legacy, tmp_path / 'legacy.pt')
    fresh = build_stage3_model(config)
    report = warm_start_stage3(fresh, tmp_path / 'legacy.pt')
    assert key in report['not_loaded'] and report['requires_stage3_force_training']
    assert dict(fresh.named_parameters())[key].requires_grad


class TrajectoryReader:
    """Distinct raw frames from multiple trajectory identities, no frame-index sampling."""
    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace(ps_per_frame=1000., data_dir='unused', represented_scope='heavy_atom')
        self.coord_quarantine = {'a': {'320/0': {7}}}
        self.c = condition()

    def _open(self, domain):
        return {domain: {t: {r: SimpleNamespace(attrs={'numFrames': 12}) for r in ('0', '1')} for t in ('320', '350')}}

    def load_frame_arrays(self, domain, temp, replica, frame):
        self.calls.append((domain, temp, replica, frame))
        shift = frame * torch.arange(6)[:, None] * torch.tensor([.02, -.01, .03])
        positions = self.c.current_positions[0] + shift
        return positions, positions * 0 + 1, True

    def build_example(self, domain, temp, replica, frame, coords, forces, valid):
        c = self.c
        atoms = SimpleNamespace(positions=coords, forces=forces, force_valid=c.atom_mask[0],
            atom_to_residue=c.atom_to_residue[0], atom_name_id=c.atom_name[0], atomic_number=c.atom_type[0],
            is_backbone=c.is_backbone[0], is_cap=torch.zeros(6, dtype=torch.bool))
        residues = SimpleNamespace(mask=c.residue_mask[0], residue_type=c.sequence_tokens[0], chain_index=torch.zeros(3, dtype=torch.long))
        batch = SimpleNamespace(atoms=atoms, residues=residues, num_graphs=1,
             temperature=torch.tensor([float(temp)]), replica_index=torch.tensor([int(replica)]), domain_id=[domain],
             frame_index=torch.tensor([frame]), units=SimpleNamespace(length='angstrom', force='kcal/mol/angstrom', temperature='kelvin'))
        return SimpleNamespace(batch=batch)


def prepared_dataset(cls, reader, keys, **kwargs):
    dataset = cls(reader, keys, **kwargs)
    for domain, *_ in keys:
        dataset._bond_cache[domain] = (reader.c.bond_index[0], reader.c.bond_type[0])
    return dataset


def test_history_is_past_same_trajectory_and_separate_from_future_lag():
    reader = TrajectoryReader()
    keys = [('a', '320', '0', 4), ('a', '350', '1', 4), ('b', '320', '1', 1)]
    history = HistoryConfig(length=3, stride_frames=2)
    stage3 = prepared_dataset(HeavyFlowPhysicsFrameDataset, reader, keys, history_config=history)
    first = stage3[0]
    assert set(reader.calls) == {('a', '320', '0', t) for t in (0, 2, 4)}
    assert first.condition.provenance.history_frames == ((0, 2, 4),)
    reader.calls.clear()
    stage3[1]
    assert set(reader.calls) == {('a', '350', '1', t) for t in (0, 2, 4)}
    early = stage3[2]
    assert early.condition.provenance.history_frames == ((0, 0, 1),)
    assert torch.equal(early.condition.x_history[:, 0], early.condition.x_history[:, 1])
    for lag in (1, 2):
        pairs = prepared_dataset(HeavyFlowTemporalPairDataset, reader, keys, history_config=history, future_lag_frames=lag)
        pair = pairs[0]
        assert torch.equal(pair.condition.x_history, first.condition.x_history)
        assert pair.condition.lag.item() == lag
        assert pair.targets.provenance.future_frame == (4 + lag,)
        assert not torch.equal(pair.targets.x_future, pair.condition.current_positions)
    dropped = prepared_dataset(HeavyFlowPhysicsFrameDataset, reader, keys, history_config=HistoryConfig(3, 2, 'drop'))
    assert len(dropped) == 2
    quarantined = prepared_dataset(HeavyFlowPhysicsFrameDataset, reader, [('a', '320', '0', 9)], history_config=history)
    assert len(quarantined) == 0  # history includes quarantined raw frame 7
    no_future = prepared_dataset(HeavyFlowTemporalPairDataset, reader, [('b', '320', '0', 11)], future_lag_frames=1)
    assert len(no_future) == 0


def test_compatible_normalizer_reuse_is_read_only(tmp_path):
    path = tmp_path / 'normalizer.json'
    payload = {'normalizer': ForceNormalizer(scale=26.98, count=123).as_dict(), 'fit_spec': {'config': {'data': {'represented_scope': 'heavy_atom'}}}}
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    normalizer, provenance = load_normalizer_artifact(path)
    assert normalizer.scale == 26.98 and provenance['original_artifact'] == payload
    assert path.read_bytes() == original
    payload['normalizer']['force_unit'] = 'other'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='normalizer'):
        load_normalizer_artifact(path)


def test_production_context_dimensions_and_padding_independence():
    from experiments.heavy_flow.train_physics import load_yaml
    from torch.nn.functional import pad
    config = load_yaml('configs/heavy_flow/stage3.yaml')
    config['sequence_encoder'] = {'backend': 'stub', 'stub_dim': 960, 'projected_dim': 64}
    model = build_stage3_model(config).context_encoder.eval()
    c = condition()
    with torch.no_grad():
        result = model(c)
        assert result.atom_features.shape == (6, 232)
        assert result.residue_context.geometry_invariant.shape == (1, 3, 136)
        assert result.residue_context.joint_scalar.shape == (1, 3, 384)
        assert len(model.fusion.blocks) == 4 and model.fusion.attention_heads == 8
        assert model.refiner.local_irreps.dim == 232 and len(model.refiner.blocks) == 2
        padded_pool = replace(result.pooling, padded_features=pad(result.pooling.padded_features, (0, 0, 0, 2)),
                               residue_mask=pad(result.pooling.residue_mask, (0, 2)),
                               residue_centers=pad(result.pooling.residue_centers, (0, 0, 0, 2)))
        padded = model.fusion(padded_pool, chain_break=pad(c.chain_break, (0, 2)))
        torch.testing.assert_close(padded.joint_scalar[:, :3], result.residue_context.joint_scalar, atol=1e-5, rtol=1e-5)
        assert padded.joint_scalar[:, 3:].abs().sum() == 0
        empty_pool = replace(result.pooling, residue_mask=torch.zeros_like(result.pooling.residue_mask))
        empty = model.fusion(empty_pool, chain_break=c.chain_break)
        assert torch.isfinite(empty.joint_scalar).all() and empty.joint_scalar.abs().sum() == 0


def test_history_metadata_and_checkpoint_config_mismatch_are_rejected(tmp_path):
    from tests.heavy_flow.test_stage3_stage4_handoff import fixture_metadata
    config = _stage3_config()
    model = build_stage3_model(config)
    bad_condition = replace(condition(), x_history=condition().x_history[:, -1:])
    with pytest.raises(ValueError, match='history length'):
        model(bad_condition)
    payload = {**fixture_metadata(), 'stage': 'stage3_atom_physics', 'config': config, 'upstream': model.state_dict()}
    payload['history'] = {**payload['history'], 'stride_frames': 2}
    with pytest.raises(ValueError, match='history'):
        build_stage4_model(_stage4_config(config), stage3_checkpoint=payload, allow_untrained_fixture=True)
    reader = TrajectoryReader()
    reader.config.ps_per_frame = None
    dataset = prepared_dataset(HeavyFlowTemporalPairDataset, reader, [('b', '320', '0', 4)])
    assert dataset[0].condition.provenance.time_per_frame == 1000.
