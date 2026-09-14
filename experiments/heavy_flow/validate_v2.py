"""Bounded CPU validation on one local real trajectory (no download/checkpoint training).

Uses cached real ESM-C embeddings, a reduced geometry/physics/decoder network,
and the existing force RMS. The resulting weights are verification fixtures.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from force_md.data.adapters.mdcath import MdCathConfig, MdCathDataset
from force_md.heavy_flow.history import HistoryConfig
from force_md.heavy_flow.physics_dataset import HeavyFlowTemporalPairDataset
from force_md.heavy_flow.checkpoint import load_normalizer_artifact, architecture_config
from force_md.heavy_flow.flow_decoder import FlowDecoder, FlowDecoderConfig
from force_md.heavy_flow.sampler import HeavyAtomFlowModel
from experiments.heavy_flow.train_physics import build_stage3_model, load_yaml, train_physics_steps
from experiments.heavy_flow.train_flow import train_flow_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--domain', default='1a0rP01')
    parser.add_argument('--frame', type=int, default=2)
    parser.add_argument('--config', default='configs/heavy_flow/stage3.yaml')
    parser.add_argument('--normalizer-path', default='outputs/heavy_flow/stage3/chunk_rotation_normalizer.json')
    parser.add_argument('--output', default='outputs/heavy_flow/v2_validation/real_batch.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(407)
    config = architecture_config(load_yaml(args.config))
    config['sequence_encoder']['device'] = 'cpu'
    config['sequence_encoder']['cache_only'] = True
    config['geometry'].update(num_blocks=1, hidden_irreps='8x0e + 2x1o + 1x1e + 1x2e', radial_basis=4)
    config['graph']['max_spatial_neighbors'] = 3
    config['context'].update(joint_scalar_dim=32, fusion_blocks=2, attention_heads=4, dropout=0.,
                             geometry_modality_dropout=0., attention_hidden_dim=16, identity_embedding_dim=4)
    config['physics'].update(physics_blocks=1, scalar_dim=12, vector_channels=2, axial_channels=1,
                             pair_dim=5, radial_basis=4, edge_embedding_dim=3)
    reader = MdCathDataset(MdCathConfig(data_dir=args.data_dir, temperatures=(320,), replicas=(0,),
                                       frames_per_trajectory=1, allow_fake_plm=True), domains=[args.domain])
    # The reader's unused legacy ESM-2 feature is stubbed; model ESM-C is real/cache-only.
    dataset = HeavyFlowTemporalPairDataset(reader, [(args.domain, '320', '0', args.frame)],
                 history_config=HistoryConfig.from_config(config), future_lag_frames=1)
    if len(dataset) != 1:
        raise ValueError('requested local frame/history/future is unavailable or quarantined')
    sample = dataset[0]
    normalizer, provenance = load_normalizer_artifact(args.normalizer_path)
    model = build_stage3_model(config, device='cpu')
    # Fail if any uncached sequence computation would be attempted.
    def no_compute(*args, **kwargs):
        raise AssertionError('validation must reuse the ESM-C cache')
    model.context_encoder.geometry_encoder.sequence_encoder._compute_one = no_compute
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    physics_losses = train_physics_steps(model, [sample], opt, normalizer=normalizer, steps=3)
    pair_gradient = sum(float(p.grad.abs().sum()) for p in model.predictor.pair_projection.parameters() if p.grad is not None)
    assert pair_gradient > 0
    decoder = FlowDecoder(model.context_encoder.output_irreps, 32, 12, 2, 1, physics_pair_dim=5,
                   config=FlowDecoderConfig(flow_blocks=2, hidden_irreps='8x0e + 2x1o + 1x1e + 1x2e',
                        conditioning_layers=(0, 1), radial_basis=4, edge_embedding_dim=3,
                        time_embedding_dim=8, atom_identity_dim=4, max_spatial_neighbors=3))
    flow = HeavyAtomFlowModel(model.context_encoder, model.predictor, model.force_head, decoder)
    # Frozen parameters remain unchanged; retain a small upstream tensor for a direct check.
    projection_before = model.context_encoder.geometry_encoder.sequence_encoder.projection.weight.detach().clone()
    opt2 = torch.optim.AdamW(flow.trainable_parameters, lr=1e-4)
    logs = train_flow_steps(flow, [sample], opt2, steps=3, report_gradient_norms=False, seed=409)
    assert flow.upstream_is_frozen
    assert torch.equal(projection_before, model.context_encoder.geometry_encoder.sequence_encoder.projection.weight)
    result = {'purpose': 'verification_fixture_not_completed_training', 'device': 'cpu',
              'reduced_network': True, 'esmc_backend': 'real_esmc_300m_existing_cache_only',
              'sample_provenance': sample.condition.provenance.as_dict(),
              'atom_count': int(sample.condition.atom_mask.sum()), 'residue_count': int(sample.condition.residue_mask.sum()),
              'history_length': sample.condition.history_length, 'prediction_lag_frames': sample.condition.lag.tolist(),
              'force_normalizer': normalizer.as_dict(), 'normalizer_sha256': provenance['sha256'],
              'stage3_losses': physics_losses, 'pair_projection_gradient_l1': pair_gradient,
              'stage4_losses': [row['loss'] for row in logs], 'upstream_frozen': flow.upstream_is_frozen,
              'projection_unchanged': True, 'checkpoint_written': False}
    reader.close()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
