# Stage 4 heavy-atom Cartesian rectified flow report

> Historical report. Current implementation and training commands:
> [heavy-flow v2](docs/heavy_flow_v2_architecture.md). V2 uses typed pair messages at updated coordinates and requires a strict v2 upstream checkpoint.

## 요약

Stage 4 direct Cartesian velocity-field decoder를 구현했다. Stage 1–3의
ESM-C/geometry/context/physics 출력을 inference-time conditioning으로만 사용하고,
학습되는 경계는 flow decoder의 adapters·gates·message blocks로 제한했다.
기존 H0/H1a/H1b/H2 및 residue-frame transition head는 Stage 4 경로에 연결하지
않았다.

Stage 4 시작 시에는 이제 Stage 3 checkpoint를 반드시 handoff한다.
`build_from_stage3_config(checkpoint["config"])`로 동일한 Stage 3 composite를
구성하고 `checkpoint["upstream"]`를 `strict=True`로 로드한 뒤, ESM-C projection,
geometry encoder, atom→residue pooling, seq–geo fusion, residue→atom refinement,
physics predictor와 force head 전체를 freeze/eval 상태로 decoder에 연결한다.
새 Stage 3 checkpoint는 이 state를 `upstream` 키에도 저장하며, 새 Stage 4
checkpoint는 재개를 위해 upstream/config/provenance를 내장한다.

## 구현 범위

| 영역 | 구현 |
|---|---|
| Flow contract | `FlowAtomTopology`, `FlowConditionBundle` 및 packed/dense/mask 계약 |
| Path/target | `align_future_to_current()`의 target-only Kabsch, `x0=current+sigma(lag)*noise`, COM noise 제거, linear interpolation, true RF velocity target |
| Decoder | `FlowDecoder.forward(x_s, flow_time, atom_context, residue_context, physics_state, atom_topology, temperature, lag)`; 출력 `[B,N,3]` polar `1o` velocity |
| Conditioning | atom/residue/physics/force/time/temperature/lag/topology를 별도 adapter와 gate로 주입; physics vector/axial은 equivariant tensor product 경로 사용 |
| Graph | covalent/chain edges 고정, spatial neighbors는 현재 `x_s`에서 cutoff/max-neighbor로 동적 재계산 |
| Solver | Euler와 Heun, `steps=1` 경계, trajectory 선택 반환 |
| Geometry | bond/angle/peptide/backbone torsion/sidechain torsion/chirality/clash regularizer 및 term별 gradient norm |
| Training | frozen upstream wrapper, RF + geometry objective, adapter-only optimizer, bounded CLI |
| Sampling | `model.sample(condition_or_bundle, num_samples, seed, solver, steps)`; future/GT force/oracle 입력 없음 |
| Upstream handoff | learned Stage 3 checkpoint 필수; missing/incomplete upstream 또는 non-strict shape mismatch는 즉시 실패 |

주요 파일:

- [flow_types.py](src/force_md/heavy_flow/flow_types.py)
- [rectified_flow.py](src/force_md/heavy_flow/rectified_flow.py)
- [flow_decoder.py](src/force_md/heavy_flow/flow_decoder.py)
- [geometry_losses.py](src/force_md/heavy_flow/geometry_losses.py)
- [solver.py](src/force_md/heavy_flow/solver.py)
- [sampler.py](src/force_md/heavy_flow/sampler.py)
- [train_flow.py](experiments/heavy_flow/train_flow.py)
- [sample_flow.py](experiments/heavy_flow/sample_flow.py)
- [stage4.yaml](configs/heavy_flow/stage4.yaml)

## Leakage 및 고정 경계

`x_future`는 `train_flow_steps()`에서 path와 target velocity를 만들 때만 읽는다.
`HeavyAtomFlowModel.encode_condition()`과 decoder forward에는 전달되지 않는다.
force mean/log-variance 역시 Stage 3 predictor/head가 현재 condition에서 예측한
값만 사용한다. `HeavyAtomFlowModel`은 기본적으로 context encoder, physics
predictor, force head의 `requires_grad`를 끄고 eval 상태를 유지한다.

Kabsch는 loss target인 future에만 적용한다. 현재 좌표와 physics/context를
정렬하거나 future-derived graph를 만들지 않는다. Sampling path는 현재 좌표와
seeded noise만으로 base를 만든다.

## 검증 결과

### Stage 4 전용 회귀

```text
PYTHONPATH=src:. python -m pytest -q tests/heavy_flow/test_stage4_flow.py
8 passed
```

검증한 항목은 다음과 같다.

- decoder signature에 future/target/GT force 인자 없음
- target-only Kabsch, `s=0/1` path boundary, mask
- 회전 equivariance 및 현재 좌표 기반 dynamic graph cache
- atom permutation consistency
- masked RF/geometry loss의 finite gradient
- physics conditioning을 0으로 만들었을 때 velocity 변화
- seed 재현성, 다른 noise, Euler/Heun `steps=1` 및 multi-step
- decoder adapter gradient와 upstream freeze

### 실제 mdCATH 인접-frame preflight

실제 shard에서 `1a0rP01 / 320 / replica 0`의 frame `0 → 1`을 사용했다.
현재 frame과 다음 frame은 서로 다른 입력 객체로 읽었고, 다음 frame은 RF
training path의 target에만 넣었다.

```text
represented atoms: 1,017 heavy atoms
residues:          129
RF steps:          2
loss:              6.032518 -> 5.982415
flow loss:         5.992997 -> 5.943625
sampling:          Euler, 1 step, K=2
sample shape:      [2, 1, 1017, 3]
finite output:     True
```

이 preflight는 데이터·topology·path·decoder·solver가 실제 단백질 크기에서
연결되는지 확인하는 smoke run이다. 환경에 ESM-2 cache가 없으므로 reader는
`allow_fake_plm=True`, Stage 4 sequence encoder는 명시적 test-only `stub`으로
실행했다. 따라서 이 수치는 ESM-C scientific result나 성능 주장으로 해석하지
않는다. 실제 Stage 4 실행은 `stage4.yaml`의 ESM-C backend와 strict cache를
사용해야 한다.

### 전체 회귀

수정 후 `PYTHONPATH=src:. python -m pytest -q tests`를 실행했다. 84% 지점까지
failure 없이 진행됐으나, 이후 CPU 장시간 케이스가 6분 이상 진행되지 않아
무한 대기를 피하려고 실행을 중단했다. 이 중단은 테스트 failure가 아니다.
Stage 1–3 기존 회귀 및 Stage 4 전용 회귀는 별도로 통과했다.

## 실행 예시

bounded training:

```bash
PYTHONPATH=src:. python experiments/heavy_flow/train_flow.py \
  --config configs/heavy_flow/stage4.yaml \
  --stage3-checkpoint outputs/heavy_flow/stage3/checkpoint.pt \
  --data-dir data --max-domains 1 --max-samples 2 --steps 4 \
  --output outputs/heavy_flow/stage4/checkpoint.pt
```

sampling:

```bash
PYTHONPATH=src:. python experiments/heavy_flow/sample_flow.py \
  --checkpoint outputs/heavy_flow/stage4/checkpoint.pt \
  --data-dir data --max-domains 1 --sample-index 0 \
  --num-samples 8 --solver heun --steps 20 \
  --output outputs/heavy_flow/stage4/samples.pt
```

## 남은 scientific 단계

구현 및 bounded preflight는 완료했지만, full mdCATH transition training, K=8
validation subset, long-horizon stability screening, 그리고 실제 ESM-C cache를
사용한 성능 평가는 아직 수행하지 않았다. 현재 mdCATH adapter의 기본 frame-only
경로는 identity future도 지원하므로, scientific training에서는 반드시 실제
동일 trajectory의 temporal future pair dataset을 공급해야 한다.
