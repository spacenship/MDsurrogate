# Claude Code Opus 구현 프롬프트 — Phase 1.5 Physics-Latent Transition Probe

아래 프롬프트는 이미 학습이 끝난 Phase 1을 보존한 채, force-supervised representation이 실제 미래 구조 전이에 도움이 되는지를 작은 vertical slice로 검증하기 위한 것이다. 전체 Phase 2 stochastic flow를 한 번에 만들지 않는다.

## Claude Code Opus에 입력할 프롬프트

```text
당신은 protein molecular dynamics, coarse graining, geometric deep learning, PyTorch/e3nn 및 분산 학습에 익숙한 research software engineer다. 이 저장소에는 Force-Conditioned Protein MD 프로젝트의 Phase 1이 실제 구현·학습되어 있다. 기존 구현을 폐기하거나 별도 toy project를 만들지 말고, Phase 1 checkpoint와 tensor contract를 그대로 재사용하는 Phase 1.5 transition probe를 구현하라.

나는 코드를 이해하며 단계적으로 진행하고 싶다. 아래 Checkpoint를 반드시 순서대로 수행한다. 첫 응답에서는 Checkpoint 0만 수행하고 어떤 파일도 수정하지 않는다. 각 Checkpoint가 끝날 때 다음을 짧고 구체적으로 보고하고 내 "다음" 응답을 기다린다.

1. 생성·수정한 파일
2. 데이터 흐름과 tensor shape
3. 핵심 수학·물리적 의미
4. 실행한 테스트와 결과
5. 아직 남은 위험 또는 결정 사항

내가 명시적으로 "계속 모두 진행"이라고 하지 않는 한 여러 Checkpoint를 한 번에 건너뛰지 마라.

════════════════════════════════════════
0. 현재 상태와 이번 작업의 목적
════════════════════════════════════════

Phase 1의 알려진 상태는 다음과 같다. 저장소를 조사해 실제 코드·config·checkpoint와 일치하는지 확인하되, 불일치 시 보고서보다 실제 코드를 우선하고 차이를 기록하라.

- checkpoint: `runs/phase1_full/last.pt`
- 모델: `LocalPhysicsModel`
- 단일 frame의 구조·서열·온도로 원자 힘, residue 힘·torque, uncertainty와 residue physics latent를 예측
- graph hierarchy: `Atom → Residue → Backbone → Residue → Atom`, 2 cycles
- latent irreps: `64x0e + 16x1o + 8x2e`, residue당 총 152차원
- frozen PLM: `facebook/esm2_t33_650M_UR50D`
- main target scope: heavy atom
- mdCATH frame spacing: 1000 ps = 1 ns
- Phase 1의 40-frame subsampling은 trajectory당 약 12.8 ns 간격이며, Phase 1.5 lag pair 생성에는 사용할 수 없음
- Phase 1 최종 성능은 atomic force에는 강하지만 residue net force·torque와 hidden residual에는 한계가 있음
- energy branch는 최종 checkpoint에서 비활성

이번 작업의 유일한 연구 질문은 다음과 같다.

    현재 구조와 history만 사용하는 transition baseline보다
    Phase 1의 force-supervised physics representation을 사용한 모델이
    1 ns와 4 ns 미래 구조를 더 잘 예측하는가?

Physics latent는 새로운 관측 정보가 아니다.

    z_phys = g(q_t, sequence, temperature)

즉 force label로 사전학습된 representation/inductive bias다. 따라서 단순히 feature dimension을 늘려 성능이 오른 것을 physics 효과라고 해석하면 안 된다. 모든 비교는 동일 split, 동일 transition backbone, 동일 training budget과 가능한 한 비슷한 trainable parameter budget을 사용한다.

════════════════════════════════════════
1. 이번 범위에서 하지 않을 것
════════════════════════════════════════

다음을 구현하거나 재학습하지 마라.

- Phase 1 전체 재작성 또는 120,000-step 재학습
- Phase 1 energy branch 재활성화
- hidden-hydrogen residual head 개선
- 8 ns, 16 ns full experiment
- stochastic flow matching 전체 구현
- Chapman–Kolmogorov loss
- 장시간 autoregressive rollout
- HFM-like consistency
- side-chain all-atom decoder 전체 구현
- 기존 데이터·체크포인트·사용자 파일 삭제
- full multi-seed 장시간 학습의 자동 실행

Phase 1.5는 버릴 toy code가 아니라 이후 Phase 2에 남길 `lag-pair dataset`, `conditioning interface`, `transition target`, `metrics`, `ablation runner`의 최소 구현이다.

════════════════════════════════════════
2. 바꾸면 안 되는 과학적 전제
════════════════════════════════════════

- 순간 force를 1–4 ns 동안 직접 적분해 미래 좌표를 만들지 않는다.
- residue net force `sum_a f_ia`는 병진에 대한 올바른 observable이지만 내부 force cancellation 때문에 충분한 conformational descriptor가 아니다.
- torque도 rigid rotation 정보이며 internal compression/shear를 모두 보존하지 않는다.
- ground-truth force는 production arm의 입력이 아니다. 오직 supervision 또는 명시적으로 분리된 oracle arm에서만 사용한다.
- production inference에서는 Phase 1이 예측한 force/uncertainty 또는 force-supervised latent만 사용한다.
- protein-only coarse dynamics는 open, generally non-Markovian system이다. deterministic Hamiltonian dynamics라고 부르지 않는다.
- mdCATH force는 특정 force field와 simulation protocol의 label이다.
- reflection을 symmetry로 강제하지 않고 chirality를 보존한다.
- Kabsch alignment를 input feature를 만드는 편법으로 사용하지 않는다. Global rigid-body motion을 제거한 target 정의와 평가에만 proper-rotation Kabsch를 사용할 수 있다.
- 최종 지표 하나만 보고 결론내리지 않는다. RMSD, frame rotation, pair distance/contact와 clash를 함께 본다.

════════════════════════════════════════
3. 저장소 조사와 변경 원칙
════════════════════════════════════════

Checkpoint 0에서 다음을 수행하라.

1. `AGENTS.md`, `CLAUDE.md`, `README.md`, `pyproject.toml`, 관련 config와 test를 읽는다.
2. `phase1_report.md`, `docs/open_questions.md`, `docs/phase1_hierarchy_and_contracts.md`, `docs/phase2_interface.md`가 있으면 모두 읽는다.
3. `LocalPhysicsModel`, `latent_contract()`, checkpoint loader, mdCATH adapter, split/index 생성, quarantine, frame geometry, collate, DDP training code를 찾는다.
4. 실제 checkpoint가 존재하면 metadata만 안전하게 검사한다. 임의로 덮어쓰지 않는다.
5. Phase 1 전체 test suite를 먼저 실행해 baseline을 기록한다.
6. 현재 code path에 맞춘 구체적 구현 계획과 예상 파일 목록을 제시한다.
7. 불명확한 tensor contract는 추측하지 말고 코드에서 확인한다.

기존 모듈을 재사용하고 public contract를 깨지 마라. 새 경로 이름은 저장소 구조에 맞게 조정할 수 있지만, 역할은 아래 Checkpoint 구성을 유지한다. 새 dependency는 가능한 한 추가하지 않는다. 현재 conda/Python 환경을 그대로 사용하고 Python 3.11을 강제하지 않는다.

════════════════════════════════════════
4. Checkpoint 1 — Raw-trajectory LagPairDataset
════════════════════════════════════════

목표: 원본 mdCATH trajectory에서 정확한 1 ns/4 ns transition pair를 만드는 재사용 가능한 dataset과 manifest를 구현한다.

필수 규칙:

- Phase 1에서 trajectory당 40개를 골랐던 subsampled frame index를 재사용하지 않는다.
- 원본 frame index에서 `lag_frames = lag_ps / ps_per_frame`로 계산한다.
- 기본 lag는 `[1000, 4000]` ps이고 `ps_per_frame=1000`이면 `[1, 4]` frame이다.
- 나눗셈 결과가 정수가 아니면 조용히 반올림하지 말고 명시적으로 실패한다.
- `(domain, temperature, replica, trajectory)`가 같은 범위 안에서만 pair를 만든다.
- trajectory 경계, temperature, replica를 넘지 않는다.
- history length 기본값은 2다. 즉 `(t-1, t) → (t+lag)`를 구성하며 `t-1 >= 0`을 보장한다.
- history length는 config로 1 또는 2를 선택할 수 있게 한다.
- 현재·history·future coordinate가 모두 유효해야 한다.
- 공정한 A–E ablation을 위해 기본 manifest는 oracle arm까지 동일 pair를 쓰도록 현재 frame의 GT force도 유효한 pair만 포함한다. production-only manifest 옵션은 별도로 허용할 수 있다.
- 기존 force/coordinate quarantine을 반드시 재사용한다. 누락된 quarantine path가 설정되면 Phase 1과 동일하게 fail closed한다.
- domain-level train/validation split을 Phase 1에서 그대로 복원한다. frame 단위로 다시 split하지 않는다.
- 동일한 domain이 train과 validation에 함께 나타나면 실패한다.
- pair ID에는 domain, temperature, replica, current frame, future frame, lag를 포함한다.
- index 생성은 deterministic하고 seed/config/checksum을 manifest metadata에 저장한다.
- padded dense representation을 만들지 않고 기존 flattened-ragged contract를 따른다.
- 전체 623 GB를 메모리에 올리지 않는다. lazy loading/indexing을 사용한다.
- smoke config에서 domain/trajectory/pair 수를 제한할 수 있게 한다.

반드시 테스트할 것:

- 1 ns는 실제 원본 frame index 차이 1, 4 ns는 차이 4
- 40-frame Phase 1 subsample의 인접 entry가 1 ns로 잘못 사용되지 않음
- cross-trajectory/cross-temperature/cross-replica pair 0개
- train/validation domain leakage 0개
- 마지막 frame, 짧은 trajectory, invalid/quarantined frame 처리
- 같은 config/seed에서 manifest가 byte-level 또는 content-level deterministic

Checkpoint 1 종료 시 실제 작은 mdCATH sample로 pair 예시 3개를 표시하되, 대형 전체 index는 아직 만들지 마라.

════════════════════════════════════════
5. Checkpoint 2 — Transition target와 geometry metrics
════════════════════════════════════════

목표: global diffusion/rotation에 속지 않으면서 미래 residue-frame 변화를 평가할 target과 metric을 구현한다.

기본 target은 future backbone frame이다.

- 현재와 미래의 공통 valid Cα를 사용한다.
- proper-rotation Kabsch(`det(R)=+1`)로 future 전체 단백질을 current에 align하여 global rigid-body motion을 제거한다.
- reflection을 허용하지 않는다.
- aligned future에 대해 residue별 Cα translation target과 N–CA–C frame orientation target을 만든다.
- rotation target은 기존 SO(3) utility가 있으면 재사용한다. 없으면 numerically stable한 relative rotation/log-map 또는 6D rotation representation을 구현하고 선택 이유를 문서화한다.
- loss/metric의 frame convention과 left/right multiplication을 명시한다.
- local-frame translation을 사용할 경우 다음 관계를 테스트한다.

      delta_r_global = R_current @ delta_r_local

- Kabsch는 target canonicalization 및 metric에만 사용하고 model input을 align하는 데 사용하지 않는다.

최소 metric:

1. Kabsch-aligned Cα RMSD
2. residue translation RMSE
3. residue-frame rotation geodesic error(degree)
4. pair-distance MAE
5. configurable contact metric
6. backbone 또는 reconstructed proxy의 clash rate
7. lag별 metric과 domain-macro 평균

가능하면 `phi/psi` torsion MAE도 추가하되, 기존 topology/geometry utility로 신뢰성 있게 계산할 수 있을 때만 구현한다. 무리하게 새 parser를 만들지 않는다.

반드시 테스트할 것:

- 임의 global translation/rotation 전후 metric 동일
- proper Kabsch가 reflection을 생성하지 않음
- identical structure에서 translation/rotation error 0에 가까움
- 알려진 회전 각도에 대한 geodesic error 정확성
- residue ordering/mask 불일치 시 명시적 실패

════════════════════════════════════════
6. Checkpoint 3 — Frozen Phase 1 feature extraction
════════════════════════════════════════

목표: Phase 1 checkpoint를 안전하게 복원하고 transition conditioner가 사용할 정보를 구조화한다.

기본 설정:

- `freeze_phase1: true`
- `eval()` mode
- `torch.no_grad()` 또는 명시적 detach
- Phase 1 normalization/config/latent contract를 checkpoint에서 복원
- runtime config가 checkpoint contract와 다르면 실패
- Phase 1 checkpoint를 수정하거나 optimizer state를 덮어쓰지 않음

구조화된 feature bundle을 만든다. 실제 field name은 기존 출력 contract에 맞추되 의미는 다음을 포함한다.

- residue physics latent `z_phys`, irreps와 row order metadata 포함
- predicted atomic force mean
- predicted atomic force log-variance/uncertainty
- atom→residue membership
- residue predicted net force/torque가 이미 있으면 포함
- current residue frames와 atom local coordinates
- residue/atom valid mask

GT force는 동일 bundle의 일반 field로 섞지 않는다. Oracle 전용 타입 또는 명시적 `OracleFeatureBundle`로 분리한다. Production conditioner가 oracle field에 접근하려 하면 테스트에서 실패하게 만들어라.

선택 사항으로 frozen feature cache를 구현할 수 있다. 구현한다면:

- checkpoint hash, Phase 1 config hash, sample/pair ID를 cache key에 포함
- domain 또는 trajectory 단위 shard
- atomic write
- stale/incompatible cache를 조용히 재사용하지 않음
- 원자 수가 다른 ragged sample 지원

반드시 테스트할 것:

- latent shape와 irreps가 checkpoint contract와 일치
- residue row order가 batch contract와 일치
- frozen mode에서 Phase 1 parameter gradient가 생성되지 않음
- checkpoint/config mismatch 검출
- production feature path에서 GT force 접근 불가

════════════════════════════════════════
7. Checkpoint 4 — ForcePattern/Shape Conditioner
════════════════════════════════════════

목표: residue net-force 합산만으로 사라지는 internal force pattern과 residue shape를 fixed-size conditioner로 만든다.

각 atom을 residue local frame으로 변환한다.

    y_ia       = R_i^T (x_ia - r_i)
    f_local_ia = R_i^T f_ia

Production arm에서는 `f_ia`로 Phase 1의 predicted atomic force mean을 사용한다. GT atomic force는 oracle arm에서만 사용한다.

최소 explicit moment:

    F_i   = sum_a f_ia
    tau_i = sum_a y_ia × f_local_ia
    M_i   = sum_a y_ia ⊗ f_local_ia

`M_i`는 최소한 다음 성분으로 분해하거나 동등한 정보를 보존한다.

- trace / isotropic compression scalar
- antisymmetric component / torque-related component
- symmetric traceless component / directional stress·shear

그 위에 permutation-invariant learned atom-set pooling을 구현한다.

atom message 입력 후보:

- 기존 atom embedding/type
- local coordinate `y_ia`
- predicted local force
- predicted log-variance
- backbone/side-chain flag
- 기존 코드에서 신뢰성 있게 얻을 수 있는 element 또는 vdW-radius/size feature

임의의 vdW radius table을 근거 없이 만들지 않는다. 기존 force-field atom type/radius가 있으면 재사용하고, 없다면 element/type embedding과 coordinate radial moment를 우선 사용한다. 새 상수표가 꼭 필요하면 출처와 단위를 문서화한다.

pooling은 기존 flattened-ragged/segment operation을 사용한다. residue별 dense atom padding을 도입하지 않는다. 단순히 raw force를 먼저 합산한 뒤 MLP에 넣지 말고, atom별 nonlinear message를 만든 뒤 segment pooling한다.

다음 conditioner를 하나의 공통 interface 아래 구현한다.

- `ZeroConditioner`: structure-only control용 고정 크기 zero condition
- `ResidueForceTorqueConditioner`: predicted residue F/torque
- `PhysicsLatentConditioner`: current 152D force-supervised latent
- `ForcePatternShapeConditioner`: z_phys + predicted atom-force pattern + explicit moments + shape feature
- `OracleAtomicForceConditioner`: GT atom force를 사용하는 진단 전용 arm

각 conditioner는 동일한 최종 conditioning dimension `d_cond`를 출력한다. 서로 다른 입력 차원 때문에 parameter count가 달라지는 것은 기록하고, adapter depth/hidden width를 가능한 한 맞춘다. e3nn irreps를 scalar MLP에 무작정 flatten해 rotation convention을 깨지 않는다. global irreps를 local frame으로 옮기거나 equivariant projection을 사용하고 선택을 문서화한다.

반드시 테스트할 것:

- atom 순서를 permutation해도 residue conditioner 출력이 동일하거나 정의한 equivariant 방식으로 동일
- global SE(3) transform 전후 conditioner invariant/equivariant contract 유지
- `(+f, -f)` 예제에서 net force는 0이지만 force moment/learned representation은 nonzero가 될 수 있음
- uncertainty 증가 시 force feature gating이 설정대로 감소
- zero conditioner와 모든 arm의 최종 shape 동일
- production arm에서 GT force를 전달하면 거부

════════════════════════════════════════
8. Checkpoint 5 — Minimal TransitionProbe
════════════════════════════════════════

목표: 전체 Phase 2 flow가 아니라 representation 비교에 필요한 작은 deterministic transition model을 구현한다.

공통 base input:

- current residue/backbone frames
- 기본 history `(t-1, t)`
- residue type/PLM feature
- temperature
- physical lag embedding(1 ns 또는 4 ns)
- current sequence/spatial residue graph

모든 arm은 동일한 transition backbone과 output head를 사용하고 conditioner만 교체한다. Conditioner enum/config를 사용하며 arm별 model copy-paste를 금지한다.

권장 arm:

    A: `structure_only`       = ZeroConditioner
    B: `force_torque`        = predicted residue F/torque
    C: `physics_latent`      = Phase 1 z_phys
    D: `force_pattern_shape` = z_phys + predicted atom-force pattern/moments/shape
    E: `oracle_force`        = GT atomic force pattern, 진단 전용

Transition backbone은 작은 residue-level graph model로 구현한다.

- 2–4 interaction blocks 정도의 probe 규모
- 동일 sequence/spatial edge semantics 재사용
- global SE(3) transform에 대한 올바른 output behavior
- residue translation update와 frame orientation update 출력
- optional uncertainty head는 기존 안정적인 utility가 있을 때만 사용
- probe 목적을 넘어서는 stochastic flow/noise schedule을 구현하지 않음

Loss 최소 구성:

    L = lambda_pos  * robust translation loss
      + lambda_rot  * SO(3) geodesic loss
      + lambda_pair * pair-distance auxiliary loss
      + lambda_clash * differentiable clash penalty (optional, small)

Loss weight와 단위를 config에 기록한다. Position과 rotation scale 차이를 정규화 없이 더하지 않는다. Generated/proposed future frame에서 geometry metric을 계산한다.

History는 모든 arm에 동일하게 제공한다. `history_length=1` 진단은 config로 가능하지만 기본 ablation은 2로 고정한다.

공정성 조건:

- 동일 pair manifest
- 동일 seed
- 동일 batch order
- 동일 optimizer/schedule/max steps
- 동일 transition backbone
- 동일 `d_cond`
- arm별 trainable parameter count 기록
- frozen Phase 1이 기본
- validation은 domain-macro와 frame/micro 값을 구분

`z_phys`는 raw input에서 결정되는 feature이므로, C가 A보다 좋아져도 "새로운 물리 관측을 추가했다"고 쓰지 않는다. "force-supervised representation/pretraining이 transition optimization 또는 generalization을 개선했다"고만 해석한다.

반드시 테스트할 것:

- 모든 arm의 output shape 동일
- global transform equivariance/invariance
- future GT를 input conditioner에서 읽지 않음
- lag embedding을 바꾸면 condition이 바뀌지만 tensor contract는 유지
- 한 batch overfit smoke에서 loss가 감소
- frozen Phase 1 parameter 불변

════════════════════════════════════════
9. Checkpoint 6 — Training, ablation runner와 보고서
════════════════════════════════════════

다음 config/entry point 역할을 저장소 규칙에 맞게 구현한다.

- smoke config: 수 개 domain, 적은 pair, 짧은 step
- short-probe config: 1 ns/4 ns, bounded steps
- full Phase 1.5 config: 구현만 제공하고 자동 장시간 실행하지 않음
- 단일 arm training CLI
- A–E ablation runner
- evaluation-only CLI
- resume/checkpoint

권장 seed는 `[0, 1, 2]`지만 Checkpoint 6에서는 seed 0 smoke와 bounded sanity run만 자동 실행한다. full 3-seed experiment는 예상 pair 수, step 수, 시간과 명령을 보고한 뒤 사용자가 실행 여부를 결정하게 한다.

DDP를 지원해야 한다면 기존 Phase 1 training infrastructure를 재사용한다.

- normalization/statistics는 rank 간 일관
- validation aggregation은 metric 정의를 보존
- rank별 sample count를 고려한 weighted aggregation
- checkpoint atomic write
- config/split/manifest/checkpoint hash 저장

최종 결과 파일은 최소한 다음 field를 가진 tidy CSV/JSON으로 남긴다.

- arm
- seed
- lag_ns
- split
- step/checkpoint
- parameter_count
- trainable_parameter_count
- domain_count
- pair_count
- Cα RMSD
- translation RMSE
- rotation geodesic error
- pair-distance/contact metric
- clash rate
- training/validation loss

`docs/phase1_5_design.md`와 `docs/phase1_5_report.md`를 작성한다.

보고서에는 다음 표가 있어야 한다.

| Arm | 1 ns 주요 지표 | 4 ns 주요 지표 | baseline 대비 변화 | 해석 |
|---|---:|---:|---:|---|

실행하지 않은 full experiment 결과를 만들어내지 않는다. Smoke 결과와 full 결과를 명확히 구분하고 미실행 항목은 `pending`으로 쓴다.

════════════════════════════════════════
10. 해석을 위한 decision gate
════════════════════════════════════════

결과를 다음 규칙으로 해석하되 단일 seed의 작은 차이를 결론으로 만들지 않는다.

1. `physics_latent`가 `structure_only`보다 1 ns와 4 ns에서 일관되게 우수
   - 현재 Phase 1 representation을 유지하고 full Phase 2로 확장할 근거

2. `force_torque`는 효과 없고 `physics_latent`는 우수
   - residue 단순 합산보다 learned equivariant latent가 적절함

3. `force_pattern_shape`가 `physics_latent`보다 우수
   - explicit force moment/shape conditioner를 Phase 2 interface에 채택

4. `oracle_force`도 `structure_only`보다 우수하지 않음
   - 1–4 ns transition에 instantaneous force를 직접 조건화하는 가치가 낮을 가능성
   - history/stochastic memory 중심으로 Phase 2 재설계 고려

5. `oracle_force`는 우수하지만 predicted-force arm은 우수하지 않음
   - force 정보 자체는 유용하지만 Phase 1 prediction 또는 representation bottleneck이 문제

6. train은 좋아지고 held-out domain은 좋아지지 않음
   - capacity/overfitting 가능성, physics contribution으로 주장 금지

정량적인 go/no-go는 3 seeds와 domain-level confidence interval을 본 뒤 결정한다. 임의의 1% 차이를 성공으로 선언하지 않는다.

════════════════════════════════════════
11. 최종 품질 기준
════════════════════════════════════════

완료 조건:

- 기존 Phase 1 tests가 모두 통과
- 새 lag-pair, geometry, conditioner, leakage, equivariance tests 통과
- smoke에서 A–E가 동일 manifest와 output contract로 실행
- 적어도 한 batch overfit 또는 bounded sanity run에서 optimization 정상
- production arm이 GT force/future frame을 input으로 읽지 않음
- 1 ns/4 ns가 raw trajectory frame 기준으로 정확함
- 코드와 문서에서 Phase 1.5를 full stochastic Phase 2로 과장하지 않음
- Phase 1 checkpoint가 변하지 않음
- README 또는 실행 문서에 재현 명령 포함

각 Checkpoint에서 실패한 테스트를 숨기거나 삭제하지 마라. 원인을 수정하고 다시 실행하라. 기존 unrelated 사용자 변경은 보존한다. git commit/push는 요청받기 전까지 하지 않는다.

이제 Checkpoint 0만 수행하라. 파일은 수정하지 말고, 저장소 조사 결과와 실제 구현 계획을 보고한 뒤 멈춰라.
```

## 사용 순서

1. 저장소 루트에서 Claude Code Opus를 실행한다.
2. 위 코드 블록 전체를 첫 프롬프트로 입력한다.
3. Checkpoint 0 보고에서 실제 클래스·경로·checkpoint contract가 보고서와 맞는지 확인한다.
4. 문제가 없으면 `다음`으로 Checkpoint 1부터 순차 진행한다.
5. Checkpoint 6의 smoke 결과를 확인하기 전에는 full multi-seed 학습을 시작하지 않는다.
