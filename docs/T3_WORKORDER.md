# T3 작업 지시서 — Claude Code 용

> **작성 2026-08-05.** Phase 3(베이스라인 구성)을 Claude Code 로 진행하기 위한 순서서.
> 전체 맥락은 `PROJECT_DIRECTION.md`, 작업 목록은 `TASKS.md`, 측정 규칙은 `docs/v2/PROTOCOL.md` §1~§13.
>
> **아래 §0 을 통째로 복사해 Claude Code 첫 메시지로 붙여넣는다.**

---

## 0. 붙여넣을 프롬프트

```
저장소 C:\Users\sch\PycharmProjects\IEEE_sensors 에서 Phase 3(T3) 작업을 한다.

[연구 방향 — 이것부터]
논문(IEEE Sensors Letters, 8월 말 목표)의 범위는
**embedding vector 기반 eye blink detection + edge device 실시간 성능 검증**이다.
신원 제거·복원 방지·프라이버시는 범위 밖이다(특허·후속 연구로 분리됨).
문서 어딘가에 "최종 목표는 인코더 단 신원 재식별 불가능"이 남아 있으면 낡은 것이다.

[먼저 읽을 것 — 이 순서로]
1. docs/T3_WORKORDER.md   ★ 이 작업의 순서·명령·가드레일. 여기부터
2. docs/TASKS.md §Phase 3 · §9 · §10
3. docs/EXPERIMENT_PLAN.md §1(비교 대상) · §5(지표)
4. docs/v2/PROTOCOL.md §1~§13  ※ §14·§15 는 범위 밖이다. 게이트로 적용하지 마라

[작업 순서 — 이 순서를 바꾸지 마라]
T3-8  sym16 절제 실험      코드 수정 0. 가장 먼저 GPU 에 던진다
T3-5  D 스윕               코드 수정 0
T3-1  image_cnn 대조군     코드 작성 필요. 가장 오래 걸린다
T3-6  확정 재실행          위 셋의 결과로 최종 설정을 정한 뒤 마지막에

[🔴 절대 규칙]
1. 탐색 런에는 --save-models 를 **절대 붙이지 마라**.
   MODELS = "models/v2" 가 하드코딩이라 기존 확정 체크포인트를 덮어쓴다.
   (이 프로젝트는 이미 게이트가 증거를 두 번 지운 사고가 있었다)
2. --out 은 런마다 반드시 다르게 준다. 같으면 results/v2/train_encoder.json 을 덮어쓴다.
   점수 사이드카는 --out 에서 파생되므로 --out 만 다르면 안전하다.
3. 격자·시드·분할·프로브는 src/v2/common/ 하나만 쓴다. 스크립트가 직접 구현 금지.
4. 새 스크립트를 짜기 전에 같은 일을 하는 스크립트가 이미 있는지 먼저 확인해라.
5. val 에서 고르고 val 에서 보고하지 마라.
6. 탐색(fold 0·1) 숫자를 확정 값으로 인용하지 마라.
7. 실행 전에 git 커밋해라. src/v2/ 가 대부분 미추적이라 결과 JSON 의 git_commit 이
   실행 코드를 지목하지 못한다.

[출력 규칙]
- 칭찬하지 마라. 동의 여부보다 무엇이 틀릴 수 있는지를 말해라
- 추측 금지. 확인된 것만 쓰고 불확실하면 "미확인"이라고 명시해라
- 모든 수치에 출처 파일 경로를 병기해라
- 마지막 줄에: 가장 확신 없는 지점 1개

먼저 docs/T3_WORKORDER.md 를 읽고, §1 사전 점검부터 시작해라.
```

---

## 1. 사전 점검 (실행 전 5분)

```bash
cd C:\Users\sch\PycharmProjects\IEEE_sensors

# 1-1. 커밋 — 결과 JSON 이 실행 코드를 지목할 수 있어야 한다
git status
git add -A && git commit -m "T3 시작 전 스냅샷"

# 1-2. 확정 체크포인트 백업 (덮어쓰기 사고 대비)
#      PowerShell: Copy-Item -Recurse models\v2 models\v2_confirmed_20260805
cp -r models/v2 models/v2_confirmed_20260805

# 1-3. 데이터 무결성
python -c "import numpy as np; d=np.load('data/processed/v2/index.npz'); print('events', len(d['e_is_blink']))"
#   -> events 28728
python -c "import numpy as np; a=np.load('data/processed/v2/frames_m22.npy',mmap_mode='r'); print(a.shape)"
#   -> (532109, 64, 160)

# 1-4. GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

> ❌ 하나라도 실패하면 멈춘다.

---

## 2. T3-8 — sym16 절제 실험 (코드 수정 0)

### 왜 하는가

`src/v2/model/encoder.py` 는 *"비대칭 stride 가 이 설계의 핵심이다"* 라고 적고 있고,
이것이 논문 Method 의 **가장 독자적인 주장**이다. 그런데 근거가 **계산 논거뿐이다**:

> 눈꺼풀 상하 간격 8.73 px → 세로 stride 16 이면 **0.55 px** 가 되어 신호가 사라진다

**`grep -rl "sym16" results/` 결과 0건 — 한 번도 학습된 적이 없다.**
심사자가 반드시 묻는다: *"대칭 stride 를 실제로 돌려봤나?"*

| 구조 | 특징맵 | 세로 stride | 병목 눈꺼풀 | conv MMAC | PR-AUC |
|---|---|---:|---:|---:|---|
| sym16 | 128×4×10 | 16 | **0.55 px** | 9.22 | **미측정** |
| **vpres** (채택) | 64×32×5 | 2 | 4.37 px | **12.41** | 0.9886 ± 0.0038 |
| vfull | 64×64×5 | 1 | 8.73 px | 24.21 | 미측정 |

**vpres 는 sym16 보다 conv 연산이 34% 많다.** 그 비용을 정당화하는 숫자가 필요하다.

### 명령

```bash
# sym16 — fold 0·1 × 3 seed = 6런
python -m src.v2.train_encoder --arch sym16 --latent 16 \
    --folds 0 1 --seeds 0 1 2 \
    --out results/v2/abl_sym16.json

# vfull — 여유 되면 (비용 상한 확인용)
python -m src.v2.train_encoder --arch vfull --latent 16 \
    --folds 0 1 --seeds 0 1 2 \
    --out results/v2/abl_vfull.json

# vpres 재현 기준선 — 같은 fold·seed 로 비교해야 공정하다
#   ⚠️ 확정 런(5fold×3seed)의 0.9886 과 직접 비교하지 마라. fold 수가 다르다
python -m src.v2.train_encoder --arch vpres --latent 16 \
    --folds 0 1 --seeds 0 1 2 \
    --out results/v2/abl_vpres_ref.json
```

- `--save-models` **없음** ← 필수
- 점수 사이드카는 `results/v2/abl_*_scores/` 에 자동 저장된다

### 완료 조건

- 세 구조의 (PR-AUC ± , conv MMAC, 병목 눈꺼풀 px) 표
- **판정**: `abl_vpres_ref` − `abl_sym16` 의 짝지은 피험자 부트스트랩 95% CI
- ⚠️ **sym16 이 비슷하게 나오면 설계 주장을 바꿔야 한다.** 9.22 MMAC 짜리가 edge 에
  유리하므로 오히려 sym16 채택이 맞을 수 있다. **결과를 보기 전에 이 가능성을 적어둔다**

### 예상 소요

6런 × 약 18분 = **약 2시간** (확정 런 15런/266분 기준 환산)

---

## 3. T3-5 — embedding vector 차원 D 확정 (코드 수정 0)

### 왜 하는가

D=16 은 v1 스윕(8명·한쪽눈, **인용 금지**)에서 온 값이고, "작은 D 가 신원을 덜 흘린다"는
기대도 있었으나 **실측으로 기각**됐다(512→32 로 16배 좁혀도 선형 재식별 0.9992→0.9947).
프라이버시가 범위 밖이 된 지금 **D 를 작게 유지할 근거가 남아 있지 않다.**

그리고 **D 를 키우는 비용이 거의 0이다** (`E.analyse()` 실측):

| D | conv MMAC | fc MMAC | 총 MMAC | fc 비중 |
|---:|---:|---:|---:|---:|
| 8 | 12.41 | 0.016 | 12.43 | 0.13% |
| **16** | 12.41 | 0.033 | **12.44** | 0.26% |
| 32 | 12.41 | 0.066 | 12.48 | 0.53% |
| 64 | 12.41 | 0.131 | 12.54 | 1.05% |

### 명령

```bash
for D in 8 32 64; do
  python -m src.v2.train_encoder --arch vpres --latent $D \
      --folds 0 1 --seeds 0 1 2 \
      --out results/v2/dim_D${D}.json
done
# D=16 은 T3-8 의 abl_vpres_ref.json 을 그대로 쓴다 (같은 fold·seed)
```

PowerShell:
```powershell
foreach ($D in 8,32,64) {
  python -m src.v2.train_encoder --arch vpres --latent $D `
      --folds 0 1 --seeds 0 1 2 `
      --out "results/v2/dim_D$D.json"
}
```

### 완료 조건

- (D, PR-AUC ± , 파라미터 수, 총 MMAC, 예상 ONNX 크기) 표
- 선택 근거 1문장
- 💡 **성능이 같으면 작은 D 를 고른다.** "16차원 벡터만으로 된다"가 논문에 더 유리하고
  모델 크기·메모리에도 좋다. 뚜렷이 오를 때만 키운다

### 예상 소요

9런 = **약 3시간**

---

## 4. T3-1 — image_cnn 대조군 (코드 작성 필요) 🔴 논문의 핵심

### 왜 하는가

전사문 03:02 — *"내세우는 거는 이미지 기반으로 하는 애들하고 성능 비교를 해요."*
**이게 없으면 "임베딩이 이미지보다 낫다/비슷하다"를 어떤 형태로도 쓸 수 없다.**

### 설계 — 기존 구조를 최대한 재사용한다

`src/v2/model/encoder.py` 는 이미 잘 분리돼 있다:

| 함수 | 역할 |
|---|---|
| `build(arch, d_latent)` | 크롭 → D차원 벡터 (**ours**) |
| `build_ear_frontend(k_in, d_latent)` | EAR 스칼라 → D차원 (**대조군 1**) |
| `build_head(d_latent, t, hidden)` | D차원 × 19프레임 → blink/unblink (**공용 헤드**) |

→ **프론트엔드만 갈아끼우면 헤드는 그대로 쓴다.** 공정 비교가 구조적으로 보장된다.

### 할 일

1. `src/v2/model/encoder.py` 에 **`build_image_cnn(variant, d_latent)`** 추가
   - `variant="free"` — 크기 제한 없이 합리적으로 설계. 성능 천장
   - `variant="matched"` — 총 파라미터 수(또는 MMAC)를 `vpres + 선택된 D` 와 **±10% 이내**
   - ⚠️ **병목을 없애는 게 요점이 아니다.** image_cnn 도 프레임당 D차원을 내야
     같은 헤드를 쓸 수 있다. 차이는 **"학습된 표현을 거치는가"** 가 아니라
     **"크롭 전체를 판정에 직접 노출하는가"** 로 잡는다.
     → 구현 방향은 Claude Code 가 제안하고 **사람이 승인한 뒤** 진행할 것.
     이 부분은 설계 판단이 필요하므로 임의로 결정하지 마라
2. `SPECS` 에 image_cnn 구조 추가 또는 별도 dict
3. `analyse()` 가 image_cnn 도 다룰 수 있게 확장 (MMAC 비교에 필요)
4. `train_encoder.py` 에 `--front {ours,image_cnn_free,image_cnn_matched}` 분기 추가
   - ⚠️ **기존 `--arch` 와 혼동하지 마라.** `--arch` 는 인코더 구조(vpres/sym16/vfull),
     `--front` 는 프론트엔드 종류다. 이름을 헷갈리면 나중에 결과 해석이 꼬인다

### 명령 (코드 작성 후)

```bash
# 탐색 — fold 0·1 × 3 seed 로 먼저 확인
python -m src.v2.train_encoder --front image_cnn_free --latent <선택D> \
    --folds 0 1 --seeds 0 1 2 --out results/v2/imgcnn_free_pilot.json
python -m src.v2.train_encoder --front image_cnn_matched --latent <선택D> \
    --folds 0 1 --seeds 0 1 2 --out results/v2/imgcnn_matched_pilot.json

# 확정 — 5 fold × 3 seed
python -m src.v2.train_encoder --front image_cnn_free --latent <선택D> \
    --out results/v2/imgcnn_free.json
python -m src.v2.train_encoder --front image_cnn_matched --latent <선택D> \
    --out results/v2/imgcnn_matched.json
```

### 완료 조건

- 5 fold × 3 seed, ours 와 **동일 분할·격자·중단 규칙**
- PR-AUC + 짝지은 피험자 부트스트랩 CI
- **파라미터 수 · MMAC 을 같은 표에 기록** (이게 없으면 matched 의 의미가 없다)

### 예상 소요

코드 1~2일 + 런 **약 9시간** (2변형 × 15런)

---

## 5. T3-6 — 확정 재실행 (마지막에)

### 왜 하는가

두 가지가 현재 불가능하다.

1. **서브그룹 풀링 + δ 판정(ear_head 상대)** — 확정 런이 ear_head 가중치를 저장하지 않았다.
   실제로 `models/v2/fold0_seed0/` 에는 `encoder.pt`, `head.pt` 뿐이고
   `earhead_front.pt`, `earhead_head.pt` 가 **없다**. 사이드카 패치는 코드에 적용돼
   있으므로 재실행하면 얻어진다
2. **Recall · F1 미계산** — 저장 지표가 `accuracy · precision · pr_auc · roc_auc · thr` 뿐이다.
   논문 Table I 에 필요하다 → `run_fold` 의 지표 계산부에 추가

### 순서

1. T3-8·T3-5·T3-1 결과로 **최종 arch·D 를 확정**한다
2. Recall·F1 계산을 코드에 추가한다
3. **백업 확인 후** `--save-models` 로 확정 런을 돌린다

```bash
# ⚠️ 이 런은 models/v2/ 를 덮어쓴다. §1-2 백업이 있는지 먼저 확인
ls models/v2_confirmed_20260805

python -m src.v2.train_encoder --arch <확정arch> --latent <확정D> \
    --save-models \
    --out results/v2/train_encoder_final.json
```

### 완료 조건

- `models/v2/fold*_seed*/` 에 `earhead_front.pt`, `earhead_head.pt` 가 생겼는가
- 런 지표에 `recall`, `f1` 이 포함됐는가
- `posthoc_subgroups.py` 를 재실행해 **이벤트 가중 풀링 + ear_head 상대 δ 판정**이 나오는가

### 예상 소요

**약 4.5시간**

---

## 6. 실행 큐 요약

```
[사람]  §1 사전 점검 (커밋 + 백업)          5분
   ↓
[GPU]   T3-8  sym16 / vfull / vpres_ref     약 2h   ← 오늘 바로 던질 수 있다
   ↓                                                  (코드 수정 0)
[GPU]   T3-5  D ∈ {8, 32, 64}               약 3h
   ↓
[사람]  T3-8·T3-5 결과 검토 → arch·D 확정
[사람]  T3-1 image_cnn 설계 제안 → 승인      1~2일
   ↓
[GPU]   T3-1  image_cnn 2변형                약 9h
   ↓
[사람]  Recall·F1 계산 추가
   ↓
[GPU]   T3-6  확정 재실행 (--save-models)    약 4.5h
```

> **GPU 가 도는 동안 사람은 T6-1~T6-3 관련 연구 조사와 T5-3(v2 ONNX export) 코드를 한다.**
> 8월 말 마감이라 병렬화가 필수다 (`TASKS.md` §10).

---

## 7. 미뤄둔 것 (T3 안이지만 P1 이하)

| # | 작업 | 사유 |
|---|---|---|
| T3-2 | random projection 대조군 | image_cnn 이 우선. 여유 되면 fold 0·1 로만 |
| T3-3 | 평균 벡터 대조군 (정보 0 하한) | 위와 동일 |
| T3-4 | 랜덤 초기화 encoder 를 깜빡임 축에서 측정 | 위와 동일 |
| T3-7 | 기존 경량 모델(B8) 재현 | T6-1 조사 결과를 보고 판단 |

---

## 8. 판단이 필요하면 멈출 것

다음은 **임의로 결정하지 말고 사람에게 물어라.**

| # | 항목 |
|---|---|
| 1 | `image_cnn` 의 구체적 구조 (층 수·채널·병목 유무) |
| 2 | `matched` 를 파라미터 수로 맞출지 MMAC 으로 맞출지 |
| 3 | sym16 이 vpres 와 비슷하게 나왔을 때 어느 쪽을 채택할지 |
| 4 | D 스윕에서 성능과 크기가 상충할 때의 우선순위 |
| 5 | 확정 런의 arch·D 최종 선택 |
