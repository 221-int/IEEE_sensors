# CLEANUP — 파일 정리 감사

> **작성 2026-08-05.** 저장소 전체를 스캔해 삭제 후보를 등급별로 분류했다.
> **아직 아무것도 삭제하지 않았다.** 실행 스크립트는 §실행 스크립트 참조.
>
> 현재 총 용량: **468 GB** (그중 `data/mEBAL2/*.zip` 이 **442 GB**)

---

## 🔵 2026-08-05 결정

| 등급 | 결정 |
|---|---|
| **A** 임시·빌드 산출물 | ✅ **삭제** |
| **B** v1 자산 (모델·중간 데이터) | ✅ **삭제.** 단 **eyeblink8 클립 9 는 보존** (Pi 640×480 측정용) |
| **C** 원본 데이터셋 아카이브 442 GB | 🔴 **건드리지 않는다.** 재추출 가능성을 남긴다 |
| **D** 삭제 금지 목록 | 🔴 **유지** |

**예상 회수: 약 6.1 GB** (468 GB → 약 462 GB).
용량의 대부분은 C 이므로 디스크가 크게 줄지는 않는다. 이번 정리의 목적은
**용량 확보가 아니라 "낡은 자산이 실수로 인용·재사용되는 것을 막는 것"** 이다.

---

## 요약

| 등급 | 내용 | 회수 용량 | 결정 |
|---|---|---|---|
| **A** | 임시·빌드 산출물. 되돌릴 필요 없음 | 약 **5.1 GB** | ✅ 삭제 |
| **B** | v1 자산. 논문 인용 금지이고 v2 에서 재사용 안 함 | 약 **1.0 GB** | ✅ 삭제 |
| **C** | 원본 데이터셋 아카이브 | 442 GB | 🔴 **보존** |
| **D** | 🔴 **삭제 금지.** 지우면 실험이 깨진다 | — | 🔴 유지 |

---

## A. 안전하게 삭제 가능 (약 5.1 GB)

| # | 경로 | 크기 | 사유 |
|---|---|---|---|
| A1 | `data/processed/v2/frames_m18.npy` | **5.1 GB** | MARGIN **2.2 확정 🔒** (PROTOCOL §8-bis). m18 은 비교용이었고 검증(merge V6)이 끝났다. **필요하면 `shards/*.npz` 의 `frames_m18` 키에서 재생성 가능** |
| A2 | `temp_ppt_build/` | 3.9 MB | 랩미팅 PPT 빌드 산출물(슬라이드 jpg·png·node 스크립트). 최종 `.pptx` 는 `docs/` 에 있다 |
| A3 | `tmp/` | 2.2 MB | 임시 pdf |
| A4 | `data/_legacy_public/talkingFace/` | 23 MB | **코드 어디서도 참조하지 않는다** (grep 확인). v1·v2 실험에 미사용 |
| A5 | `docs/랩미팅_2026-08-05_v2/` | 724 KB | pptx 에서 뽑은 슬라이드 png. 원본 pptx 가 있다 |
| A6 | `docs/랩미팅_2026-08-05_v2.pptx.inspect.ndjson` | 40 KB | pptx 검사 로그 |
| A7 | `__pycache__/` 11개 디렉터리 | 소량 | `.gitignore` 대상 |
| A8 | `.idea/` | 16 KB | PyCharm 설정. `.gitignore` 대상 (팀 공유 불필요) |
| A9 | `benchmark/` | 0 | `.gitkeep` 만 있는 빈 폴더. 쓰이지 않는다 |

> `eye_dataset/` 은 **이미 삭제됨** (2026-08-05 확인).

---

## B. v1 자산 — 삭제 검토 (약 730 MB)

v2 는 mEBAL2 단독이고, PROTOCOL §13 이 **v1 수치의 논문 인용을 금지**한다.
다만 `results/*.json`(v1) 은 작고 `CHANGELOG.md` 의 근거이므로 **남긴다.**

| # | 경로 | 크기 | 사유 | 위험 |
|---|---|---|---|---|
| B1 | `models/sweep/` | **139 MB** | v1 차원 스윕 인코더 15개. **8명·한쪽눈이라 인용 금지**. D 확정은 T3-5 에서 v2 로 새로 한다 | 없음 |
| B2 | `data/processed/parts_both/` | 220 MB | v1 Eyeblink8 중간 분할 산출물 | 없음 |
| B3 | `data/processed/eyeblink8_eyes.npz` | 219 MB | v1 학습 데이터. v2 는 mEBAL2 단독 | 재생성하려면 eyeblink8 원본 필요 (D2 로 보존됨) |
| B4 | `data/processed/parts/` | 83 MB | v1 중간 산출물 (한쪽눈 시절) | 없음 |
| B5 | `models/autoencoder/` | 29 MB | v1 오토인코더 (128차원). v2 인코더는 구조가 다름 | `models/onnx/` 재생성 불가 → B6 과 함께 지운다 |
| B6 | `models/disentangled/` | 29 MB | v1 GRL PoC. **한쪽눈 64×64 예비라 인용 금지** | 후속 연구는 **설계만 참고**하고 코드를 새로 쓴다 (`PATENT_AND_FUTURE_WORK.md` §4-1) |
| B7 | `models/onnx/` | 14 MB | **v1 배포 아티팩트.** v2 는 T5-3 에서 새로 export 한다 | 지우면 v1 Pi 재현 불가. 어차피 인용 금지 |
| B8 | `models/_diag_s4/`, `models/classifier/` | 76 KB | v1 판정기 (문서에 "이상치, 교체 대상") | 없음 |
| B9 | `models/privacy_metrics.json` | 0 | **날짜·출처 불명. 인용 금지**로 이미 표시됨 | 없음 |
| B10 | `data/processed/eyeblink8_eyes_labels.csv` | 824 KB | v1 라벨 | 없음 |

> ⚠️ **B3·B5·B7 을 지우면 v1 파이프라인이 실행 불가능해진다.** 새 방향에서 v1 을
> 다시 돌릴 계획은 없지만, 되돌릴 수 없다는 점은 알고 지운다.
> 코드(`src/dataset/`, `src/encoder/`, `src/eval/`, `src/experiments/`)는 **남긴다** —
> 용량이 작고 후속 연구가 설계를 참고한다.

---

## C-0. 🔴 "학습을 처음부터 다시 시켜야 하지 않나?" — 코드로 확인한 답

**결론: 재학습은 zip 을 필요로 하지 않는다. 단 zip 은 "크롭 재추출"의 유일한 원천이 맞다.**

### 의존 관계 (코드 실측)

```
data/mEBAL2/Processed_Data.zip     ─┐
  (landmarks.csv, box.csv)          │
data/mEBAL2/Webcams-EEG *.zip      ─┤→ phase2_extract.py → 크롭 추출
  (color.mp4)                       │        ↓
data/raw/mEBAL2/_probe/            ─┘   phase3_merge.py
  (58명 blink/unblink 라벨)                 ↓
                            data/processed/v2/{shards, frames_m22.npy, index.npz}
                                            ↓
                            train_encoder.py ← **여기서부터는 zip 을 안 본다**
```

근거 (`src/v2/`):

| 파일 | zip 참조 |
|---|---|
| `phase2_extract.py` | `PD_ZIP = "data/mEBAL2/Processed_Data.zip"` / `WEBCAM_ZIPS = "data/mEBAL2/Webcams-EEG *.zip"` — **여기만 zip 을 연다** |
| `train_encoder.py` | `np.load(data_dir/"index.npz")` + `np.load(data_dir/f"frames_{tag}.npy", mmap_mode="r")` — **zip 참조 0건** |
| `phase5_reid.py`, `posthoc_subgroups.py`, `diagnose_subject.py` | 동일. `data/processed/v2/` 만 본다 |

### 앞으로 할 작업 중 zip 이 필요한 것

| 작업 | zip 필요? |
|---|---|
| T3-1 image_cnn 학습 (2변형 × 15런) | ❌ 같은 크롭을 읽는다 |
| T3-5 D 스윕 | ❌ |
| T3-6 재실행 (ear_head 가중치 + Recall·F1) | ❌ |
| T3-2·T3-3 random projection · 평균 벡터 | ❌ |
| T4-1 광학 층화 | ❌ 이미 측정된 photometrics 사용 |
| T4-4~T4-8 합성 열화 | ❌ 기존 크롭에 변형을 가한다 |
| T5-x 경량화 · Pi 5 측정 | ❌ `User 1/color.mp4` 가 이미 풀려 있다 |
| **민감도 S7 (U18·좌석 오염 되살리기)** | ❌ `index_pre_flags.npz` 로 재추출 없이 가능 |

**8월 말 로드맵 전체에서 zip 이 필요한 작업은 0건이다.**

### 그럼에도 zip 이 유일한 원천인 것 🔴

`data/raw/` 에 풀려 있는 것을 실제로 세어 보면:

| 항목 | 상태 |
|---|---|
| blink / unblink 라벨 CSV | ✅ **58명분 전부** (`_probe/` 에 116개) |
| `PD_landmarks.csv`, `PD_box.csv` | ⚠️ **각 1개뿐 — User 1 것만이다** |
| `color.mp4` | ⚠️ **User 1 것만** (8.3 GB) |

즉 **나머지 57명의 랜드마크·박스·영상은 zip 안에만 있다.**
크롭을 다시 뽑아야 하는 상황이 오면 `Processed_Data.zip`(32.7 GB) **과**
`Webcams-EEG *.zip`(434 GB) **둘 다** 있어야 한다. 하나만 남겨서는 의미가 없다.

### 크롭을 다시 뽑아야 하는 경우

1. 크롭 배율을 2.2 에서 바꾼다 → **MARGIN 2.2 는 🔒 확정** (PROTOCOL §8-bis)
2. 크롭 해상도를 64×160 에서 바꾼다 → 확정
3. 눈 정렬 방식(회전 정렬)을 바꾼다
4. 심사자가 다른 전처리 조건을 요구한다
5. User 1 이외 사용자의 **원본 프레임** 그림이 필요하다 (크롭으로는 안 되는 경우)

1~3 은 확정 사항이라 가능성이 낮다. **4~5 가 zip 을 남기는 실질적 이유다.**

> 🔵 **그래서 2026-08-05 에 "zip 은 건드리지 않는다"로 결정했다.**
> 442 GB 를 심사 대응 보험료로 지불하는 선택이며, 근거 있는 판단이다.
> 다만 **재학습 때문에 필요한 것은 아니다** — 그 이유로 남긴 거라면 오해였다.

---

## C. 🔴 원본 데이터셋 아카이브 — **보존 결정 (2026-08-05)**

> 🔵 **결정: 건드리지 않는다.** 아래 분석은 나중에 다시 판단할 때를 위한 기록이다.
> 442 GB 를 그대로 두는 대가로 **크롭 규격 재추출 가능성**을 남긴다.
> 심사자가 다른 전처리 조건을 요구하거나 크롭 배율을 바꿔야 할 때 대응할 수 있다.
>
> ⚠️ 대신 **디스크 여유를 별도로 확인해 둘 것.** 이번 정리로 회수되는 것은 약 6 GB 뿐이다.

### (참고) 삭제를 고려한다면 — 아래는 보류된 분석

| 파일 | 크기 |
|---|---|
| `data/mEBAL2/Webcams-EEG 1-15.zip` | 118 GB |
| `data/mEBAL2/Webcams-EEG 46-58.zip` | 112 GB |
| `data/mEBAL2/Webcams-EEG 16-30.zip` | 104 GB |
| `data/mEBAL2/Webcams-EEG 31-45.zip` | 100 GB |
| `data/mEBAL2/Processed_Data.zip` | 32.7 GB |
| `data/mEBAL2/Blinks-Unblinks.zip` | 6.3 GB |
| **합계** | **442 GB** |

### 삭제해도 되는 근거

`docs/PROJECT.md` §8 은 *"원본 아카이브 441 GB. **크롭 검증 끝날 때까지 삭제 금지**"* 라고
적었다. **그 조건은 충족됐다.**

- 크롭 추출 완료: 532,109장, 57명 → `data/processed/v2/shards/*.npz`
- 병합 검증 **8/8 PASS** → `results/v2/merge_report.json`
- 크롭 배율 **2.2 확정 🔒**, 결측 정책·입력 정규화 전부 확정
- 라벨은 `data/raw/mEBAL2/_probe/` 에 58명분이 이미 추출돼 있다 (916 MB)
- Pi 측정용 720p 영상은 `data/raw/mEBAL2/User 1/` 에 이미 있다 (8.3 GB)

즉 **논문에 필요한 것은 전부 추출이 끝났다.**

### 삭제하면 잃는 것

1. **크롭 규격을 다시 바꾸면 재추출 불가.** 배율·해상도·눈 정렬 방식을 바꾸려면
   원본 영상이 다시 필요하다 → **441 GB 재다운로드** (며칠 소요, 라이선스 재신청 가능성)
2. **User 1 이외의 원본 영상**을 다시 볼 수 없다. 실패 사례를 원본 프레임으로
   확인하고 싶을 때 제약이 생긴다 (크롭으로는 가능)
3. 심사자가 다른 전처리 조건을 요구하면 대응 불가

### 판단

**8월 말 마감**이고 MARGIN 2.2 가 🔒 확정이므로 재추출 가능성은 낮다.
다만 되돌릴 수 없으므로 **단계적 삭제**를 권한다.

| 단계 | 대상 | 회수 | 권장 |
|---|---|---|---|
| **C1** | `Webcams-EEG *.zip` 4개 | **434 GB** | ✅ **권장.** 용량의 98%. User 1 영상은 이미 추출돼 있다 |
| **C2** | `Processed_Data.zip` | 32.7 GB | 🟡 보류 가능. 무엇이 들었는지 미확인 |
| **C3** | `Blinks-Unblinks.zip` | 6.3 GB | 🟡 **남기는 것을 권함.** 라벨 원본이고 작다. `_probe/` 가 여기서 나왔다면 재추출 안전망 |
| — | `mEBAL2_description.pdf` | 549 KB | ✅ **남긴다.** 데이터셋 문서 |

> **C1 만 해도 434 GB 를 회수한다.** C2·C3 는 39 GB 라 급하지 않다.

---

## D. 🔴 삭제 금지

| 경로 | 크기 | 왜 필요한가 |
|---|---|---|
| `data/processed/v2/shards/` | 5.9 GB | **크롭의 단일 진리원.** m22·m18 이미지와 프레임별 메타데이터 전부 여기 있다 |
| `data/processed/v2/frames_m22.npy` | 5.1 GB | 학습이 쓰는 memmap. `train_encoder`·`diagnose_subject` 가 직접 연다 |
| `data/processed/v2/index.npz` | 36 MB | 이벤트 인덱스. **27개 위치에서 참조** |
| `data/processed/v2/index_pre_flags.npz` | 35 MB | 좌석 오염 필터 **이전** 인덱스. 민감도 S7 에 필요 |
| `data/raw/mEBAL2/_probe/` | 916 MB | **58명 blink/unblink 라벨 CSV.** 이게 없으면 아무것도 못 한다 |
| `data/raw/mEBAL2/User 1/` | 8.3 GB | ① 로더 회귀 테스트(`check_mebal2.py`) ② **Pi 720p 측정용 영상** (color.mp4, 1280×720 30fps 37,641프레임) |
| `data/raw/mEBAL2/glasses_labels_58*.csv` | 소량 | 안경 라벨 |
| **`data/_legacy_public/eyeblink8/eyeblink8/9/`** | **9.1 MB** | 🔴 **Pi 640×480 측정 클립!** `27122013_152435_cam.avi` = 640×480 30fps 5,183프레임. v1 이 이걸로 쟀고, **v2 재측정도 같은 조건이어야 인코더 변경 효과만 분리된다** (`EXPERIMENT_PLAN.md` §6-3) |
| `models/v2/` | 5.1 MB | 15런 체크포인트. 점수 복원·재채점에 필요 (T3-6) |
| `results/v2/` | 14 MB | **논문 표의 1차 증거** |
| `results/*.json` (v1) | 소량 | `CHANGELOG.md` 의 근거. 작아서 지울 이유 없음 |
| `results/pi_ours.{json,csv}` | 1.3 MB | v1 Pi 측정. **논문 인용은 금지지만 v2 와 비교할 내부 기준** |
| `src/` 전체 | 1.8 MB | ⚠️ **`src/dataset/capture_eye_dataset.py` 특히 주의** — `compute_ear` 를 제공하며 `src/eval/ear_baseline.py`·`src/deploy/run_video.py` 가 import 한다. 지우면 EAR 베이스라인과 Pi 하네스가 깨진다 |
| `docs/archive/` | 196 KB | 인용은 금지지만 **연구 기록 보존**. 작다 |
| `docs/figures/`, `docs/v2/figures/` | 3.2 MB + | 논문 그림 후보 · U1 실패 사례 시각화 |
| `paper/` | 28 KB | 논문 초안 |

### `data/_legacy_public/eyeblink8/` 부분 삭제

Pi 측정에 필요한 것은 **클립 9 하나(9.1 MB)** 뿐이다. 나머지 7개 클립(295 MB)은
v1 학습용이었으므로 삭제 가능하다.

| 유지 | 삭제 가능 |
|---|---|
| `eyeblink8/9/` (9.1 MB) — 🔴 Pi 480p 측정 클립 | `eyeblink8/{1,2,3,4,8,10,11}/` (295 MB) |

> ⚠️ **클립 9 를 지우면 v1 과 동일 조건 Pi 비교가 불가능해진다.**
> 그러면 640×480 측정은 다른 클립을 써야 하고, "인코더 변경 효과만 분리"라는 논거가 약해진다.

---

## 실행 스크립트 — 확정본 (2026-08-05)

> `data/mEBAL2/*.zip` (442 GB) 는 **건드리지 않는다.**
> 실행 전 `git status` 로 커밋할 것이 없는지 확인한다.

### PowerShell (Windows — 이 프로젝트의 기본 환경)

```powershell
cd C:\Users\sch\PycharmProjects\IEEE_sensors

# ---------- 0. 안전장치: Pi 측정 클립 2개가 있는지 먼저 확인 ----------
Test-Path "data\_legacy_public\eyeblink8\eyeblink8\9\27122013_152435_cam.avi"   # True 여야 함
Test-Path "data\raw\mEBAL2\User 1\RealSense\Color_Webcam\color.mp4"             # True 여야 함

# ---------- A. 임시·빌드 산출물 (약 5.1 GB) ----------
Remove-Item -Force  "data\processed\v2\frames_m18.npy"
Remove-Item -Recurse -Force temp_ppt_build, tmp, .idea, benchmark
Remove-Item -Recurse -Force "data\_legacy_public\talkingFace"
Remove-Item -Recurse -Force "docs\랩미팅_2026-08-05_v2"
Remove-Item -Force  "docs\랩미팅_2026-08-05_v2.pptx.inspect.ndjson"
Get-ChildItem -Recurse -Directory -Filter __pycache__ |
  Where-Object { $_.FullName -notmatch '\\\.git\\' } |
  Remove-Item -Recurse -Force

# ---------- B. v1 모델·중간 데이터 (약 730 MB) ----------
Remove-Item -Recurse -Force models\sweep, models\autoencoder, models\disentangled, models\onnx
Remove-Item -Recurse -Force models\_diag_s4, models\classifier
Remove-Item -Force  models\privacy_metrics.json
Remove-Item -Recurse -Force data\processed\parts, data\processed\parts_both
Remove-Item -Force  data\processed\eyeblink8_eyes.npz, data\processed\eyeblink8_eyes_labels.csv

# ---------- B+. eyeblink8 클립 7개 (295 MB) ----------
#   🔴 클립 9 는 남긴다 — Pi 640×480 측정용
foreach ($d in 1,2,3,4,8,10,11) {
  Remove-Item -Recurse -Force "data\_legacy_public\eyeblink8\eyeblink8\$d" -ErrorAction SilentlyContinue
}
```

### bash (WSL · macOS · Pi)

```bash
cd /path/to/IEEE_sensors

# ---------- 0. 안전장치 ----------
ls -la "data/_legacy_public/eyeblink8/eyeblink8/9/"*.avi
ls -la "data/raw/mEBAL2/User 1/RealSense/Color_Webcam/color.mp4"

# ---------- A. 임시·빌드 산출물 (약 5.1 GB) ----------
rm -f  "data/processed/v2/frames_m18.npy"
rm -rf temp_ppt_build tmp .idea benchmark
rm -rf "data/_legacy_public/talkingFace"
rm -rf "docs/랩미팅_2026-08-05_v2"
rm -f  "docs/랩미팅_2026-08-05_v2.pptx.inspect.ndjson"
find . -name "__pycache__" -type d -not -path "./.git/*" -exec rm -rf {} +

# ---------- B. v1 모델·중간 데이터 (약 730 MB) ----------
rm -rf models/sweep models/autoencoder models/disentangled models/onnx
rm -rf models/_diag_s4 models/classifier
rm -f  models/privacy_metrics.json
rm -rf data/processed/parts data/processed/parts_both
rm -f  data/processed/eyeblink8_eyes.npz data/processed/eyeblink8_eyes_labels.csv

# ---------- B+. eyeblink8 클립 7개 (295 MB) ----------
#   🔴 클립 9 는 남긴다 — Pi 640×480 측정용
for d in 1 2 3 4 8 10 11; do rm -rf "data/_legacy_public/eyeblink8/eyeblink8/$d"; done

# ---------- C. 원본 아카이브 — 실행하지 않는다 (2026-08-05 보존 결정) ----------
# rm -f "data/mEBAL2/Webcams-EEG "*.zip     # 434 GB — 🔴 보류
# rm -f "data/mEBAL2/Processed_Data.zip"    #  32.7 GB — 🔴 보류
# rm -f "data/mEBAL2/Blinks-Unblinks.zip"   #   6.3 GB — 🔴 보류
```

### 삭제되는 것 한눈에

```
data/processed/v2/frames_m18.npy            5.1 GB   ← 회수량의 84%
data/_legacy_public/eyeblink8/{1,2,3,4,8,10,11}/  295 MB
data/processed/parts_both/                  220 MB
data/processed/eyeblink8_eyes.npz           219 MB
models/sweep/                               139 MB
data/processed/parts/                        83 MB
models/autoencoder/                          29 MB
models/disentangled/                         29 MB
data/_legacy_public/talkingFace/             23 MB
models/onnx/                                 14 MB
temp_ppt_build/                             3.9 MB
tmp/                                        2.2 MB
docs/랩미팅_2026-08-05_v2/                  724 KB
그 외 (labels.csv, _diag_s4, classifier, .idea, benchmark, __pycache__)
────────────────────────────────────────────────────
합계                                       약 6.1 GB
```

## 삭제 후 확인 — 4개 전부 통과해야 한다

```bash
# 1) 학습 데이터가 온전한가
python -c "import numpy as np; d=np.load('data/processed/v2/index.npz'); print(len(d['e_is_blink']))"
#    → 28728

# 2) m22 memmap 이 살아 있는가 (학습이 이걸 연다)
python -c "import numpy as np; a=np.load('data/processed/v2/frames_m22.npy',mmap_mode='r'); print(a.shape)"
#    → (532109, 64, 160)

# 3) Pi 측정 클립 두 개가 살아 있는가
python -c "
import cv2
for p in ['data/_legacy_public/eyeblink8/eyeblink8/9/27122013_152435_cam.avi',
          'data/raw/mEBAL2/User 1/RealSense/Color_Webcam/color.mp4']:
    c=cv2.VideoCapture(p); print(p.split('/')[-1], int(c.get(3)),'x',int(c.get(4))); c.release()"
#    → 27122013_152435_cam.avi 640 x 480
#    → color.mp4 1280 x 720

# 4) EAR 베이스라인이 여전히 import 되는가
python -c "from src.dataset.capture_eye_dataset import compute_ear; print('ok')"
#    → ok
```

> ❌ **하나라도 실패하면 멈추고 무엇이 지워졌는지 확인한다.**
> 특히 3번 — Pi 클립을 잃으면 640×480 측정을 v1 과 같은 조건으로 할 수 없다.

## 삭제 후 갱신해야 할 문서

| 문서 | 갱신 내용 |
|---|---|
| `docs/PROJECT.md` §8 파일 구조 | 삭제된 항목 제거 (`models/sweep`, `models/autoencoder` 등) |
| `docs/PROGRESS.md` §6-5 | "보류" → "삭제됨" |
| `docs/PATENT_AND_FUTURE_WORK.md` §7 | v1 **모델 체크포인트가 사라졌음**을 명시. ⚠️ 코드(`train_disentangled.py`, `recon_attack.py`)는 남으므로 **설계 참고는 계속 가능** |
| `docs/EXPERIMENT_PLAN.md` §2 | `frames_m18` 삭제 반영 (필요 시 shards 에서 재생성) |
| `.gitignore` | `models/sweep/`, `models/autoencoder/*.pt`, `models/disentangled/*.pt` 항목 정리 |

> 이 갱신은 **삭제를 실제로 실행한 뒤에** 한다. 지금 미리 고치면 문서와 파일이 어긋난다.

---

## 정리하지 않은 것 — 알고 남긴다

| 항목 | 크기 | 왜 남기나 |
|---|---|---|
| `data/mEBAL2/*.zip` | **442 GB** | 🔵 2026-08-05 보존 결정. 크롭 재추출 가능성을 남긴다 |
| `.git/objects` | 391 MB | `results/` 와 `models/onnx/` 를 커밋해 왔기 때문. 히스토리 재작성은 위험 대비 이득이 적다 |
| `docs/archive/` | 196 KB | 인용 금지지만 연구 기록. 작다 |
| `results/*.json` (v1) | 소량 | `CHANGELOG.md` 의 근거 |
| `src/{dataset,encoder,classifier,eval,experiments,deploy,tools}` | 1.8 MB | v1 코드. **후속 연구가 설계를 참고**하고, `capture_eye_dataset.compute_ear` 는 현재도 쓰인다 |
| `docs/figures/` (v1 그림) | 3.2 MB | 후속 연구·특허 자료 |

> ⚠️ **`models/onnx/` 를 지우면서 `src/deploy/export_onnx.py` 는 남긴다.**
> 코드가 있으니 v1 ONNX 는 이론상 재생성 가능하지만, **`models/autoencoder/encoder.pt` 도
> 함께 지우므로 실제로는 재생성 불가**하다. v1 Pi 측정은 이제 `results/pi_ours.json`
> 기록으로만 남는다 — 어차피 논문 인용 금지이므로 문제되지 않는다.
