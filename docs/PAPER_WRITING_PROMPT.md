# 논문 집필 세션 프롬프트 — 아래 블록을 통째로 복사해 붙여넣는다

```
저장소 C:\Users\sch\PycharmProjects\IEEE_sensors 에서 IEEE Sensors Letters 논문을 쓴다.

[이번 세션의 범위]
paper/IEEE_Sensors_letters_overleaf.tex 를 **내용 전부 갈아엎는다.**
그 파일은 2026-07-21 자 v1 원고이고, 제목이 "Real-Time Fatigue Detection System Using
Personalized..." 에 초해상도(FSRCNN/ESPCN)·PERCLOS 중심이라 **현재 연구와 내용이 완전히
다르다.** 재사용하는 것은 **형식뿐**이다:
  - \documentclass{IEEE_lsens} 및 프리앰블
  - \IEEEtitleabstractindextext / \maketitle 구조
  - 표·그림 환경, \cite 스타일, 섹션 배치 관행
본문·초록·제목·표·그림·참고문헌은 **전부 새로 쓴다.** v1 문장을 살려 쓰지 마라.

[먼저 읽을 것 — 이 순서로]
1. docs/STATUS_2026-08-08.md   ★ 확정 숫자 단일 출처. 여기부터
2. docs/PAPER_OUTLINE.md       섹션별 구성·Table I·Table II·**서술 규칙**
3. docs/v2/REFERENCES.md       인용 후보(실제 인용 상한 15~20편)
4. docs/v2/RELATED_WORK.md §A·§A2·§A3
5. docs/PROJECT_DIRECTION.md §5  허용/금지 서술 범위

[논문의 주장 — 이것만 쓴다]
"눈 깜빡임 검출을 원본 이미지가 아니라 encoder 의 16차원 embedding 위에서 수행하고,
 동등한 정확도를 훨씬 적은 비용으로 얻으며, Raspberry Pi 5 에서 실시간으로 돈다."

정확도 우위가 아니라 **비용**이 기여축이다.

[🔴 절대 금지 — 하나라도 어기면 심사에서 무너진다]
1. "임베딩이 이미지보다 정확하다" 금지. ours(0.9886)는 Image-CNN(+head)(0.9906)보다 **낮다**.
   쓸 것: 동등한 정확도를 연산 2.6배·파라미터 5.7배 적게 (δ=0.02 기준 non_inferior)
2. "ours − Image-CNN(max) = +0.0766" 단독 인용 금지. 그 격차는 표현이 아니라
   **시간 모델링**이다(같은 백본+우리 헤드 = +0.0806). 그리고 원문은 프레임별
   open/closed 라벨로 학습했는데 우리는 창당 라벨 하나(max-pooling MIL)라 문헌 방법을
   과소평가했을 수 있다. **두 각주 모두 필수**
3. "비대칭 stride 가 핵심" 금지. 절제 실험에서 기각됐다(vpres−vdrop +0.0010 [−0.0032,+0.0069])
4. "causal TCN" 금지. 대칭 패딩이라 창 안에서 미래를 본다(실측).
   쓸 것: "19프레임 링버퍼 위의 양방향 TCN, 미래 프레임을 기다리지 않으므로 추가 지연 없음"
5. "privacy-preserving" / "identity-removed" / "irreversible" 금지.
   쓸 수 있는 최대치: "the classification stage operates on embeddings rather than raw eye images"
6. Nousias 2025 대비 "파라미터 420배" 금지. 그들의 최고 모델은 174,500 params 로 우리의 2.1배다
7. mEBAL2 원논문의 99% 와 우리 PR-AUC 를 나란히 놓지 마라. 입력 스펙트럼·인원·지표가 다르다
8. v1 수치(results/pi_*.json, 8명 공개셋, 초해상도 실험) 인용 금지
9. "최초"·"유일" 금지. T6 조사는 "검색 범위에서 못 찾았다"이지 부재 증명이 아니다
10. 탐색(fold 0·1) 숫자 금지. 확정은 5fold×3seed 뿐이다

[정직하게 실을 것 — 숨기면 심사자가 먼저 찾는다]
- Image-CNN(+head) 0.9906 이 Table I **최고값**이라는 사실
- Image-CNN(max) 시드 std 0.0291 (ours 의 7.7배) — Discussion 한 줄, 주장으로 밀지 말 것
- e2e 로는 11% 차이뿐이고 **얼굴 검출이 68~87%** 를 먹는다는 것
  → "에지에서 병목은 인코더가 아니라 검출 프론트엔드다"
- U1 전이 실패, 안경군 이득이 미착용군보다 낮다는 것
- image_cnn ONNX 지연은 **무작위 가중치**로 쟀다(구조 동일 → 지연 유효, 정확도와 분리)

[분량 — 약 4페이지. 여유가 없다]
I. Intro 0.6p / II. Related Work 0.4p / III. Method 1.0p (Fig.1 필수) /
IV. Setup 0.6p / V. Results 1.2p (Table I·II, Fig.2) / VI. Conclusion 0.2p
Fig.3(실패 사례)은 지면이 남을 때만.

[그림·표 자산]
- Fig. 1: docs/v2/figures/fig1_architecture.svg (숫자 전부 실측)
- Table I·II: docs/PAPER_OUTLINE.md §V 에 확정값이 채워져 있다. **그대로 옮긴다**

[작업 방식]
- 기존 tex 를 덮어쓰지 말고 **새 파일로 쓴 뒤** 사람이 비교해 교체하게 해라
- 모든 수치에 출처 파일 경로를 주석(%)으로 병기해라
- 확보되지 않은 것은 쓰지 마라. 미확보: 720p 보조 측정, T4-2/T4-3 층화,
  T6 [검색요약] 원문 확인(현재 0건 — Related Work 2문단은 이게 선행 조건이다)

[출력 규칙]
- 칭찬하지 마라. 무엇이 틀릴 수 있는지를 말해라
- 추측 금지. 불확실하면 "미확인"이라고 명시해라
- 마지막 줄에: 가장 확신 없는 지점 1개

먼저 docs/STATUS_2026-08-08.md 와 docs/PAPER_OUTLINE.md 를 읽고,
tex 골격(프리앰블·섹션 배치)만 남긴 새 파일 계획을 먼저 보고해라.
```
