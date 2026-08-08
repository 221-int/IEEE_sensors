# 새 세션 프롬프트 — 아래 블록을 통째로 복사해 붙여넣는다

```
저장소 C:\Users\sch\PycharmProjects\IEEE_sensors 에서 IEEE Sensors Letters 논문 작업을 이어간다.

[연구 방향]
논문 범위는 **embedding vector 기반 eye blink detection + edge device 실시간 성능 검증**이다.
신원 제거·복원 방지·프라이버시는 범위 밖이다(특허·후속 연구로 분리됨).
문서에 "최종 목표는 인코더 단 신원 재식별 불가능"이 남아 있으면 낡은 것이다.

[먼저 읽을 것 — 이 순서로]
1. docs/STATUS_2026-08-08.md   ★ 확정된 숫자와 남은 일. 여기부터
2. docs/PAPER_OUTLINE.md       논문 구성·Table I·Table II·서술 규칙
3. docs/v2/PROTOCOL.md §1~§13  ※ §14·§15 는 범위 밖. 게이트로 적용하지 마라
4. docs/EXPERIMENT_PLAN.md §8  완료 조건 E1~E10 현황

[상태 요약 — 성능 축은 끝났다]
- Table I 5행 확정(5fold×3seed): ours 0.9886±0.0038 / Image-CNN(+head) 0.9906±0.0037 /
  Image-CNN(max) 0.9114±0.0291 / EAR-head 0.9724 / EAR-rule 0.8931
- ours − Image-CNN(+head) = −0.0040 [−0.0063,−0.0020], δ=0.02 기준 **non_inferior**
- 서브그룹: EAR-head 상대 다섯 그룹 전부 superior / Image-CNN(+head) 상대 전부 non_inferior
- 구조 절제: 세로 해상도 논거 **지지되지 않음**(vpres−vdrop +0.0010 [−0.0032,+0.0069])
- Pi 5: 네 모드 전부 G-E1 PASS. 단 **출처 JSON 이 저장소에 없다**

[🔴 절대 규칙]
1. 탐색(fold 0·1) 숫자를 확정 값으로 인용하지 마라. 확정은 5fold×3seed 뿐이다
2. **"임베딩이 이미지보다 정확하다"는 쓸 수 없다.** ours 는 Image-CNN(+head)보다 0.004 낮다.
   쓸 수 있는 것은 "동등한 정확도를 연산 2.6배·파라미터 5.7배 적게"(non_inferior)
3. **"ours − Image-CNN(max) = +0.0766" 단독 인용 금지.** 그 격차는 표현이 아니라
   시간 모델링이다(같은 백본 + 우리 헤드 = +0.0806). 그리고 원문은 프레임별
   open/closed 라벨로 학습했는데 우리는 창당 라벨 하나라 문헌 방법을 과소평가했을 수 있다.
   **두 각주 모두 필수**
4. **"비대칭 stride 가 핵심"을 Method 근거로 쓰지 마라.** 절제 실험에서 기각됐다
5. **"causal TCN" 이라고 쓰지 마라.** 대칭 패딩이라 창 안에서 미래를 본다(실측).
   정확한 서술은 "19프레임 링버퍼 위의 양방향 TCN, 시스템 추가 지연 없음"
6. **Nousias 2025 대비 "파라미터 420배" 쓰지 마라.** 그들의 최고 모델은 174,500 params 다
7. v1 수치(results/pi_*.json, 8명 공개셋)는 **인용 금지**
8. 격자·시드·분할·프로브는 src/v2/common/ 하나만 쓴다
9. 런이 도는 동안 그 런이 임포트한 모듈(train_encoder.py, model/encoder.py, common/*,
   dataset/crop.py)을 수정하지 마라. 지문이 오염된 전례가 있다
10. 실행 전에 커밋해라

[할 일 — 우선순위 순]
① Pi 클립·JSON 확인 (사용자가 전달 예정)
   - results/v2/pi_*_480p.json 을 저장소에 넣고 Table II-b 숫자를 파일로 검증
   - ⚠️ 480p 클립 매니페스트(10,268프레임/342s)와 보고된 런(31,268프레임/300s)이 안 맞는다.
     clips/list7.txt(7회 concat)로 만든 클립으로 보이나 파일이 clips/ 에 없다.
     어느 클립으로 쟀는지 확인하고 clips_manifest.json 을 갱신해라
② T4 강건성 — 광학 3분위 층화. 값은 이미 있다(results/v2/photometrics_58.json).
   posthoc_subgroups 에 광학 층화를 추가하면 된다. 이벤트 가중 풀링이 주 숫자
③ T6 관련 연구 — 에지/경량 깜빡임 검출. docs/v2/RELATED_WORK.md §A 확장.
   [원문]/[검색요약] 구분을 지켜라
④ 논문 집필 — PAPER_OUTLINE.md 순서대로. Table I·Fig.1 은 준비됨

[출력 규칙]
- 칭찬하지 마라. 동의 여부보다 무엇이 틀릴 수 있는지를 말해라
- 추측 금지. 확인된 것만 쓰고 불확실하면 "미확인"이라고 명시해라
- 모든 수치에 출처 파일 경로를 병기해라
- 마지막 줄에: 가장 확신 없는 지점 1개

먼저 docs/STATUS_2026-08-08.md 를 읽고 시작해라.
```
