# 🚫 results/split_eval.json — 폐기 (2026-07-30)

EAR 임계값 선택에 **성긴 고정 격자**를 써서 fold1 의 EAR 임계값이 격자 경계값(0.100)에 걸렸다.
저FA 구간을 짚지 못해 EAR recall 이 0.521 로 깎였고, 그 결과 "Ours 0.854 vs EAR 0.878 →
대등 / 판정 불가" 라는 잘못된 결론이 나왔다.

**유효한 결과는 `results/split_eval_calib.json` 의 `none` 조건이다.**
Ours 0.803 ± 0.321 vs EAR 0.977 ± 0.035 (차이 −0.174).

이 파일의 숫자는 어떤 문서·발표에도 인용하지 말 것. 상세: `docs/정정_2026-07-30.md`
