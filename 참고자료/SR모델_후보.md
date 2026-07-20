# 라즈베리파이5용 경량 SR 모델 후보 정리

정리일: 2026-07-17
용도: 눈 영역 품질이 낮을 때(멀거나·저해상도·저조도) 얼굴/눈 ROI를 SR로 개선 → 랜드마크·EAR 안정화.

---

## 결론 (추천)

**1순위: OpenCV `cv2.dnn_superres` + FSRCNN-small (또는 ESPCN), 2×~3×, "눈/얼굴 ROI에만" 적용.**
- 우리 파이프라인이 이미 OpenCV 기반 → **추가 프레임워크 0, 드롭인 통합**.
- ESPCN/FSRCNN은 경량이라 Pi에서 실시간(전체 프레임도 Pi4에서 10~15fps 보고). 우리는 **작은 눈 패치**에만 돌리니 훨씬 빠름.
- ★ 핵심 트릭: 1280×720 전체가 아니라 **눈 bbox(예: 40×24px)만 잘라 2×~4×** → 연산량 미미 → 조건부로만 켜면 실시간 여유.

---

## 후보 비교

| 모델 | 프레임워크 | Pi 실현성 | 통합 난이도 | 비고 |
|---|---|---|---|---|
| **FSRCNN / FSRCNN-small** | OpenCV dnn_superres | ★ 좋음(경량) | 매우 쉬움 | small은 더 빠름/조금 덜 정확. **1순위** |
| **ESPCN** | OpenCV dnn_superres | ★ 좋음(sub-pixel) | 매우 쉬움 | 실시간 영상 업스케일 설계. 대안 1순위 |
| LapSRN | OpenCV dnn_superres | 보통(ESPCN보다 느림) | 쉬움 | 2×/4×/8×, 품질↑ 속도↓ |
| EDSR | OpenCV dnn_superres | 나쁨(무거움) | 쉬움 | 품질 최고지만 Pi 실시간 부적합 → 제외 |
| **edge-SR (eSR)** | PyTorch/직접 | ★ 좋음(초경량) | 보통 | 단일 conv급 초경량, Pi CPU 2×/3× 부분성공. FSRCNN도 느리면 대안 |
| Real-ESRGAN (small/general) | ncnn-vulkan | 보통~나쁨 | 어려움 | 품질 최고, GAN. 눈 ROI만이면 시도 가능하나 무거움 → 후순위 |

---

## OpenCV dnn_superres 사용법 (robust.py 에 바로 연결)

```python
import cv2
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("FSRCNN-small_x2.pb")   # 또는 ESPCN_x2.pb
sr.setModel("fsrcnn", 2)             # ("espcn", 2) 등
eye_hr = sr.upsample(eye_crop)       # 눈 ROI만 입력
```
→ 지금 `eyeblink/robust.py` 의 `SuperResolution` 스텁 자리에 그대로 넣으면 됩니다.

**사전학습 모델(.pb) 받는 곳 (opencv_contrib 공식 링크):**
- FSRCNN: https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models
- ESPCN:  https://github.com/fannymonori/TF-ESPCN/tree/master/export
- LapSRN: https://github.com/fannymonori/TF-LapSRN/tree/master/export
- (참고) dnn_superres 안내: https://github.com/opencv/opencv_contrib/blob/master/modules/dnn_superres/README.md

**초경량 대안:**
- edge-SR (eSR): https://github.com/pnavarre/eSR
- Real-ESRGAN ncnn: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan

---

## 실험/논문에서 반드시 확인할 것
1. **SR이 실제로 랜드마크·EAR을 개선하는가** — 저해상도/암전 셋에서 SR 유무 검출 정확도 비교(순차 SR이 오히려 해치는 사례 있음 → 검증 필수).
2. **배율/모델별 Pi5 fps·전력** 실측 → "조건부 SR로도 실시간 유지" 근거.
3. **품질 게이트 임계값** — 눈 가로 픽셀 수 몇 px 미만일 때 SR을 켤지(예: 저해상도 <30×30에서 랜드마크 급락 근거 활용).
4. SR 모델은 자연영상 학습이라 얼굴 특화가 아님 → 1차는 FSRCNN/ESPCN로, 필요 시 얼굴 특화 SR 검토.
