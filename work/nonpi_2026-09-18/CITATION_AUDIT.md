# 인용 확인 기록 — 2026-09-18

원문에서 이번 원고가 사용하는 주장과 관련된 부분을 확인했다. 논문 전체를 재현한 것은 아니다. 검색 결과의 초록만 확인한 경우와 저자 원문을 확인한 경우를 구분한다.

| 원고 키 | 확인 근거 | 현재 원고에 허용되는 사용 / 조치 |
|---|---|---|
| daza2020 | [저자 PDF](https://biometrics.eps.uam.es/fierrez/files/2020_ICMI_mEBAL_Daza.pdf), §4–5 | CNN과 13프레임 점수 max 집계 확인. 원고의 양눈 gray 64×160, 이벤트 BCE는 재현 원형이 아니라 명시된 변형이다. 원논문은 mEBAL 학습→HUST-LEBW 검증도 수행했으므로 외부 검증 자체를 새 기여로 주장하지 않는다. |
| daza2024 | [저자 공개 원문](https://arxiv.org/html/2309.07880v2), §3–5 | 전체 180명과 다중스펙트럼 데이터, CNN/시계열 비교 및 HUST-LEBW 평가 확인. 본 연구 57명은 전체 데이터셋이 아니다. |
| soukupova2016 | [정확한 CVWW 원문](https://cmp.felk.cvut.cz/ftp/articles/cech/Soukupova-CVWW-2016.pdf) | EAR 시간창 SVM을 사용하는 방법. EAR를 오직 고정 임계값 규칙이라고 소개하면 부정확하다. 별도의 55쪽 학위논문 파일은 이 학회 원문의 대체 서지로 사용하지 않는다. |
| fodor2023 | [저자 PDF](https://adamfodor.com/pdf/2023_Fodor_Adam_MDPI_BlinkLinMulT.pdf) | transformer 기반 다중 데이터셋 평가 선행연구. 특징 추출+시계열 모델링 자체를 최초 기여로 주장하지 않는다. 2026년 업데이트된 GitHub v2의 수치와 2023년 논문 수치를 혼합하지 않는다. |
| hu2019 | [저자 논문 PDF](https://cse.buffalo.edu/~jsyuan/papers/2020/Guilei_tifs20.pdf), [IEEE 서지](https://ieeexplore.ieee.org/document/8933048/) | HUST-LEBW와 LSTM 기반 방법. 기존 참고문헌에서 누락된 Junsong Yuan을 추가했다. 온라인 공개 2019년, 최종 권 15(2020) pp.2194–2208을 구분한다. |
| zeng2023 | [저자 프로젝트](https://wenzhengzeng.github.io/mpeblink/), [CVPR 원문](https://openaccess.thecvf.com/content/CVPR2023/papers/Zeng_Real-Time_Multi-Person_Eyeblink_Detection_in_the_Wild_for_Untrimmed_Video_CVPR_2023_paper.pdf) | 다중 인물·비절단 영상 평가. 프로젝트의 HUST-LEBW 비절단 subset 정정도 확인: HUST에 비절단 평가가 전혀 없다고 쓰면 안 된다. |
| arcaya2021 | [저자 업로드 원문](https://www.researchgate.net/publication/344463803_Deep_Learning_for_Eye_Blink_Detection_Implemented_at_the_Edge), 본문 IR sensing/STM32 구현 부분 | 연결형 안경의 IR 센서 시계열 CNN. RGB 프레임·FaceMesh·Pi와 직접 지연/전력 우열 비교하지 않는다. DOI 10.1109/LES.2020.3029313. |
| capacitive2024 | [출판사 원문](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1394397/full) | 정전용량 센서 기반 안경. 센싱 방식이 다르므로 camera 기반 직접 대조군이 아니다. |
| nousias2025 | [출판사](https://www.mdpi.com/2313-433X/11/1/27), [동일 논문 공개 PDF](https://pdfs.semanticscholar.org/193e/eb175b32bac9c6c2d7aa6edfce13605b1565.pdf), 방법·Table 3 | 3D autoencoder의 latent classifier와 예측 누적·구간 검출이 이미 존재한다. 학습은 시퀀스 분류, 추론은 누적 프레임/구간 분석임을 구별한다. 임베딩으로 깜빡임을 판정한 최초라는 주장은 배제한다. |
| rosenfield2011 | [출판사 서지/요약](https://onlinelibrary.wiley.com/doi/10.1111/j.1475-1313.2011.00834.x) | 제목·권·쪽 확인. 전체 유료 본문 검토 완료로 표시하지 않는다. 일반 배경 인용이며 본 시스템의 의학적 효용 증거가 아니다. |
| fogelton2016, eyeblink8data | [데이터 저자 페이지](https://www.blinkingmatters.com/research) | 영상 8개/4명, blink ID와 좌우눈 주석, 수정 버전별 계수·평가 단위 차이를 확인. 이번 결과는 로컬 9번 영상의 양눈 blink ID 41개에 대한 pilot이며 공식 per-eye F1과 비교하지 않는다. |

확인되지 않은 항목: 본 연구의 직접 Pi RGB 동일 입력 대조 실험을 대신할 타 논문의 공정한 수치 비교, 미입수 RNN raw prediction 파일, 외부 전체 영상의 확보·평가. 인용을 찾았다는 것과 직접 실험 비교가 가능하다는 것은 다르다.
