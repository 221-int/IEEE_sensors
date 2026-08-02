"""v2 공용 모듈.

여기 있는 네 개가 v2의 안전장치 전부입니다. 실험 스크립트는 임계값 격자·시드·
분할·프로브를 **직접 구현하지 않고** 반드시 여기서 가져다 씁니다.

    thresholds  임계값 격자 단일 구현 (ours/baseline 동일 격자 강제)
    repro       시드·결정성 봉인
    splits      fold 고정 · 시간블록/무작위 분할 · 균형 표집 · chance 보고
    probes      선형/MLP 프로브 단일 구현
"""

__all__ = ["thresholds", "repro", "splits", "probes"]
