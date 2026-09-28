"""Table I 추가 작업 전용 — 19프레임 링버퍼 위의 LSTM / BiLSTM 판정 헤드.

이 파일은 **새 파일**이다. 기존 `encoder.py` 의 `build_head`(TCN) 은 건드리지 않는다
(전력측정 작업과 동일 원칙: 확정 산출물이 걸린 파일은 복제해서 작업한다).

왜 이 헤드가 필요한가
--------------------
Table I 의 image_cnn 계열 비교는 지금 두 가지뿐이다.
  * `image_cnn_max`  — mEBAL 원문 §5.1 의 max-pooling 판정 (학습 파라미터 없음)
  * `image_cnn_head` — 같은 백본 + **우리 TCN 헤드**
문헌에서 blink detection 의 표준 시간 모델은 RNN(LSTM/BiLSTM) 이므로, "우리 TCN 이
좋아서 이긴 게 아니라 시간 모델을 붙였으니 이긴 것 아니냐"는 반론이 남는다.
**같은 백본에 LSTM/BiLSTM 을 붙인 행**이 있어야 그 반론이 닫힌다.

causality 표기 (작업지시서 §encoder.py 48-54, `build_head` 주석과 같은 규칙)
-------------------------------------------------------------------------
* `lstm`   — 단방향. 창 안에서도 과거만 본다. **가장 보수적인 대안.**
* `bilstm` — 창(19프레임) **안에서만** 양방향. 미래 프레임을 기다리지 않으므로
  시스템 추가 지연은 0 이고, 이는 기존 TCN 헤드와 **동일한 성질**이다
  (TCN 도 `padding=dilation` 대칭 Conv1d 라 창 안에서 미래를 본다).
  따라서 bilstm 은 **본행(main row)** 으로 보고한다.
  🔴 창 **밖**까지 보게 만들면(예: 이벤트 경계를 넘겨 전체 시퀀스에 BiLSTM) 그것은
  oracle 이며 **별도 행으로 분리 표기**해야 한다. 이 파일은 창 밖을 절대 보지 않는다.
  ("causal LSTM" 이라는 표현은 bilstm 에 대해서는 틀린 말이니 쓰지 말 것.)

결측 처리
--------
결측 프레임은 창 **내부 아무 위치에나** 올 수 있다(뒤쪽만이 아니다). 그래서
`pack_padded_sequence` 를 쓰지 않는다 — 그것은 뒤쪽 패딩만 다룬다.
대신 TCN 헤드와 **같은 규칙**을 쓴다: 입력을 0 으로 채워 넣되 pooling 에서 제외한다.
"""

from __future__ import annotations


def rnn_head_mmac(d: int = 16, hidden: int = 16, t: int = 19,
                  layers: int = 1, bidirectional: bool = False,
                  mlp_hidden: int = 64) -> float:
    """헤드 **호출당** MMAC. `encoder.temporal_head_mmac` 과 같은 자리에서 쓴다.

    LSTM 한 방향 한 층의 시점당 MAC = 4 * (D*H + H*H) (게이트 4개).
    Table I 의 edge 비용 열을 채울 때 이 값을 쓴다.
    """
    dirs = 2 if bidirectional else 1
    macs = 0
    din = d
    for _ in range(layers):
        macs += t * dirs * 4 * (din * hidden + hidden * hidden)
        din = hidden * dirs
    pooled = 2 * hidden * dirs                     # (mean, max) concat
    macs += pooled * mlp_hidden + mlp_hidden
    return macs / 1e6


def build_rnn_head(d_latent: int = 16, t: int = 19, hidden: int = 16,
                   layers: int = 1, bidirectional: bool = False,
                   mlp_hidden: int = 64, dropout: float = 0.3):
    """벡터 T개 + 마스크 -> 깜빡임 로짓. LSTM/BiLSTM 판정 헤드.

    시그니처와 반환 규약을 `encoder.build_head` 와 맞춘다:
    `forward(z, mask)` 에서 z (N, T, D), mask (N, T) -> (N,) 로짓.
    그래야 학습 루프를 한 줄도 고치지 않고 헤드만 바꿔 끼울 수 있다.
    """
    import torch
    from torch import nn

    class RNNHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(d_latent, hidden, num_layers=layers,
                               batch_first=True, bidirectional=bidirectional,
                               dropout=(dropout if layers > 1 else 0.0))
            out_dim = hidden * (2 if bidirectional else 1)
            self.mlp = nn.Sequential(nn.Linear(2 * out_dim, mlp_hidden), nn.ReLU(True),
                                     nn.Dropout(dropout), nn.Linear(mlp_hidden, 1))
            self.bidirectional = bidirectional

        def forward(self, z, mask):                 # z: (N, T, D), mask: (N, T)
            m = mask.unsqueeze(-1)                  # (N, T, 1)
            h, _ = self.rnn(z * m)                  # 결측은 0 입력 (TCN 헤드와 동일)
            h = h * m                               # 결측 시점 출력은 pooling 에서 제외
            n = m.sum(dim=1).clamp(min=1.0)
            avg = h.sum(dim=1) / n
            mx = h.masked_fill(m == 0, float("-inf")).max(dim=1).values
            mx = torch.nan_to_num(mx, neginf=0.0)
            return self.mlp(torch.cat([avg, mx], dim=1)).squeeze(1)

    return RNNHead()


if __name__ == "__main__":
    for name, bi in (("lstm", False), ("bilstm", True)):
        print(f"{name:>7}  H=16  {rnn_head_mmac(16, 16, bidirectional=bi):.4f} MMAC/call")