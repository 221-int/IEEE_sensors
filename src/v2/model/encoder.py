"""v2 인코더 — 세로 해상도를 지키는 비대칭 CNN + 인과적 시간 헤드.

설계 근거 (전부 실측)
--------------------
**1. 세로를 줄이면 신호가 사라진다.**
눈꺼풀 상하 간격은 크롭 세로 64 px 중 **8.7 px** 이다
(눈꺼풀간격/span = 0.1201 실측, MARGIN 2.2, out_w 160 → 0.1201 × 160 / 2.2).
stride-2 conv 를 세로로 4번 쓰면 8.7 / 16 = **0.55 px** 가 되어 깜빡임이 특징맵에서
사라진다. 3×3 커널이 변화를 보려면 최소 3 px 은 남아야 하므로
**세로 총 stride 를 2 로 제한**한다 (8.7 / 2 = 4.35 px).

**2. 가로는 줄여도 된다.**
깜빡임은 세로 운동이다. 가로 160 px 은 두 눈을 나란히 담기 위한 것이지 해상도가
필요해서가 아니다. 그래서 stem 에서 가로만 4배로 줄이고, 이후에도 가로 위주로 줄인다.
**비대칭 stride 가 이 설계의 핵심이다.**

**3. 시퀀스는 인과적 스트리밍으로 처리한다.**
mEBAL2 의 라벨 단위가 19프레임 이벤트이므로 판정은 시퀀스 단위여야 한다. 그러나
프레임마다 19장을 다시 인코딩하면 비용이 19배가 되어 Pi 예산을 넘는다.
→ **프레임당 인코딩 1회 + 벡터 19개 링버퍼 + 시간 헤드**. 시간 헤드는 0.2 MMAC 로
사실상 공짜다.

**4. 결측은 마스크로 알린다.**
얼굴 해소 실패 프레임은 크롭 자체가 없다. 0 으로 채우기만 하면 "변화 없음"과
"모름"이 구분되지 않으므로, 시간 헤드에 **마스크를 함께 넣고 pooling 에서 제외**한다.

이 파일의 위치 (2026-08-05 정정)
-------------------------------
여기 있는 것이 **논문에 실리는 인코더**다. IEEE Sensors Letters 논문의 범위는
**embedding vector 기반 eye blink detection + edge device 실시간 성능 검증**이며,
신원 억제(적대 학습 GRL·정보 병목)는 **논문 범위 밖**이다
(`docs/PROJECT_DIRECTION.md`, `docs/PATENT_AND_FUTURE_WORK.md` §4).

⚠️ 이전 판(2026-08-04)의 "최종 목표는 인코더 단에서 신원 재식별을 불가능하게
만드는 것 / 억제 기전은 이번 라운드에 포함" 주석은 **폐기했다.**

⚠️ **`d_latent` 는 아직 확정이 아니다.** 구조(vpres, 세로 총 stride 2)만 확정이며,
그 근거는 위의 세로 해상도 논거다(프라이버시와 무관하므로 그대로 유효).
D 는 **깜빡임 검출 성능 + edge 비용(MMAC·ONNX 크기·지연)** 으로 확정한다
(`docs/TASKS.md` T3-5). 파레토 기준은 더 이상 적용하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- 실측 상수
APERTURE_PER_SPAN = 0.1201   # 눈꺼풀 상하 간격 / 두 눈 중심 거리 (User 1·46, n=4,992)
CROP_H, CROP_W = 64, 160
MARGIN = 2.2


def aperture_px(out_w: int = CROP_W, margin: float = MARGIN) -> float:
    """크롭에서 눈꺼풀 간격이 몇 픽셀인가. span 과 무관하다."""
    return APERTURE_PER_SPAN * out_w / margin


@dataclass(frozen=True)
class Conv:
    c_in: int
    c_out: int
    k: tuple[int, int]
    s: tuple[int, int]

    def out_hw(self, h: int, w: int) -> tuple[int, int]:
        return (h + self.s[0] - 1) // self.s[0], (w + self.s[1] - 1) // self.s[1]

    def macs(self, h: int, w: int) -> int:
        oh, ow = self.out_hw(h, w)
        return oh * ow * self.c_out * self.c_in * self.k[0] * self.k[1]


# ---------------------------------------------------------------- 후보 구조
# (이름, conv 목록, 설명)
SPECS: dict[str, list[Conv]] = {
    # 대조군: 흔히 쓰는 대칭 stride-2 x4. 세로가 1/16 로 줄어 신호가 사라진다.
    "sym16": [
        Conv(1, 16, (3, 3), (2, 2)), Conv(16, 32, (3, 3), (2, 2)),
        Conv(32, 64, (3, 3), (2, 2)), Conv(64, 128, (3, 3), (2, 2)),
    ],
    # 채택 후보: 세로 총 stride 2, 가로 32. stem 에서 가로만 4배 축소.
    "vpres": [
        Conv(1, 16, (3, 5), (1, 4)), Conv(16, 32, (3, 3), (2, 2)),
        Conv(32, 48, (3, 3), (1, 2)), Conv(48, 64, (3, 3), (1, 2)),
    ],
    # 더 보수적: 세로 stride 1 (전혀 안 줄임). 비용 확인용.
    "vfull": [
        Conv(1, 16, (3, 5), (1, 4)), Conv(16, 32, (3, 3), (1, 2)),
        Conv(32, 48, (3, 3), (1, 2)), Conv(48, 64, (3, 3), (1, 2)),
    ],
}


def analyse(name: str, d_latent: int = 16, h: int = CROP_H, w: int = CROP_W) -> dict:
    """구조 하나의 MMAC·특징맵·세로 신호 잔량을 계산한다 (torch 불필요)."""
    convs = SPECS[name]
    cur_h, cur_w, macs = h, w, 0
    vstride = hstride = 1
    for c in convs:
        macs += c.macs(cur_h, cur_w)
        cur_h, cur_w = c.out_hw(cur_h, cur_w)
        vstride *= c.s[0]; hstride *= c.s[1]
    c_last = convs[-1].c_out
    # 가로 평균 pooling -> (C, H') 를 flatten -> Linear(D)
    flat = c_last * cur_h
    macs_fc = flat * d_latent
    ap = aperture_px()
    return {
        "name": name, "feat": (c_last, cur_h, cur_w),
        "vstride": vstride, "hstride": hstride,
        "aperture_in_px": ap, "aperture_at_bottleneck_px": ap / vstride,
        "conv_mmac": macs / 1e6, "fc_mmac": macs_fc / 1e6,
        "total_mmac": (macs + macs_fc) / 1e6,
        "flat_dim": flat, "d_latent": d_latent,
    }


def temporal_head_mmac(d: int = 16, t: int = 19, layers: int = 3, k: int = 3) -> float:
    """시간 헤드(1D TCN, dilation 1·2·4) + 판정 MLP 의 MMAC.

    **프레임당이 아니라 이벤트당** 비용이다. 인과적 스트리밍에서는 창이 찰 때마다
    한 번 돌면 되므로 프레임당으로 환산하면 이보다 더 작다.
    """
    tcn = t * d * d * k * layers
    mlp = d * 2 * 64 + 64          # (mean, max) concat -> 64 -> 1
    return (tcn + mlp) / 1e6


def pi_ms(mmac: float, gmac_per_s: float = 15.0) -> float:
    """Pi 5 추정 지연. **미측정 상수**이므로 Phase 6 실측 전까지 확정으로 쓰지 말 것.

    Cortex-A76 @2.4GHz, NEON fp32 8 MAC/cycle/core, ORT intra-op 2스레드 →
    이론 상한 38.4 GMAC/s. conv 효율 30~50% 를 가정해 15 GMAC/s 로 잡았다.
    """
    return mmac / gmac_per_s


def report(d_latent: int = 16) -> None:
    ap = aperture_px()
    print(f"눈꺼풀 간격: 크롭에서 {ap:.2f} px (세로 {CROP_H} 중)")
    print(f"{'구조':>8}{'특징맵 CxHxW':>16}{'세로stride':>10}{'병목 신호':>10}"
          f"{'conv MMAC':>11}{'총 MMAC':>10}{'Pi 추정':>10}")
    for name in SPECS:
        r = analyse(name, d_latent)
        c, hh, ww = r["feat"]
        print(f"{name:>8}{f'{c}x{hh}x{ww}':>16}{r['vstride']:>10}"
              f"{r['aperture_at_bottleneck_px']:>10.2f}"
              f"{r['conv_mmac']:>11.2f}{r['total_mmac']:>10.2f}"
              f"{pi_ms(r['total_mmac']):>9.2f}ms")
    th = temporal_head_mmac(d_latent)
    print(f"\n시간 헤드(D={d_latent}, T=19): {th:.3f} MMAC = {pi_ms(th):.4f} ms  (이벤트당)")
    print("※ Pi 추정은 15 GMAC/s 가정. Phase 6 실측 전까지 확정 아님.")


# ---------------------------------------------------------------- torch 모델
def build(name: str = "vpres", d_latent: int = 16, in_ch: int = 1):
    """torch 인코더를 만든다. torch 가 없으면 ImportError."""
    import torch
    from torch import nn

    convs = SPECS[name]
    layers: list = []
    cin = in_ch
    for c in convs:
        layers += [
            nn.Conv2d(cin, c.c_out, c.k, c.s,
                      padding=(c.k[0] // 2, c.k[1] // 2), bias=False),
            nn.BatchNorm2d(c.c_out), nn.ReLU(inplace=True),
        ]
        cin = c.c_out

    class Encoder(nn.Module):
        """크롭 1장 -> D차원 벡터. **배포되는 것은 이 부분뿐이다.**"""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*layers)
            r = analyse(name, d_latent)
            self.fc = nn.Linear(r["flat_dim"], d_latent)
            self.d_latent = d_latent

        def forward(self, x):                       # x: (N, C, H, W)
            h = self.net(x)
            h = h.mean(dim=3)                       # 가로 평균 pooling -> (N, C, H')
            return self.fc(h.flatten(1))

    return Encoder()


def build_ear_frontend(k_in: int = 4, d_latent: int = 16):
    """대조군 1: EAR 스칼라(들) -> D차원. **판정 헤드는 우리와 동일한 것을 쓴다.**

    이렇게 해야 비교가 "무엇을 프레임마다 뽑는가"(학습된 CNN 벡터 vs EAR 스칼라)로
    좁혀진다. 규칙 기반 임계값과 비교하면 시간 모델링 능력 차이까지 우리 공로로
    잘못 계산된다.
    """
    import torch
    from torch import nn

    class EarLift(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(k_in, d_latent), nn.BatchNorm1d(d_latent),
                                     nn.ReLU(True), nn.Linear(d_latent, d_latent))
            self.d_latent = d_latent

        def forward(self, x):                   # x: (N, k)
            return self.net(x)

    return EarLift()


def build_head(d_latent: int = 16, t: int = 19, hidden: int = 64):
    """벡터 19개 + 마스크 -> 깜빡임 확률. 인과적 TCN(dilation 1·2·4).

    마스크가 0 인 시점은 pooling 에서 제외한다 — 결측을 0 으로 채우면
    "변화 없음"으로 오인된다.
    """
    import torch
    from torch import nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            d = d_latent
            self.tcn = nn.ModuleList([
                nn.Conv1d(d, d, 3, padding=p, dilation=p) for p in (1, 2, 4)
            ])
            self.norm = nn.ModuleList([nn.BatchNorm1d(d) for _ in range(3)])
            self.mlp = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(True),
                                     nn.Dropout(0.3), nn.Linear(hidden, 1))

        def forward(self, z, mask):                 # z: (N, T, D), mask: (N, T)
            h = z.transpose(1, 2)                   # (N, D, T)
            m = mask.unsqueeze(1)                   # (N, 1, T)
            for conv, bn in zip(self.tcn, self.norm):
                h = torch.relu(bn(conv(h * m))) + h
            h = h * m
            n = m.sum(dim=2).clamp(min=1.0)
            avg = h.sum(dim=2) / n
            mx = h.masked_fill(m == 0, float("-inf")).max(dim=2).values
            mx = torch.nan_to_num(mx, neginf=0.0)
            return self.mlp(torch.cat([avg, mx], dim=1)).squeeze(1)

    return Head()


if __name__ == "__main__":
    report()
