"""Phase 1 (A) — 58명 전량, **영상 없이** CSV 만으로.

    python -m src.v2.phase1_csv58 --users 1-58

왜 영상이 필요 없는가
--------------------
EAR 베이스라인은 mEBAL2 가 제공하는 랜드마크만으로 계산됩니다. 검출 통계·라벨
커버리지·창 중앙 정렬·크롭 기하(span/tilt)도 마찬가지입니다. **영상이 필요한 것은
우리 크롭의 광학 지표(밝기·선명도)뿐입니다.** 그래서 `Processed_Data.zip`(30 GB)만
있으면 58명 베이스라인이 완성되고, 110 GB 짜리 웹캠 아카이브는 뒤로 미룰 수 있습니다.

속도
----
사용자당 landmarks.csv 가 약 1 GB 입니다. 전부 파싱하면 느리므로 두 갈래로 나눕니다.

    빠른 스캔 : 모든 줄에서 '[' 개수만 세어 얼굴 수를 얻는다 (C 속도, 실수 변환 없음)
    전체 파싱 : **라벨된 이벤트 프레임만** (전체의 5~25%)

zip 에서 바로 스트리밍하므로 디스크에 푸는 단계가 없습니다.

재개 가능
--------
사용자 한 명이 끝날 때마다 JSON 에 append 합니다. 이미 있는 사용자는 건너뜁니다.
실행 시간 상한이 있는 환경에서 여러 번 나눠 돌릴 수 있습니다.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zipfile

import numpy as np

from src.v2.common import repro
from src.v2.common import thresholds as TH
from src.v2.dataset import crop as C
from src.v2.dataset import mebal2 as M

ZIP = "data/mEBAL2/Processed_Data.zip"
PROBE = "data/raw/mEBAL2/_probe"
OUT = "results/v2/phase1_csv58.json"
N_MESH = M.N_MESH


def _atomic_dump(obj: dict, path: str) -> None:
    """임시 파일에 쓰고 rename.

    프로세스가 쓰기 도중 죽으면(실행 시간 상한·OOM) 결과 파일이 0바이트로 남아
    **누적분을 통째로 잃습니다.** 실제로 한 번 당했습니다. rename 은 원자적이라
    죽더라도 직전 상태가 온전히 남습니다.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse_users(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


_scan = M.scan_stream          # 파서는 src/v2/dataset/mebal2.py 하나만 쓴다


def process_user(zf: zipfile.ZipFile, user: int) -> dict:
    t0 = time.perf_counter()
    sess = M.Session.from_probe(PROBE, user)
    ev, isb = sess.events()
    need = {f for s, e, _ in ev for f in range(int(s), int(e) + 1)}

    with zf.open(f"Processed_Data/User {user}/box.csv") as fh:
        box_counts, box_parsed = _scan(fh, need, 4, 1)
    with zf.open(f"Processed_Data/User {user}/landmarks.csv") as fh:
        lm_counts, lm_parsed = _scan(fh, need, 3, N_MESH)
    t_scan = time.perf_counter() - t0

    n_frames = len(lm_counts)
    n_no_mesh = sum(1 for v in lm_counts.values() if v == 0)
    n_multi = sum(1 for v in lm_counts.values() if v > 1)
    n_dis = sum(1 for f, v in lm_counts.items() if v != box_counts.get(f, 0))

    # --- 라벨 프레임에서만 기하/EAR ---
    ears, spans, tilts, ok_frames = {}, [], [], set()
    for f in need:
        faces = lm_parsed.get(f)
        if faces is None or faces.shape[0] == 0:
            continue
        sel = M.select_face(f, faces, box_parsed.get(f))
        if sel.status != M.OK:
            continue
        xy = sel.mesh[:, :2]
        ok_frames.add(f)
        ears[f] = C.ear_both(xy)["mean"]
        le_a, le_b, re_a, re_b = C.eye_corners(xy)
        le, re = (le_a + le_b) / 2, (re_a + re_b) / 2
        d = re - le
        spans.append(float(np.linalg.norm(d)))
        ang = float(np.degrees(np.arctan2(d[1], d[0])))
        tilts.append(ang - 180 if ang > 90 else (ang + 180 if ang < -90 else ang))

    # 이벤트마다 EAR 특징 2개를 뽑는다.
    #   drop   = (창 양끝 평균 - 창 최저) / 양끝 평균   -> 깜빡임이 만든 상대 하강
    #   minear = 창 최저 EAR                            -> 절대 임계값 방식(고전 EAR 규칙)
    # blink 와 unblink **양쪽 모두** 계산해야 EAR 베이스라인의 실제 분리 성능이 나온다.
    # blink 쪽만 재면 "얼마나 떨어지나"는 알아도 "구분이 되나"는 모른다.
    offs_b, offs_u, n_dead = [], [], 0
    feat_drop, feat_min, lab, ev_meta, n_miss, curves = [], [], [], [], [], []
    for (s, e, _), b in zip(ev, isb):
        v = np.array([ears.get(f, np.nan) for f in range(int(s), int(e) + 1)], float)
        if np.isnan(v).all():
            n_dead += 1
            continue
        (offs_b if b == 1 else offs_u).append(int(np.nanargmin(v)))
        edge = np.nanmean([v[0], v[-1]])
        if np.isfinite(edge) and edge > 0:
            feat_drop.append(float((edge - np.nanmin(v)) / edge))
            feat_min.append(float(np.nanmin(v)))
            lab.append(int(b))
            ev_meta.append((int(s), int(e)))
            n_miss.append(int(np.isnan(v).sum()))
            # 19프레임 EAR 곡선 원본. **대조군 1(EAR 스칼라 x 19프레임 -> 동일 판정 헤드)**
            # 을 만들려면 이게 필요하다. 여기서 안 남기면 1 GB CSV 를 또 스캔해야 한다.
            curves.append([None if not np.isfinite(x) else round(float(x), 5) for x in v])

    fd = np.array(feat_drop); fm = np.array(feat_min); yl = np.array(lab, int)
    ev_used = np.array(ev_meta, np.int64).reshape(-1, 2)   # (start, end)
    dr = fd[yl == 1] if fd.size else np.array([])       # blink 하강률 (기존 지표 유지)
    dr_un = fd[yl == 0] if fd.size else np.array([])    # unblink 하강률 (빠져 있던 조각)
    # EAR 베이스라인의 사용자별 성능. 임계값 선택과 무관한 값이라 지금 단계에서 적절하다.
    auc_drop = TH.roc_auc(TH.canonical(fd, higher_fires=True), yl) if fd.size else None
    auc_min = TH.roc_auc(TH.canonical(fm, higher_fires=False), yl) if fm.size else None
    ob, ou = np.array(offs_b), np.array(offs_u)
    sp, tl = np.array(spans), np.array(tilts)
    # 크롭 폭 = span * MARGIN. 160px 로 줄이면 다운스케일, 늘리면 업스케일(보간 커널이 갈린다)
    cubic_rate = float(np.mean(sp * C.MARGIN < C.OUT_W)) if sp.size else None

    def q(a):
        return None if a.size == 0 else {
            "mean": float(a.mean()), "std": float(a.std()),
            "p10": float(np.percentile(a, 10)), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90))}

    return {
        "user": user, "subject": sess.subject,
        "seconds": round(time.perf_counter() - t0, 1), "scan_seconds": round(t_scan, 1),
        "frames": n_frames,
        "no_face_rate": n_no_mesh / n_frames if n_frames else None,
        "multi_face_rate": n_multi / n_frames if n_frames else None,
        "count_disagree_rate": n_dis / n_frames if n_frames else None,
        "n_blink": int((isb == 1).sum()), "n_unblink": int((isb == 0).sum()),
        "labelled_frames": len(need), "labelled_usable": len(ok_frames),
        "coverage": len(need) / n_frames if n_frames else None,
        "events_all_missing": n_dead,
        "ear_argmin": {
            "blink_median": float(np.median(ob)) if ob.size else None,
            "blink_within3": float(np.mean(np.abs(ob - 9) <= 3)) if ob.size else None,
            "unblink_within3": float(np.mean(np.abs(ou - 9) <= 3)) if ou.size else None,
        },
        "ear_drop": q(dr),
        "ear_drop_unblink": q(dr_un),
        "ear_baseline_auc": {
            "note": "EAR 만으로 blink vs unblink 를 얼마나 가르는가. 0.5 = 무작위. "
                    "영상 없이 계산되므로 이 값이 곧 사용자별 EAR 베이스라인 성능이다.",
            "by_drop_ratio": auc_drop,
            "by_min_ear": auc_min,
            "n_blink_used": int((yl == 1).sum()), "n_unblink_used": int((yl == 0).sum()),
        },
        "span_px": q(sp), "tilt_deg": q(tl),
        "interp_cubic_rate": cubic_rate,
        # 이벤트 단위 원자료. 이걸 남겨두면 fold 별 EAR 베이스라인·delta 산정에
        # 1 GB CSV 를 다시 스캔할 필요가 없다.
        "_events": {
            "start": ev_used[:, 0].tolist(), "end": ev_used[:, 1].tolist(),
            "is_blink": yl.tolist(), "drop": fd.tolist(), "min_ear": fm.tolist(),
            "n_missing": n_miss, "session_frames": n_frames,
            "ear19": curves,      # (n_events, 19) EAR 곡선. null = 얼굴 해소 실패
        },
    }


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="1-58")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    done: dict = {}
    if os.path.exists(args.out) and not args.redo and os.path.getsize(args.out) > 0:
        try:
            with open(args.out, encoding="utf-8") as f:
                done = json.load(f).get("users", {})
        except json.JSONDecodeError:
            print(f"  ! {args.out} 이 손상됐습니다(쓰기 중 중단). 처음부터 다시 돌립니다.")

    todo = [u for u in parse_users(args.users) if str(u) not in done]
    print(f"대상 {len(parse_users(args.users))}명 / 남은 {len(todo)}명")
    if not todo:
        print("모두 완료됨")
        return 0

    zf = zipfile.ZipFile(ZIP)
    for u in todo:
        try:
            r = process_user(zf, u)
        except KeyError:
            print(f"  User {u}: zip 에 항목 없음 — 건너뜀")
            continue
        done[str(u)] = r
        a = r["ear_baseline_auc"]
        db = r["ear_drop"]["median"] if r["ear_drop"] else float("nan")
        du = r["ear_drop_unblink"]["median"] if r["ear_drop_unblink"] else float("nan")
        print(f"  User {u:2d} {r['seconds']:5.1f}s  검출실패 {r['no_face_rate']:6.2%}  "
              f"커버리지 {r['coverage']:6.2%}  ev {r['n_blink']:4d}+{r['n_unblink']:4d}  "
              f"창중앙 {r['ear_argmin']['blink_within3']:.2f}  "
              f"하강 blink {db:.3f} / unblink {du:.3f}  "
              f"AUC drop {a['by_drop_ratio']:.3f}  minEAR {a['by_min_ear']:.3f}")
        _atomic_dump({"env": repro.env_fingerprint(), "users": done}, args.out)
    print(f"\n  -> {args.out}  (누적 {len(done)}명)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
