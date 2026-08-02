"""Phase 1 파일럿 — 사용자 1명으로 처리량과 크롭 품질을 먼저 잰다.

    python -m src.v2.phase1_pilot --user 1 --n-frames 20

58명을 바로 돌리기 전에 **한 명으로 견적을 낸다**. 원본 zip 이 441 GB 라 처리량을
모르는 채로 대량 실행에 들어가면 되돌리기 어렵습니다.

두 부분으로 나뉩니다.

  A. 영상 없이 (CSV 만)  — 검출 통계 · 라벨 커버리지 · EAR 곡선 · 창 중앙 정렬 검증
     ★ EAR 은 제공 랜드마크만으로 계산되므로 **베이스라인은 디코딩이 전혀 필요 없습니다.**
       영상이 필요한 것은 우리 크롭뿐입니다. 58명 EAR 베이스라인은 CSV 만으로 완성됩니다.

  B. 영상 필요            — 크롭 기하 · 광학 지표 · 순차 디코딩 처리량

산출물
    results/v2/phase1_pilot_user{NN}.json
    docs/v2/figures/pilot_user{NN}_crops.png
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from src.v2.common import repro
from src.v2.dataset import crop as C
from src.v2.dataset import mebal2 as M

PROBE = "data/raw/mEBAL2/_probe"
RAW = "data/raw/mEBAL2"


def video_path(user: int) -> str:
    return os.path.join(RAW, f"User {user}", "RealSense", "Color_Webcam", "color.mp4")


# --------------------------------------------------------------- A. 영상 없이
def part_a(sess: M.Session, n_video_frames: int | None) -> dict:
    t0 = time.perf_counter()
    det = sess.detection_stats()
    ev, isb = sess.events()
    lens = np.unique(ev[:, 1] - ev[:, 0]) if len(ev) else np.array([])

    labelled = {f for s, e, _ in ev for f in range(int(s), int(e) + 1)}
    total = n_video_frames if n_video_frames else det["frames"]

    # --- EAR 곡선 + 창 중앙 정렬 검증 (영상 불필요) ---
    # 이벤트가 깜빡임 중심에 정렬돼 있다면, blink 창 19프레임의 EAR 최저점이
    # 중앙(오프셋 9) 근처여야 합니다. 이 가정이 깨지면 시퀀스 설계 전체가 흔들립니다.
    offs_blink, offs_unblink, drops = [], [], []
    n_incomplete = 0
    for (s, e, _), b in zip(ev, isb):
        vals = []
        for f in range(int(s), int(e) + 1):
            sel = sess.face(f)
            vals.append(C.ear_both(sel.mesh[:, :2])["mean"]
                        if sel.status == M.OK else np.nan)
        v = np.array(vals, float)
        if np.isnan(v).all():
            n_incomplete += 1
            continue
        off = int(np.nanargmin(v))
        (offs_blink if b == 1 else offs_unblink).append(off)
        if b == 1:
            edge = np.nanmean([v[0], v[-1]])
            if np.isfinite(edge) and edge > 0:
                drops.append(float((edge - np.nanmin(v)) / edge))

    ob = np.array(offs_blink)
    ou = np.array(offs_unblink)
    return {
        "seconds": round(time.perf_counter() - t0, 1),
        "detection": det,
        "events": {
            "n_blink": int((isb == 1).sum()), "n_unblink": int((isb == 0).sum()),
            "distinct_window_len": [int(x) for x in lens],
            "events_with_no_face_at_all": n_incomplete,
        },
        "coverage": {
            "video_frames": total,
            "unique_labelled_frames": len(labelled),
            "coverage": len(labelled) / total if total else None,
        },
        "ear_center_alignment": {
            "note": "blink 창 19프레임 중 EAR 최저 지점의 위치. 중앙 = 9",
            "blink_argmin_median": float(np.median(ob)) if ob.size else None,
            "blink_argmin_mean": float(ob.mean()) if ob.size else None,
            "blink_within_1_of_centre": float(np.mean(np.abs(ob - 9) <= 1)) if ob.size else None,
            "blink_within_3_of_centre": float(np.mean(np.abs(ob - 9) <= 3)) if ob.size else None,
            "unblink_argmin_median": float(np.median(ou)) if ou.size else None,
            "unblink_within_3_of_centre": float(np.mean(np.abs(ou - 9) <= 3)) if ou.size else None,
        },
        "ear_drop_ratio_blink": {
            "note": "(창 양끝 평균 EAR - 창 최저 EAR) / 양끝 평균. 완전 폐안일수록 1에 가깝다",
            "median": float(np.median(drops)) if drops else None,
            "p10": float(np.percentile(drops, 10)) if drops else None,
            "p90": float(np.percentile(drops, 90)) if drops else None,
            "n": len(drops),
        },
    }


# --------------------------------------------------------------- B. 영상 필요
def part_b(sess: M.Session, video: str, n_frames: int, out_png: str,
           max_frame: int | None = None) -> dict:
    """max_frame: 이 프레임 번호까지만 표본을 뽑는다.

    시킹이 금지돼 있으므로 마지막 표본까지 **순차로** 넘어가야 하고, 그 시간이 곧
    이 함수의 실행 시간입니다. 실행 시간 상한이 있는 환경(개발 샌드박스)에서는
    세션 전체를 한 번에 통과할 수 없으므로 구간을 잘라 돌립니다. 그 경우
    `session_fraction_covered` 로 어디까지 봤는지 반드시 함께 보고합니다.
    """
    cap = cv2.VideoCapture(video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()

    ev, _ = sess.events()
    labelled = sorted({f for s, e, _ in ev for f in range(int(s), int(e) + 1)})
    usable = [f for f in labelled if sess.face(f).status == M.OK]
    if max_frame is not None:
        usable = [f for f in usable if f <= max_frame]
    if not usable:
        return {"error": "얼굴이 해소된 라벨 프레임이 없습니다"}
    pick = [usable[i] for i in np.linspace(0, len(usable) - 1, n_frames).astype(int)]
    pick = sorted(set(pick))

    t0 = time.perf_counter()
    crops, metas, photos, fails = [], [], [], 0
    for idx, frame in M.iter_frames(video, pick):
        sel = sess.face(idx)
        g, meta = C.crop_both_eyes(frame, sel.mesh[:, :2])
        if g is None:
            fails += 1
            continue
        crops.append(g); metas.append(meta.as_dict()); photos.append(C.photometrics(g))
    elapsed = time.perf_counter() - t0

    if crops:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        cols = 4
        rows = int(np.ceil(len(crops) / cols))
        sheet = np.zeros((rows * C.OUT_H, cols * C.OUT_W), np.uint8)
        for i, g in enumerate(crops):
            r, c = divmod(i, cols)
            sheet[r * C.OUT_H:(r + 1) * C.OUT_H, c * C.OUT_W:(c + 1) * C.OUT_W] = g
        cv2.imwrite(out_png, sheet)

    def agg(key, src):
        v = np.array([d[key] for d in src], float)
        return {"mean": float(v.mean()), "std": float(v.std()),
                "min": float(v.min()), "max": float(v.max())}

    last = max(pick)
    return {
        "video": {"path": video, "w": w, "h": h, "frames": n_total},
        "max_frame_limit": max_frame,
        "session_fraction_covered": round((last + 1) / n_total, 3) if n_total else None,
        "sampled_frames": len(pick), "crops_ok": len(crops), "crop_failed": fails,
        "throughput": {
            "seconds": round(elapsed, 2),
            "frames_traversed": last + 1,
            "grab_fps": round((last + 1) / elapsed, 1),
            "note": "순차 grab 으로 마지막 표본까지 도달하는 데 걸린 시간 기준",
            "est_full_pass_s": round(n_total / max((last + 1) / elapsed, 1e-9), 1),
        },
        "geometry": {k: agg(k, metas) for k in ("span_px", "tilt_deg", "crop_w_px")},
        "interp_cubic_rate": float(np.mean([m["interp_cubic"] for m in metas])),
        "padded_rate": float(np.mean([m["padded"] for m in metas])),
        "photometrics": {k: agg(k, photos) for k in ("brightness", "contrast", "sharpness")},
        "crop_spec": {"out_h": C.OUT_H, "out_w": C.OUT_W, "margin": C.MARGIN,
                      "status": "미확정 — Phase 1/2 에서 확정"},
        "contact_sheet": out_png,
    }


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=1)
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--max-frame", type=int, default=None,
                    help="이 프레임까지만 표본을 뽑는다. 실행 시간 상한이 있는 환경용. "
                         "워크스테이션에서는 지정하지 말 것(세션 전체를 봐야 드리프트가 잡힌다)")
    ap.add_argument("--skip-a", action="store_true", help="A 부분 건너뛰기(이미 잰 경우)")
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()

    vp = video_path(args.user)
    has_video = os.path.exists(vp)
    sess = M.Session.from_probe(PROBE, args.user, video=vp if has_video else None)

    print(f"Phase 1 파일럿 — User {args.user}")
    print(f"  landmarks {sess.landmarks_csv}")
    print(f"  video     {vp}  ({'있음' if has_video else '없음'})")

    n_vid = None
    if has_video:
        cap = cv2.VideoCapture(vp); n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()

    out: dict = {"env": repro.env_fingerprint(), "user": args.user}
    path = f"results/v2/phase1_pilot_user{args.user:02d}.json"
    if args.skip_a and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            out = json.load(f)
        print("\n[A] 이전 결과 재사용")
        a = out["part_a"]
    else:
        print("\n[A] 영상 없이 (CSV 만)")
        a = part_a(sess, n_vid)
        out["part_a"] = a
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:      # A 를 먼저 저장(B 가 죽어도 남게)
            json.dump(out, f, ensure_ascii=False, indent=1)
    d, e, c = a["detection"], a["events"], a["coverage"]
    print(f"  프레임 {d['frames']}  얼굴해소 {d['ok']}  검출실패 {d['no_face_rate']:.2%}  "
          f"다중얼굴 {d['multi_face_rate']:.2%}")
    print(f"  얼굴수 불일치 {d['count_disagree_rate']:.2%}  박스없는 메시 {d['mesh_without_box']}")
    print(f"  이벤트 blink {e['n_blink']} / unblink {e['n_unblink']}  창길이 {e['distinct_window_len']}")
    print(f"  라벨 커버리지 {c['coverage']:.4f}  (고유 {c['unique_labelled_frames']} / {c['video_frames']})")
    al = a["ear_center_alignment"]
    print(f"  EAR 최저 위치(중앙=9): blink 중앙값 {al['blink_argmin_median']}  "
          f"±3 안 {al['blink_within_3_of_centre']:.1%}  |  unblink 중앙값 {al['unblink_argmin_median']}")
    dr = a["ear_drop_ratio_blink"]
    print(f"  blink EAR 하강률 중앙 {dr['median']:.3f} (p10 {dr['p10']:.3f} / p90 {dr['p90']:.3f})")

    if has_video and not args.skip_video:
        print("\n[B] 영상 디코딩")
        png = f"docs/v2/figures/pilot_user{args.user:02d}_crops.png"
        b = part_b(sess, vp, args.n_frames, png, args.max_frame)
        out["part_b"] = b
        t, g = b["throughput"], b["geometry"]
        print(f"  표본 {b['sampled_frames']}  크롭성공 {b['crops_ok']}  실패 {b['crop_failed']}"
              f"  세션 커버 {b['session_fraction_covered']:.1%}")
        print(f"  순차 디코딩 {t['grab_fps']} fps  -> 전체 1회 통과 추정 {t['est_full_pass_s']}s "
              f"(58명 {t['est_full_pass_s']*58/3600:.1f}h)")
        print(f"  span {g['span_px']['mean']:.1f}±{g['span_px']['std']:.1f} px  "
              f"crop_w {g['crop_w_px']['mean']:.1f} px  "
              f"tilt {g['tilt_deg']['mean']:+.1f}°")
        print(f"  업스케일(INTER_CUBIC) 비율 {b['interp_cubic_rate']:.1%}  "
              f"패딩 {b['padded_rate']:.1%}")
        p = b["photometrics"]
        print(f"  밝기 {p['brightness']['mean']:.1f}±{p['brightness']['std']:.1f}  "
              f"선명도 {p['sharpness']['mean']:.1f}±{p['sharpness']['std']:.1f}")
        print(f"  -> {png}")

    path = f"results/v2/phase1_pilot_user{args.user:02d}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
