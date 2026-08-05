"""좌석 이탈 플래그 적용 + fold 재설계. **재추출 없이** index.npz 만 갱신한다.

    python -m src.v2.phase3_apply_flags

무엇을 하는가
------------
1. 58명 랜드마크를 다시 훑어 **프레임마다 선택된 얼굴의 화면 x**를 구한다
2. 주 좌석에서 `--tol` px 이상 벗어난 프레임을 `f_off_seat` 로 표시한다
3. 그 프레임을 하나라도 포함한 이벤트를 `e_off_seat` 로 표시한다
4. 제외 사용자(기본 U18)를 `e_excluded_user` 로 표시한다
5. `e_valid` 를 **결측 정책 AND 좌석 정상 AND 제외 사용자 아님** 으로 갱신한다
6. fold 를 **이벤트 수 + 수집 배치 이중 층화**로 다시 얼린다

왜 이렇게 하는가
---------------
mEBAL2 배포본은 **mEBAL1 38명(2020-02) + mEBAL2 신규 20명(2022)** 이다
(StudentData.csv 의 Exam_day_is 로 확인). 두 배치는 수집 시기가 2년 차이나고
실측 지표가 다르다 — 다중 얼굴 0.1% vs 11.2%, EAR AUC 0.933 vs 0.885,
밝기 57 vs 70. **fold 가 배치를 섞지 않으면 fold 마다 난이도가 달라진다.**
기존 fold 는 이벤트 수만 층화했고 배치 균형은 우연이었다.

2022 배치는 여러 학생이 한 방에서 동시에 녹화됐고, 주 피험자가 검출 실패한
프레임에서 **옆자리 사람이 선택**된다. U18 은 라벨 프레임의 28.3%, 이벤트의 31.2%
가 그런 경우라 통째로 제외한다 — 좌석 탐지가 완전하지 않으므로 3분의 1이 오염된
사용자에서는 잔여 위험이 크다. 나머지는 5% 이하라 이벤트 단위 제거로 충분하다.

산출물
    data/processed/v2/index.npz          갱신 (백업: index_pre_flags.npz)
    src/v2/common/folds_5fold.json       갱신 (백업: folds_5fold_v1.json)
    results/v2/apply_flags_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import zipfile

import numpy as np

from src.v2.common import repro, splits
from src.v2.dataset import crop as C
from src.v2.dataset import mebal2 as M
from src.v2.phase2_face_position import PD_ZIP, PROBE, exam_days

DATA = "data/processed/v2"
FOLDS = splits.FOLDS_PATH
REPORT = "results/v2/apply_flags_report.json"
EXCLUDE_DEFAULT = [18]


def face_x_per_frame(zf: zipfile.ZipFile, user: int) -> dict[int, float]:
    """라벨 프레임마다 선택된 얼굴의 눈 중심 x."""
    sess = M.Session.from_probe(PROBE, user)
    ev, _ = sess.events()
    need = {f for s, e, _ in ev for f in range(int(s), int(e) + 1)}
    with zf.open(f"Processed_Data/User {user}/box.csv") as fh:
        _, bp = M.scan_stream(fh, need, 4, 1)
    with zf.open(f"Processed_Data/User {user}/landmarks.csv") as fh:
        _, lp = M.scan_stream(fh, need, 3, M.N_MESH)
    out = {}
    for f in sorted(need):
        sel = M.select_face(f, lp.get(f), bp.get(f))
        if sel.status != M.OK:
            continue
        xy = sel.mesh[:, :2]
        la, lb, ra, rb = C.eye_corners(xy)
        out[f] = float((((la + lb) / 2 + (ra + rb) / 2) / 2)[0])
    return out


def primary_seat(x: np.ndarray, tol: float, bins: int = 64, w: int = 1280) -> float:
    """주 좌석 = **질량이 가장 큰 구간**의 중앙값. 최빈 구간을 쓰면 옆자리를 오인한다."""
    h, e = np.histogram(x, bins=bins, range=(0, w))
    ctr = (e[:-1] + e[1:]) / 2
    mass = np.array([h[np.abs(ctr - c) <= tol].sum() for c in ctr])
    c0 = ctr[int(mass.argmax())]
    sel = np.abs(x - c0) <= tol + (e[1] - e[0])
    return float(np.median(x[sel]))


# fold 생성기는 `src/v2/common/splits.py` 로 옮겼다 (2026-08-03).
# 규칙 #4: 격자·시드·분할·프로브는 공용 모듈 하나만 쓴다. 여기 두었더니 게이트가
# 얼린 배정을 재현해 검사할 수 없었다. 로직은 그대로다.
stratified_folds_2way = splits.stratified_folds_2way


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--tol", type=float, default=250.0)
    ap.add_argument("--exclude", nargs="*", type=int, default=EXCLUDE_DEFAULT)
    ap.add_argument("--force", action="store_true",
                    help="이미 적용된 index 에 다시 적용(권장하지 않음)")
    ap.add_argument("--report", default=REPORT)
    args = ap.parse_args()

    idx = dict(np.load(os.path.join(args.data, "index.npz")))

    # **재실행 방지 — 스캔(10분) 전에 막는다.**
    # 이미 적용된 index 를 다시 넣으면 prev_valid 가 '좌석 필터가 이미 걸린 e_valid'가
    # 되어 `e_valid_missing_only`(결측 정책만 적용한 기준)를 덮어쓴다. 그러면 민감도
    # 분석 S7("좌석 필터가 결과를 얼마나 바꿨나")이 영영 불가능해진다.
    if "e_valid_missing_only" in idx and not args.force:
        raise SystemExit(
            "이미 플래그가 적용된 index.npz 입니다(e_valid_missing_only 존재).\n"
            f"  다시 돌리려면 원본을 먼저 복원하십시오:\n"
            f"    copy {os.path.join(args.data, 'index_pre_flags.npz')} "
            f"{os.path.join(args.data, 'index.npz')}\n"
            "  (--force 로 무시할 수 있으나 S7 기준선을 잃습니다)")
    f_sub = idx["f_subject"].astype(int)
    f_frm = idx["f_frame_idx"].astype(int)
    e_sub = idx["e_subject"].astype(int)
    e_rows = idx["e_rows"].astype(np.int64)
    days = exam_days()
    batch = {u: days.get(u, "")[:4] for u in np.unique(f_sub)}

    # --- 1~3. 좌석 이탈 ---
    t0 = time.perf_counter()
    f_off = np.zeros(len(f_sub), np.uint8)
    seat = {}
    zf = zipfile.ZipFile(PD_ZIP)
    for u in sorted(np.unique(f_sub)):
        fx = face_x_per_frame(zf, int(u))
        k = np.flatnonzero(f_sub == u)
        x = np.array([fx.get(int(f), np.nan) for f in f_frm[k]])
        good = np.isfinite(x)
        if good.sum() == 0:
            continue
        m = primary_seat(x[good], args.tol)
        seat[int(u)] = m
        off = np.zeros(len(k), bool)
        off[good] = np.abs(x[good] - m) > args.tol
        f_off[k] = off.astype(np.uint8)
        print(f"  U{int(u):02d} {batch[int(u)]}  좌석 {m:6.0f}  이탈 {off.mean():6.2%}", flush=True)
    print(f"  스캔 {time.perf_counter() - t0:.0f}s")

    ok = e_rows >= 0
    e_off = np.zeros(len(e_sub), np.uint8)
    for j in range(e_rows.shape[1]):
        c = ok[:, j]
        e_off[c] |= f_off[e_rows[c, j]]

    e_excl = np.isin(e_sub, args.exclude).astype(np.uint8)

    prev_valid = idx["e_valid"].astype(bool)
    new_valid = prev_valid & (e_off == 0) & (e_excl == 0)

    idx["f_off_seat"] = f_off
    idx["e_off_seat"] = e_off
    idx["e_excluded_user"] = e_excl
    idx["e_valid_missing_only"] = prev_valid.astype(np.uint8)   # 이전 기준 보존
    idx["e_valid"] = new_valid.astype(np.uint8)
    idx["f_batch2020"] = np.array([batch[int(u)] == "2020" for u in f_sub], np.uint8)
    idx["e_batch2020"] = np.array([batch[int(u)] == "2020" for u in e_sub], np.uint8)
    idx["seat_tol"] = np.float32(args.tol)
    idx["excluded_users"] = np.asarray(args.exclude, np.int32)

    # --- 6. fold 재설계 ---
    keep_users = [int(u) for u in np.unique(e_sub) if int(u) not in args.exclude]
    cnt = {u: int((e_sub[new_valid] == u).sum()) for u in keep_users}
    assign = stratified_folds_2way(cnt, {u: batch[u] for u in keep_users})
    idx["e_fold"] = np.array([assign.get(int(u), -1) for u in e_sub], np.int8)
    idx["f_fold"] = np.array([assign.get(int(u), -1) for u in f_sub], np.int8)

    bak = os.path.join(args.data, "index_pre_flags.npz")
    if not os.path.exists(bak):
        shutil.copy(os.path.join(args.data, "index.npz"), bak)
    np.savez(os.path.join(args.data, "index.npz"), **idx)

    if os.path.exists(FOLDS) and not os.path.exists(FOLDS.replace(".json", "_v1.json")):
        shutil.copy(FOLDS, FOLDS.replace(".json", "_v1.json"))
    load = [sum(cnt[u] for u in keep_users if assign[u] == f) for f in range(5)]
    n20 = [sum(1 for u in keep_users if assign[u] == f and batch[u] == "2020") for f in range(5)]
    n22 = [sum(1 for u in keep_users if assign[u] == f and batch[u] == "2022") for f in range(5)]
    with open(FOLDS, "w", encoding="utf-8") as f:
        json.dump({"n_folds": 5,
                   "criterion": "유효 이벤트 수 + 수집 배치(2020/2022) 이중 층화, 결정적",
                   "excluded_users": args.exclude,
                   "balance": {"events_per_fold": load, "n_2020": n20, "n_2022": n22,
                               "max_over_min": max(load) / max(min(load), 1)},
                   "batch": {str(u): batch[u] for u in keep_users},
                   "counts": {str(u): cnt[u] for u in keep_users},
                   "assign": {str(u): assign[u] for u in keep_users}},
                  f, ensure_ascii=False, indent=1)

    rep = {
        "env": repro.env_fingerprint(), "tol": args.tol, "excluded_users": args.exclude,
        "frames_off_seat": int(f_off.sum()), "frames_total": int(len(f_off)),
        "events_off_seat": int(e_off.sum()),
        "events_excluded_user": int(e_excl.sum()),
        "events_valid_before": int(prev_valid.sum()),
        "events_valid_after": int(new_valid.sum()),
        "events_total": int(len(e_sub)),
        "fold_balance": {"events_per_fold": load, "n_2020": n20, "n_2022": n22},
        "primary_seat_x": seat,
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)

    print(f"\n좌석 이탈 프레임 {rep['frames_off_seat']:,} / {rep['frames_total']:,} "
          f"({rep['frames_off_seat']/rep['frames_total']:.2%})")
    print(f"유효 이벤트 {rep['events_valid_before']:,} -> {rep['events_valid_after']:,} "
          f"(-{rep['events_valid_before']-rep['events_valid_after']:,})")
    print(f"fold 이벤트 {load}  max/min {max(load)/max(min(load),1):.3f}")
    print(f"fold 배치   2020 {n20} / 2022 {n22}")
    print(f"  -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
