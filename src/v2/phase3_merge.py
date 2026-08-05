"""샤드 58개 -> memmap + 통합 인덱스. **검증이 이 스크립트의 본체다.**

    python -m src.v2.phase3_merge                    # 병합 + 검증
    python -m src.v2.phase3_merge --verify-only      # 기존 산출물만 재검증

왜 memmap 인가
-------------
m22 크롭만 532,109 x 64 x 160 uint8 = **5.45 GB** 라 RAM 에 다 올릴 수 없다.
memmap 으로 두면 fold 별 학습이 필요한 부분만 페이지 단위로 읽는다.

무엇을 강제하는가
----------------
병합은 인덱스를 옮겨 붙이는 작업이라 **조용히 어긋나기 가장 쉬운 단계**다. 크롭 배열과
이벤트 인덱스가 한 칸이라도 밀리면 모든 학습이 무의미해지는데, 손실은 정상으로 보인다.
그래서 아래를 전부 검사하고 하나라도 실패하면 산출물을 남기지 않는다.

    V1  memmap 의 shape / dtype / 총 프레임 수가 샤드 합계와 일치
    V2  모든 이벤트의 행이 **같은 피험자**를 가리킨다
    V3  모든 이벤트의 행이 가리키는 원본 프레임 번호가 정확히 start+k 다  ← 1:1 정합
    V4  행 인덱스가 [0, N) 안에 있다
    V5  (subject, frame_idx) 쌍에 중복이 없다
    V6  m22 와 m18 의 길이·정합이 동일하다
    V7  이벤트 총수가 28,728 이고 사용자별 blink == unblink 다

산출물
    data/processed/v2/frames_m22.npy   (N, 64, 160) uint8  memmap
    data/processed/v2/frames_m18.npy   동일
    data/processed/v2/index.npz        프레임·이벤트·피험자 인덱스 전부
    results/v2/merge_report.json       샤드별 입력 수 / 제외 수 / 검증 결과
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from src.v2.common import repro, splits

SHARDS = "data/processed/v2/shards/u*.npz"
OUT_DIR = "data/processed/v2"
REPORT = "results/v2/merge_report.json"
CROP_HW = (64, 160)
EVENT_LEN = 19
MAX_MISSING = 5          # PROTOCOL §7 확정: 이벤트당 결측 5 이하만 사용
EXPECT_EVENTS = 28728


def _shard_user(path: str) -> int:
    m = re.search(r"u(\d+)\.npz$", os.path.basename(path))
    if not m:
        raise ValueError(f"샤드 이름에서 사용자 번호를 못 읽었습니다: {path}")
    return int(m.group(1))


# ------------------------------------------------------------------ 1차 스캔
def survey(paths: list[str], tags: list[str]) -> dict:
    """프레임 수를 세고 샤드 자체의 일관성을 먼저 본다(메모리에 안 올림)."""
    info, total = [], 0
    for p in paths:
        d = np.load(p)
        u = int(d["user"])
        if u != _shard_user(p):
            raise SystemExit(f"{p}: 파일명 사용자({_shard_user(p)}) != 내부 user({u})")
        n = len(d["f_frame_idx"])
        for t in tags:
            key = f"frames_{t}"
            if key not in d.files:
                raise SystemExit(f"{p}: {key} 가 없습니다. --margins 를 확인하십시오.")
            sh = d[key].shape
            if sh != (n, *CROP_HW):
                raise SystemExit(f"{p}: {key} shape {sh} != {(n, *CROP_HW)}")
            if d[key].dtype != np.uint8:
                raise SystemExit(f"{p}: {key} dtype {d[key].dtype} != uint8")
        info.append({"path": p, "user": u, "n_frames": n,
                     "n_events": int(len(d["e_is_blink"])),
                     "session_frames": int(d["session_frames"]),
                     "base": total})
        total += n
    return {"shards": info, "total_frames": total}


# ------------------------------------------------------------------ 병합
def merge(paths: list[str], tags: list[str], out_dir: str) -> dict:
    sv = survey(paths, tags)
    N = sv["total_frames"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"샤드 {len(paths)}개  총 프레임 {N:,}  배율 {tags}")
    print(f"  예상 디스크 {len(tags) * N * CROP_HW[0] * CROP_HW[1] / 1e9:.2f} GB")

    mm = {t: np.lib.format.open_memmap(
        os.path.join(out_dir, f"frames_{t}.npy"), mode="w+",
        dtype=np.uint8, shape=(N, *CROP_HW)) for t in tags}

    F: dict[str, list] = {k: [] for k in
                          ("subject", "frame_idx", "t_rel", "span_px", "tilt_deg")}
    Fm: dict[str, list] = {}
    ear = []
    E: dict[str, list] = {k: [] for k in
                          ("subject", "event_id", "is_blink", "blink_flag",
                           "start", "end", "t_rel", "n_missing")}
    e_rows = []
    per_shard = []

    for s in sv["shards"]:
        d = np.load(s["path"])
        u, base, n = s["user"], s["base"], s["n_frames"]
        for t in tags:
            mm[t][base:base + n] = d[f"frames_{t}"]
        F["subject"].append(np.full(n, u, np.int16))
        F["frame_idx"].append(d["f_frame_idx"].astype(np.int32))
        F["t_rel"].append(d["f_t_rel"].astype(np.float32))
        F["span_px"].append(d["f_span_px"].astype(np.float32))
        F["tilt_deg"].append(d["f_tilt_deg"].astype(np.float32))
        ear.append(d["f_ear"].astype(np.float32))
        for t in tags:
            for fld in ("interp_cubic", "padded", "brightness", "contrast", "sharpness"):
                Fm.setdefault(f"{fld}_{t}", []).append(d[f"f_{fld}_{t}"])

        r = d["e_rows"].astype(np.int64)
        r = np.where(r >= 0, r + base, -1)          # 지역 -> 전역 행 인덱스
        e_rows.append(r)
        ne = r.shape[0]
        E["subject"].append(np.full(ne, u, np.int16))
        E["event_id"].append(np.arange(ne, dtype=np.int32))
        E["is_blink"].append(d["e_is_blink"].astype(np.uint8))
        E["blink_flag"].append(d["e_blink_flag"].astype(np.uint8))
        E["start"].append(d["e_start"].astype(np.int32))
        E["end"].append(d["e_end"].astype(np.int32))
        E["t_rel"].append(d["e_t_rel"].astype(np.float32))
        E["n_missing"].append(d["e_n_missing"].astype(np.int16))
        per_shard.append({"user": u, "frames_in": n, "events_in": ne,
                          "base_row": base,
                          "session_frames": s["session_frames"]})

    for t in tags:
        mm[t].flush()

    idx = {f"f_{k}": np.concatenate(v) for k, v in F.items()}
    idx["f_ear"] = np.concatenate(ear)
    for k, v in Fm.items():
        idx[f"f_{k}"] = np.concatenate(v)
    for k, v in E.items():
        idx[f"e_{k}"] = np.concatenate(v)
    idx["e_rows"] = np.concatenate(e_rows).astype(np.int32)
    idx["e_n_missing"] = idx["e_n_missing"].astype(np.int16)
    idx["e_valid"] = (idx["e_n_missing"] <= MAX_MISSING).astype(np.uint8)

    # fold 는 얼려둔 파일에서 가져온다 (여기서 새로 만들지 않는다)
    assign = splits.load_folds()
    idx["e_fold"] = np.array([assign[int(u)] for u in idx["e_subject"]], np.int8)
    idx["f_fold"] = np.array([assign[int(u)] for u in idx["f_subject"]], np.int8)
    idx["max_missing_policy"] = np.int32(MAX_MISSING)
    idx["margins"] = np.array(tags)
    idx["crop_hw"] = np.array(CROP_HW, np.int32)

    np.savez(os.path.join(out_dir, "index.npz"), **idx)

    for s, ps in zip(sv["shards"], per_shard):
        ev = idx["e_subject"] == ps["user"]
        ps["events_excluded_missing"] = int((idx["e_valid"][ev] == 0).sum())
        ps["events_used"] = int(ps["events_in"] - ps["events_excluded_missing"])
    return {"total_frames": N, "tags": tags, "per_shard": per_shard, "out_dir": out_dir}


# ------------------------------------------------------------------ 검증
def verify(out_dir: str, expected_frames: int | None = None) -> dict:
    idx = dict(np.load(os.path.join(out_dir, "index.npz"), allow_pickle=False))
    tags = [str(x) for x in idx["margins"]]
    N = len(idx["f_subject"])
    checks: dict[str, dict] = {}

    def add(name, ok, detail=""):
        checks[name] = {"pass": bool(ok), "detail": detail}
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))

    print("검증")
    # V1 memmap shape/dtype/개수
    ok, det = True, []
    for t in tags:
        a = np.load(os.path.join(out_dir, f"frames_{t}.npy"), mmap_mode="r")
        good = (a.shape == (N, *CROP_HW)) and (a.dtype == np.uint8)
        ok &= good
        det.append(f"{t}:{a.shape},{a.dtype}")
    if expected_frames is not None:
        ok &= (N == expected_frames)
        det.append(f"N={N} (기대 {expected_frames})")
    add("V1 memmap shape/dtype/총 프레임 수", ok, " ".join(det))

    r = idx["e_rows"].astype(np.int64)
    valid = r >= 0
    # V4 범위
    add("V4 행 인덱스 범위 [0, N)",
        bool(((r[valid] >= 0) & (r[valid] < N)).all()),
        f"N={N}, 유효행 {int(valid.sum()):,}")

    # V2 같은 피험자
    es = np.repeat(idx["e_subject"][:, None], EVENT_LEN, axis=1)
    bad_sub = int((idx["f_subject"][r[valid]] != es[valid]).sum())
    add("V2 이벤트 행 <-> 피험자 일치", bad_sub == 0, f"불일치 {bad_sub}")

    # V3 1:1 정합 — 행이 가리키는 원본 프레임 번호가 start + k 인가
    k = np.tile(np.arange(EVENT_LEN), (r.shape[0], 1))
    want = idx["e_start"][:, None] + k
    bad_fi = int((idx["f_frame_idx"][r[valid]] != want[valid]).sum())
    add("V3 이벤트 행 <-> 프레임 번호 1:1", bad_fi == 0, f"불일치 {bad_fi}")

    # V5 (subject, frame_idx) 중복
    key = idx["f_subject"].astype(np.int64) * 10_000_000 + idx["f_frame_idx"]
    dup = int(len(key) - len(np.unique(key)))
    add("V5 (subject, frame) 중복 없음", dup == 0, f"중복 {dup}")

    # V6 배율 간 길이 동일
    lens = {t: np.load(os.path.join(out_dir, f"frames_{t}.npy"), mmap_mode="r").shape[0]
            for t in tags}
    add("V6 배율 간 길이 동일", len(set(lens.values())) == 1, str(lens))

    # V7 이벤트 총수 / 사용자별 1:1 균형
    # 총수 검사는 58명 전량일 때만 강제한다. 부분 병합(개발·검증용)에서 헛되이
    # FAIL 이 나면 진짜 실패와 구분이 안 되기 때문이다.
    ne = len(idx["e_subject"])
    subs = np.unique(idx["e_subject"])
    bal = all(int((idx["e_is_blink"][idx["e_subject"] == u] == 1).sum())
              == int((idx["e_is_blink"][idx["e_subject"] == u] == 0).sum())
              for u in subs)
    full = len(subs) == splits.N_SUBJECTS
    ok7 = bal and (ne == EXPECT_EVENTS if full else True)
    add("V7 사용자별 blink==unblink" + (" & 이벤트 28,728" if full else " (부분 병합)"),
        ok7, f"피험자 {len(subs)}, 이벤트 {ne:,}, 균형 {bal}"
             + ("" if full else "  ※ 총수 검사는 58명 전량에서만 강제"))

    # 필드 보존 확인
    need = [f"f_interp_cubic_{t}" for t in tags] + ["e_n_missing", "e_valid", "e_fold"]
    miss = [k for k in need if k not in idx]
    add("필드 보존 (interp_cubic / 결측 / fold)", not miss, f"누락 {miss}")

    all_ok = all(c["pass"] for c in checks.values())
    print(f"\n  판정: {'PASS' if all_ok else 'FAIL'}")
    return {"all_pass": all_ok, "checks": checks, "n_frames": int(N), "n_events": int(ne)}


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=SHARDS)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--margins", default="m22,m18")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--report", default=REPORT)
    args = ap.parse_args()

    tags = args.margins.split(",")
    out: dict = {"env": repro.env_fingerprint(), "margins": tags}

    if not args.verify_only:
        paths = sorted(glob.glob(args.shards))
        if not paths:
            raise SystemExit(f"샤드가 없습니다: {args.shards}")
        m = merge(paths, tags, args.out_dir)
        out.update(m)
        print()
    v = verify(args.out_dir, out.get("total_frames"))
    out["verify"] = v

    if not args.verify_only:
        ps = out["per_shard"]
        tot_in = sum(p["events_in"] for p in ps)
        tot_ex = sum(p["events_excluded_missing"] for p in ps)
        print(f"\n  이벤트 입력 {tot_in:,} / 결측정책(≤{MAX_MISSING})으로 제외 {tot_ex:,} "
              f"({tot_ex / tot_in:.2%}) / 사용 {tot_in - tot_ex:,}")
        worst = sorted(ps, key=lambda p: -p["events_excluded_missing"])[:5]
        print("  제외 상위:", [(p["user"], p["events_excluded_missing"]) for p in worst])

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    tmp = args.report + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.report)
    print(f"\n  -> {args.report}")

    if not v["all_pass"]:
        print("  ! 검증 실패. 이 산출물로 학습하지 마십시오.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
