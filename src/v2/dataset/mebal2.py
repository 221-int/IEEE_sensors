"""mEBAL2 파싱 · 얼굴 해소 · 순차 프레임 읽기 — v2.

아래는 전부 **배포 아카이브를 직접 열어 확인한 사실**이며, 공식 문서와 어긋나는
곳에서는 데이터를 따릅니다.

    공식 문서                     실제
    ---------------------------  --------------------------------------
    68 facial landmarks          MediaPipe FaceMesh 468점 (얼굴당)
    box = xywh                   box = xyxy
    (미기재)                      좌표는 정규화가 아니라 **픽셀**
    (미기재)                      모든 CSV 가 쉼표가 아니라 **공백** 구분

그 밖에 이 모듈이 인코딩하고 있는 함정들:

  * landmarks.csv 는 얼굴 구분자 없이 [x,y,z] 를 **평면**으로 늘어놓습니다.
    468개씩 잘라야 합니다.
  * box 개수와 landmark 얼굴 개수가 **일치하지 않습니다**. assert 로 막으면 6% 가
    죽습니다. 인덱스로 짝짓지 말고 **메시 중심점이 어느 박스 안에 드는지**로
    기하 매칭합니다.
  * Time.csv 만 1-based, landmarks/box/blink CSV/영상은 0-based 입니다.
  * **영상 프레임 시킹 금지.** cv2 의 CAP_PROP_POS_FRAMES 는 h264 에서 가장 가까운
    키프레임으로 가고 조용히 엉뚱한 프레임을 돌려줍니다 → 크롭이 전부 어긋납니다.
    순차 grab()/retrieve() 만 씁니다.
  * 제공된 눈 크롭 이미지를 쓰지 않고 **원본에서 다시 자릅니다**(크기가 제각각이고,
    눈 사이 거리 기반 스케일 규칙을 복원할 수 없습니다).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import cv2
import numpy as np

N_MESH = 468        # 얼굴당 FaceMesh 점 수
EVENT_LEN = 19      # 모든 이벤트가 고정 19프레임 (End - Start = 18)
FPS = 30.0

# 프레임 번호 기준. 0 = 0-based, 1 = 1-based.
FRAME_BASE = {"video": 0, "landmarks": 0, "box": 0, "blink": 0, "unblink": 0, "time": 1}

OK, NO_MESH, NO_ROW = "ok", "no_mesh", "no_row"


def time_row_for_frame(frame_idx: int) -> int:
    """0-based 영상 프레임 -> Time.csv 의 0-based 행 인덱스.

    Time.csv 의 Frame 열이 1-based 이므로 행 i 가 Frame i+1 을 담습니다. 즉 0-based
    프레임 f 의 행은 그냥 f 입니다. 이 off-by-one 이 **한 곳에만** 살도록 함수로 둡니다.
    """
    return frame_idx


# ------------------------------------------------------------------ 파싱
def _parse_bracket_rows(path: str, item_len: int, what: str) -> dict[int, np.ndarray]:
    """'<frame> [[a,b,..],[a,b,..]]' 공백 구분 행 -> {frame: (n, item_len)}.

    빈 리스트 '[]' 는 (0, item_len) 배열이 됩니다(검출 실패).
    """
    out: dict[int, np.ndarray] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        head = f.readline()
        if "[" in head:                     # 헤더가 없는 파일이면 되감기
            f.seek(0)
        for lineno, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            try:
                frame_s, rest = line.split(" ", 1)
                frame = int(frame_s)
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: 행 시작이 이상합니다: {line[:60]!r}") from e
            rest = rest.strip()
            if rest in ("[]", ""):
                out[frame] = np.zeros((0, item_len), np.float32)
                continue
            flat = np.fromstring(rest.replace("[", " ").replace("]", " "),
                                 sep=",", dtype=np.float32)
            if flat.size % item_len:
                raise ValueError(f"{path}:{lineno}: {what} 개수 {flat.size} 가 "
                                 f"{item_len} 의 배수가 아닙니다")
            out[frame] = flat.reshape(-1, item_len)
    return out


def load_landmarks(path: str) -> dict[int, np.ndarray]:
    """landmarks.csv -> {frame: (n_faces, 468, 3)} float32, **픽셀** 좌표."""
    raw = _parse_bracket_rows(path, 3, "landmark")
    out = {}
    for frame, a in raw.items():
        if a.shape[0] % N_MESH:
            raise ValueError(f"{path}: frame {frame} 의 랜드마크 {a.shape[0]} 개가 "
                             f"{N_MESH} 의 배수가 아닙니다")
        out[frame] = a.reshape(-1, N_MESH, 3)
    return out


def load_boxes(path: str) -> dict[int, np.ndarray]:
    """box.csv -> {frame: (n_faces, 4)} **xyxy**. xywh 로 읽히면 즉시 실패시킵니다."""
    boxes = _parse_bracket_rows(path, 4, "box 좌표")
    for frame, b in boxes.items():
        if b.size and not np.all((b[:, 2] >= b[:, 0]) & (b[:, 3] >= b[:, 1])):
            raise ValueError(f"{path}: frame {frame} 박스가 xyxy 가 아닙니다: {b}")
    return boxes


def load_events(path: str, has_flag: bool) -> np.ndarray:
    """blink/unblink CSV -> (n, 3) [start, end, flag].

    flag 는 Blink 열(1 = blink, 0 = possible blink)이며 Unblink.csv 에는 그 열이
    없으므로 1 로 채웁니다. 창 길이가 19가 아닌 행도 그대로 돌려주고 판단은
    호출자에게 맡깁니다 — 없다고 가정하지 않습니다.
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) < 2 or not p[0].lstrip("-").isdigit():
                continue                     # 헤더 또는 빈 줄
            flag = int(p[2]) if (has_flag and len(p) > 2) else 1
            rows.append((int(p[0]), int(p[1]), flag))
    return np.array(rows, np.int64).reshape(-1, 3)


def scan_stream(fh, need: set[int], item_len: int, n_per_face: int):
    """zip 멤버 스트림 1회 통과 -> (counts, parsed).

    사용자당 landmarks.csv 가 약 1 GB 라 전부 실수 변환하면 느립니다. 그래서 두 갈래:

        모든 줄  : '[' 개수만 세어 얼굴 수를 얻는다 (C 속도, 실수 변환 없음)
        need 만  : 전체 파싱

    counts  {frame: n_faces}          — 모든 프레임
    parsed  {frame: (n_faces, n_per_face, item_len)} — need 에 든 프레임만
             (n_per_face == 1 이면 (n_items, item_len))

    ★ phase1(측정)과 phase2(크롭 추출)가 **같은 파서**를 쓰도록 여기 둡니다.
      파서가 두 벌이 되면 두 단계의 얼굴 해소 결과가 갈릴 수 있습니다.
    """
    counts: dict[int, int] = {}
    parsed: dict[int, np.ndarray] = {}
    first = True
    for raw in fh:
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
        if not line:
            continue
        if first:
            first = False
            if "[" not in line:                     # 헤더 줄
                continue
        sp = line.find(" ")
        if sp < 0:
            continue
        try:
            frame = int(line[:sp])
        except ValueError:
            continue
        rest = line[sp + 1:]
        n_items = rest.count("[") - 1
        counts[frame] = max(n_items // n_per_face if n_per_face > 1 else n_items, 0)
        if frame in need and n_items > 0:
            flat = np.fromstring(rest.replace("[", " ").replace("]", " "),
                                 sep=",", dtype=np.float32)
            if flat.size % item_len == 0:
                a = flat.reshape(-1, item_len)
                if n_per_face > 1:
                    if a.shape[0] % n_per_face == 0:
                        parsed[frame] = a.reshape(-1, n_per_face, item_len)
                else:
                    parsed[frame] = a
    return counts, parsed


# ------------------------------------------------------------------ 얼굴 해소
def box_areas(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), np.float32)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def match_mesh_to_box(faces: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """메시 -> 박스 인덱스 매핑 (없으면 -1).

    **인덱스 순서로 짝짓지 않습니다.** 두 파일의 얼굴 개수가 6% 프레임에서 다르므로
    순서를 신뢰할 수 없습니다. 메시 중심점이 들어가는 박스를 찾고, 없으면 가장 가까운
    박스에 붙입니다. 한 박스는 한 메시에만 할당합니다.
    """
    n = faces.shape[0]
    out = np.full(n, -1, np.int64)
    if n == 0 or boxes.shape[0] == 0:
        return out
    centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0,
                        (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)
    used: set[int] = set()
    for i in range(n):
        c = faces[i, :, :2].mean(axis=0)
        inside = [j for j in range(boxes.shape[0])
                  if j not in used
                  and boxes[j, 0] <= c[0] <= boxes[j, 2]
                  and boxes[j, 1] <= c[1] <= boxes[j, 3]]
        pool = inside or [j for j in range(boxes.shape[0]) if j not in used]
        if not pool:
            continue
        j = min(pool, key=lambda k: float(np.linalg.norm(centres[k] - c)))
        out[i] = j
        used.add(j)
    return out


@dataclass
class FaceSel:
    frame: int
    status: str
    mesh: np.ndarray | None = None      # (468, 3) 픽셀
    box: np.ndarray | None = None       # (4,) xyxy 또는 None
    n_mesh: int = 0
    n_box: int = 0


def select_face(frame: int, faces: np.ndarray | None,
                boxes: np.ndarray | None) -> FaceSel:
    """프레임당 얼굴 하나를 고릅니다. 규칙: **매칭된 박스 면적 최대**.

    박스가 매칭되지 않은 메시는 자기 바운딩 박스 면적으로 대체합니다 — box.csv 에
    항목이 없다는 이유만으로 프레임을 버리지 않기 위해서입니다.
    규칙이 프레임마다 흔들리면 크롭이 튀므로 하나로 고정합니다.
    """
    if faces is None:
        return FaceSel(frame, NO_ROW)
    n_mesh = faces.shape[0]
    n_box = 0 if boxes is None else boxes.shape[0]
    if n_mesh == 0:
        return FaceSel(frame, NO_MESH, n_mesh=0, n_box=n_box)
    if boxes is None:
        boxes = np.zeros((0, 4), np.float32)
    m = match_mesh_to_box(faces, boxes)
    areas = box_areas(boxes)
    score = np.empty(n_mesh, np.float64)
    for i in range(n_mesh):
        if m[i] >= 0:
            score[i] = areas[m[i]]
        else:
            xy = faces[i, :, :2]
            score[i] = float((xy[:, 0].max() - xy[:, 0].min())
                             * (xy[:, 1].max() - xy[:, 1].min()))
    i = int(np.argmax(score))
    return FaceSel(frame, OK, faces[i], None if m[i] < 0 else boxes[m[i]],
                   n_mesh=n_mesh, n_box=n_box)


# ------------------------------------------------------------------ 영상
def iter_frames(video_path: str, frame_indices):
    """요청한 0-based 프레임을 **순서대로** yield 합니다: (idx, bgr).

    시킹하지 않습니다. 원하는 프레임까지 grab() 으로 넘기고 필요할 때만 retrieve()
    합니다. h264 는 참조 프레임 때문에 건너뛸 프레임도 디코딩해야 하므로 grab() 이
    아끼는 것은 색공간 변환 정도입니다 — 즉 커버리지가 낮아도 사실상 전 영상을
    디코딩한다고 보고 시간을 잡아야 합니다.
    """
    want = sorted({int(i) for i in frame_indices})
    if not want:
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {video_path}")
    try:
        pos, k = 0, 0
        while k < len(want):
            while pos < want[k]:
                if not cap.grab():
                    return
                pos += 1
            ok, frame = cap.retrieve() if cap.grab() else (False, None)
            if not ok:
                return
            pos += 1
            yield want[k], frame
            k += 1
    finally:
        cap.release()


# ------------------------------------------------------------------ 세션
_USER_RE = re.compile(r"User[ _]?(\d+)")


class Session:
    """mEBAL2 피험자 1명 = 폴더 1개 = 세션 1개.

    경로를 명시적으로 받습니다 — _probe 추출본과 정식 Processed_Data 추출본의
    레이아웃이 다르기 때문입니다.
    """

    def __init__(self, user: int, landmarks_csv: str, box_csv: str,
                 blink_csv: str | None = None, unblink_csv: str | None = None,
                 video: str | None = None):
        self.user = int(user)
        self.subject = f"m{int(user):02d}"
        self.landmarks_csv, self.box_csv = landmarks_csv, box_csv
        self.blink_csv, self.unblink_csv, self.video = blink_csv, unblink_csv, video
        self._lm: dict | None = None
        self._box: dict | None = None

    @classmethod
    def from_probe(cls, probe_dir: str, user: int, video: str | None = None,
                   landmarks_csv: str | None = None, box_csv: str | None = None):
        return cls(user,
                   landmarks_csv or os.path.join(probe_dir, "PD_landmarks.csv"),
                   box_csv or os.path.join(probe_dir, "PD_box.csv"),
                   os.path.join(probe_dir, f"EyeBlinks_User {user}_Blink_Right_Blink.csv"),
                   os.path.join(probe_dir, f"EyeUnblinks_User {user}_Unblink_Unblink.csv"),
                   video)

    @property
    def landmarks(self) -> dict[int, np.ndarray]:
        if self._lm is None:
            self._lm = load_landmarks(self.landmarks_csv)
        return self._lm

    @property
    def boxes(self) -> dict[int, np.ndarray]:
        if self._box is None:
            self._box = load_boxes(self.box_csv)
        return self._box

    def events(self, include_possible: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """-> (events (n,3), is_blink (n,)).

        `Blink=0`("possible blink")은 기본 제외합니다. 배포자가 Blink=1 만으로
        Unblink 와 1:1 균형을 맞췄고 그게 58/58 전원에서 성립하기 때문입니다.
        포함 조건은 민감도 분석용입니다.
        """
        ev, isb = [], []
        if self.blink_csv and os.path.exists(self.blink_csv):
            b = load_events(self.blink_csv, has_flag=True)
            if not include_possible:
                b = b[b[:, 2] == 1]
            ev.append(b); isb.append(np.ones(len(b), np.int64))
        if self.unblink_csv and os.path.exists(self.unblink_csv):
            u = load_events(self.unblink_csv, has_flag=False)
            ev.append(u); isb.append(np.zeros(len(u), np.int64))
        if not ev:
            return np.zeros((0, 3), np.int64), np.zeros((0,), np.int64)
        return np.concatenate(ev), np.concatenate(isb)

    def face(self, frame: int) -> FaceSel:
        return select_face(frame, self.landmarks.get(frame), self.boxes.get(frame))

    def detection_stats(self) -> dict:
        """영상을 열지 않고 landmarks/box CSV 만으로 계산하는 검출 통계.

        58명 전량에 대해 **디코딩 없이** 돌릴 수 있어 Phase 1 의 선행 지표가 됩니다.
        """
        n = len(self.landmarks)
        n_ok = n_no_mesh = n_multi = n_disagree = n_unmatched = 0
        for frame, faces in self.landmarks.items():
            boxes = self.boxes.get(frame)
            nb = 0 if boxes is None else boxes.shape[0]
            if faces.shape[0] != nb:
                n_disagree += 1
            sel = select_face(frame, faces, boxes)
            if sel.status == OK:
                n_ok += 1
                if sel.n_mesh > 1:
                    n_multi += 1
                if sel.box is None:
                    n_unmatched += 1
            else:
                n_no_mesh += 1
        return {"frames": n, "ok": n_ok, "no_face": n_no_mesh,
                "no_face_rate": n_no_mesh / n if n else 0.0,
                "multi_face": n_multi, "multi_face_rate": n_multi / n if n else 0.0,
                "count_disagree": n_disagree,
                "count_disagree_rate": n_disagree / n if n else 0.0,
                "mesh_without_box": n_unmatched}
