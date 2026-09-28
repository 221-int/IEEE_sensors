"""시드·결정성 봉인 — v2 단일 구현.

왜 이 파일이 존재하는가
----------------------
시드만 잡고 `cudnn.deterministic` 을 안 잡으면, 같은 시드에서 같은 숫자가 나오지
않습니다. 그러면 우리가 보고하는 ± 는 **시드 분산이 아니라 시드 분산 + 비결정성**이
되고, 그 상태에서 조건 A와 조건 B를 비교하면 차이가 어디서 왔는지 알 수 없습니다.

v2 규칙: `results/v2/` 에 들어가는 어떤 ± 숫자도, 이 모듈의 `seal()` 을 통과한
런에서만 나옵니다. `env_fingerprint()` 의 `deterministic` 이 False 인 결과 파일은
보고에 쓰지 않습니다.

PYTHONHASHSEED
--------------
이건 인터프리터 **시작 전에** 정해져야 효과가 있습니다. 실행 중에 os.environ 을
바꿔봐야 이미 만들어진 해시는 안 바뀝니다. 그래서 `ensure_hashseed()` 는 값이 없으면
환경변수를 세팅하고 **프로세스를 재실행**합니다. 엔트리 스크립트 맨 위에서 한 번
부르십시오.

사용법
------
    from src.v2.common import repro

    repro.ensure_hashseed()          # 최상단 (import 직후)
    repro.seal(seed=0)               # 시드 + 결정성
    ...
    json.dump({"env": repro.env_fingerprint(), ...}, f)
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np

HASHSEED_VALUE = "0"

_STATE: dict[str, Any] = {"seed": None, "deterministic": False, "sealed": False}


# ----------------------------------------------------------------- hashseed
def ensure_hashseed(value: str = HASHSEED_VALUE) -> None:
    """PYTHONHASHSEED 가 없으면 세팅 후 프로세스를 재실행합니다.

    재실행은 딱 한 번만 일어납니다(환경변수가 이미 있으면 그냥 통과).
    `-m` 실행을 보존하기 위해 sys.argv 가 아니라 원래 모듈 경로를 복원합니다.
    """
    if os.environ.get("PYTHONHASHSEED") == value:
        return
    env = dict(os.environ, PYTHONHASHSEED=value)
    mod = (sys.modules.get("__main__").__spec__.name
           if getattr(sys.modules.get("__main__"), "__spec__", None) else None)
    cmd = ([sys.executable, "-m", mod] + sys.argv[1:]) if mod else [sys.executable] + sys.argv
    print(f"[repro] PYTHONHASHSEED={value} 로 재실행합니다: {' '.join(cmd[:3])} ...",
          file=sys.stderr, flush=True)
    raise SystemExit(subprocess.call(cmd, env=env))


# ----------------------------------------------------------------- torch 선택적
def _torch():
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError:
        return None


# ----------------------------------------------------------------- 봉인
def seal(seed: int, deterministic: bool = True, warn_only: bool = False) -> None:
    """시드 고정 + 결정성 봉인.

    deterministic=True 이면:
        cudnn.deterministic = True, cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        CUBLAS_WORKSPACE_CONFIG=:4096:8   (CUDA >= 10.2 에서 필요)

    warn_only=False (기본) 이면 결정적 구현이 없는 연산에서 **예외로 죽습니다**.
    그게 맞습니다 — 조용히 비결정적으로 도는 것보다 낫습니다. 어떤 연산이
    걸리는지 알고 대체하려는 경우에만 warn_only=True 를 쓰십시오.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", HASHSEED_VALUE)

    torch = _torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=warn_only)

    _STATE.update(seed=int(seed), deterministic=bool(deterministic), sealed=True)


def require_sealed() -> None:
    """봉인 안 된 상태에서 결과를 쓰려는 것을 막습니다."""
    if not _STATE["sealed"]:
        raise RuntimeError("repro.seal(seed) 을 먼저 호출하십시오.")
    if not _STATE["deterministic"]:
        raise RuntimeError(
            "deterministic=False 로 봉인된 런입니다. 이 런의 결과로 ± 를 보고할 수 없습니다.")


# ----------------------------------------------------------------- DataLoader
def dataloader_kwargs(seed: int, num_workers: int = 0) -> dict:
    """DataLoader 에 그대로 넘길 결정성 인자.

    num_workers > 0 이면 워커마다 시드가 갈리므로 worker_init_fn 이 필요합니다.
    generator 를 주지 않으면 shuffle 순서가 전역 RNG 상태에 의존해, 앞에서 난수를
    몇 번 뽑았는지에 따라 배치 순서가 바뀝니다.
    """
    torch = _torch()
    if torch is None:
        return {"num_workers": num_workers}
    g = torch.Generator()
    g.manual_seed(int(seed))

    def _worker_init(worker_id: int) -> None:
        s = (int(seed) + worker_id) % (2 ** 32)
        np.random.seed(s)
        random.seed(s)

    return {"num_workers": num_workers, "generator": g,
            "worker_init_fn": _worker_init if num_workers > 0 else None,
            "persistent_workers": False}


# ----------------------------------------------------------------- 지문
def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def env_fingerprint() -> dict:
    """결과 JSON 마다 통째로 박아 넣을 환경 지문.

    나중에 "이 숫자 어떤 환경에서 나온 거지"를 파일만 보고 답할 수 있어야 합니다.
    """
    torch = _torch()
    out = {
        "seed": _STATE["seed"],
        "deterministic": _STATE["deterministic"],
        "sealed": _STATE["sealed"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    if torch is not None:
        out.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        })
    else:
        out["torch"] = None
    return out
