"""One corrected fit; checkpoints and result files are isolated from legacy runs."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.v2.common import repro
repro.ensure_hashseed()
import torch
from src.v2 import train_encoder as T
from maskfix import MaskSafeBundle
T.Bundle=MaskSafeBundle
torch.set_num_threads(4)
variant=sys.argv.pop(1)
if variant=='gap':
    sys.path.insert(0,str(ROOT/'work/nonpi_2026-09-18'))
    import train_gap
elif variant not in ['vpres','vdrop','image_head']:raise ValueError(variant)
if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refusing silent CPU training')
raise SystemExit(T.main())
