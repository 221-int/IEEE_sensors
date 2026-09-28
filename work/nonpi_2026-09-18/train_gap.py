"""Isolated lightweight control; existing trainer and data remain untouched."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.v2.common import repro
repro.ensure_hashseed()
import torch
from torch import nn
from src.v2.model import encoder as E
from src.v2 import train_encoder as T
torch.set_num_threads(4)
original=E.build_image_cnn
def build_gap(out_dim=16, in_ch=1, dropout=.5):
    m=original(out_dim,in_ch,dropout)
    m.fc=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,64),
                       nn.ReLU(True),nn.Dropout(dropout),nn.Linear(64,out_dim))
    return m
E.build_image_cnn=build_gap
original_freeze=T.freeze_env
def freeze():
    env=original_freeze()
    env['control_architecture']='image_cnn_gap: adaptive_avg_pool2d(1), 64->64->16, shared TCN'
    env['torch_num_threads']=torch.get_num_threads()
    return env
T.freeze_env=freeze
if __name__=='__main__':
    raise SystemExit(T.main())
