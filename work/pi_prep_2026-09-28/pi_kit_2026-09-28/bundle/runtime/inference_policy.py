"""Shared online eligibility and validation-threshold contract.

19 time positions, >=14 valid crops, valid current crop, stride 1 by default.
Missing positions never disappear from the time axis. Contiguous threshold
crossings define candidate events; they are not eyelid-duration estimates.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

EVENT_LEN = 19
MIN_VALID = 14
POLICY_ID = 'min14_current_valid_v1'

def eligible_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[-1] != EVENT_LEN:
        raise ValueError('Expected exactly 19 time positions')
    return (mask.sum(axis=-1) >= MIN_VALID) & mask[..., -1]

def ring_ready(ring):
    return len(ring) == EVENT_LEN and bool(eligible_mask([x is not None for x in ring]))

def advance_missing(ring, count):
    """Retain skipped source-frame positions without unbounded work."""
    for _ in range(min(max(int(count), 0), EVENT_LEN)):
        ring.append(None)

def candidate_onset(probability, threshold, was_positive):
    """Abstention breaks a candidate interval, exactly like masked offline replay."""
    positive = (probability is not None and np.isfinite(probability)
                and probability >= threshold)
    return bool(positive), bool(positive and not was_positive)

def load_contract(model_dir, graph_paths, explicit_path=None):
    """Load threshold only after verifying it belongs to the actual ONNX graphs.

    No contract returns None for latency-only runs. An explicitly requested or
    present but incompatible contract is an error, never a fallback to 0.5.
    """
    path = Path(explicit_path) if explicit_path else Path(model_dir)/'contract.json'
    if not path.exists():
        if explicit_path:
            raise FileNotFoundError(path)
        return None
    data = json.loads(path.read_text(encoding='utf8'))
    if data.get('event_len') != EVENT_LEN or data.get('min_valid_frames') != MIN_VALID:
        raise ValueError('Model contract window/missing policy differs')
    if data.get('weights') != 'trained':
        raise ValueError('A validation threshold requires trained model weights')
    if data.get('input', {}).get('norm') != 'frame_standardize':
        raise ValueError('Model normalization contract differs')
    if data.get('head_output') != 'probability (sigmoid in graph)':
        raise ValueError('Expected probability ONNX head, not logits')
    for p in graph_paths:
        p = Path(p)
        if hashlib.sha256(p.read_bytes()).hexdigest() != data.get('graph_sha256', {}).get(p.name):
            raise ValueError('Graph hash mismatch: '+str(p))
    thr = float(data['threshold_probability']); logit = float(data['threshold_logit'])
    if not math.isfinite(logit) or not math.isfinite(thr) or not 0 < thr < 1:
        raise ValueError('Invalid threshold')
    expected = 1/(1+math.exp(-logit)) if logit >= 0 else math.exp(logit)/(1+math.exp(logit))
    if abs(thr-expected) > 1e-7:
        raise ValueError('Logit/probability thresholds disagree')
    return data

def metadata(contract=None):
    return dict(policy=POLICY_ID,event_len=EVENT_LEN,min_valid_frames=MIN_VALID,
                require_current_valid=True,missing_time_positions='retained',
                event_rule='contiguous threshold positives; abstention breaks interval',
                center_offset_frames=9,
                threshold_probability=None if contract is None else contract['threshold_probability'],
                threshold_source='unavailable; scores only' if contract is None else 'model contract: source validation')
