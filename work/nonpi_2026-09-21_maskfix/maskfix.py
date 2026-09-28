"""Experiment-local correction: only gather real frames; missing tensors are zero.

The original trainer and dataset remain unchanged for provenance. This subclass
replaces only Bundle.batch for the new, explicitly versioned experiments.
"""
import numpy as np
from src.v2.train_encoder import Bundle as OriginalBundle,EVENT_LEN
from src.v2.dataset import crop as C

class MaskSafeBundle(OriginalBundle):
    def batch(self,ids):
        rows=self.rows[ids];present=rows>=0
        x=np.zeros((len(ids),EVENT_LEN,1,C.OUT_H,C.OUT_W),np.float32)
        if present.any():
            # Missing positions never index frames[0] or any other person's data.
            x[present]=C.batch_input(np.asarray(self.frames[rows[present]]))
        return x,present.astype(np.float32),self.y[ids]
