# Results so far — On-Device Privacy-Preserving Eye-Blink Representation

_Status snapshot. One encoder branch (custom autoencoder) taken end-to-end
on a public dataset with subject-separated validation._

## Direction (re-scoped)
Detect blinks (eye-health primitive) from a **privacy-preserving vector**
instead of raw pixels / EAR, running lightweight on the **edge (Pi5, ONNX)**.
Privacy is **structural**: only vectors are stored/transmitted and no decoder is
deployed, so raw eye video is not recoverable in deployment.
Input = **both eyes in one crop → one vector** (blinks are bilateral).
**Identity re-identification / disentanglement is moved to FUTURE WORK**
(needs many subjects); the identity results below are preliminary, kept as a
future-work direction, not a current claim.
_The numbers below were measured on the earlier ONE-eye pipeline; they will be
re-run after the both-eyes rebuild + retrain._

## Pipeline (all validated)
```
eye crop (both eyes, gray 64x160)  ->  encoder  ->  128-d vector  ->  blink O/X
   [canonical crop, shared]           [autoencoder]                 [MLP classifier]
```
No landmarks / EAR in the judgment path. Decoder = local validation tool only.

## Data
- **Eyeblink8**, 8 clips -> both-eyes 64x160 crops (~35k frames, 30->15 fps).
  (Rebuild the .npz after the both-eyes switch.)
- Labels (manual, EAR-independent): `blink_event` 5.25%, `eye_closed` 2.49%.
- **Subject-separated** val = clips eb04, eb11 (74 blink events).

## Utility — vector model vs EAR (same clips, same metric, one eye)
Event-level blink recall vs frame false-alarm:

| Operating point | EAR baseline | Vector model (ours) |
|---|---|---|
| false-alarm ≈ 1.5% | recall 0.946 | recall 0.932 |
| recall ≈ 0.96 | false-alarm 4.8% | false-alarm **3.1%** |

**Parity in the usable region** (differences ≈ 1 event / 74). EAR's false-alarm
explodes when pushed for recall (13–50%); the vector model stays < 3%.
→ Phase-0 "≥ EAR baseline" met. See `docs/figures/vector_vs_ear.png`.
Autoencoder reconstruction sanity: val MSE 0.003 (`models/autoencoder/recon_val.png`).

## Privacy (current, NOT-yet-disentangled encoder)
| Metric | Value | Meaning |
|---|---|---|
| Identity re-ID (vector -> subject) | **0.998** (chance 0.125) | vector fully carries identity |
| Reconstruction PSNR | 32.4 dB | eye appearance recoverable |

→ The reconstruction-trained vector is **not private yet**. This is the
**left endpoint of the utility–privacy Pareto curve** and the quantitative
motivation for identity-disentanglement.

## Disentanglement PoC (FUTURE WORK — preliminary, one-eye pipeline)
Encoder trained with a blink head + gradient-reversal identity adversary, with
lambda warmup (DANN schedule) + gradient clipping. Frame-split over all 8
subjects; utility = blink ROC-AUC, privacy = identity re-ID.

| lambda | blink AUC (utility) | re-ID (leak, chance 0.125) |
|---|---|---|
| 0.0 | 0.984 | 0.942 |
| 0.1 | 0.988 | 0.913 |
| 0.2 | 0.983 | 0.868 |
| 0.5 | 0.975 | 0.819 |
| 1.0 | 0.977 | 0.767 |

**Identity leakage drops (re-ID 0.94 → 0.77) at essentially zero utility cost
(AUC stays ~0.98)** → the core hypothesis holds: identity is removable while
keeping blink utility. See `docs/figures/pareto_disentangle.png`.
Not yet at chance: full removal needs higher lambda / better method /
many-identity data. (Naive fixed-lambda GRL was unstable; warmup fixed it.)

## Where we are (of the 9-step plan)
Done: 1 preprocess, 2 (train/val, test split pending), 3-B autoencoder,
4 decoder check, 5 single-frame judgment, **6 evaluation (utility + privacy)**,
**disentanglement PoC (mechanism validated)**.
Pending: extend lambda sweep to trace the full frontier; 3-A off-the-shelf
encoder, 7 A/B compare, 8 ONNX + Pi latency, 9 registration window; and a
many-identity dataset for the rigorous Pareto claim.

## Caveats
- Small val (2 subjects, 74 events): "parity" is the honest claim, not "better".
- re-ID with 8 clips is proof-of-concept; a many-identity set is needed for a
  strong privacy claim.
- EAR baseline uses one eye (apples-to-apples with our one-eye model).
