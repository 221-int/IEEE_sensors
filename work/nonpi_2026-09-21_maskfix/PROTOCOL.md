# Corrected missing-input experiment — frozen before new fits

2026-09-21. The submission audit discovered that OriginalBundle.batch substitutes frames[0] (U1) at all missing positions. CNN BatchNorm runs before the temporal mask, so this substitution influences valid embeddings and gradients. U1 is validation in rotation 0 and test in rotation 1. Direct valid-frame split integrity is insufficient to detect this route. See ../nonpi_2026-09-21/submission_audit/masked_placeholder_audit.json.

## Correction

Only gather valid frame rows. Initialize missing normalized tensors to exact zero before any encoder operation. Retain the temporal mask, all event labels, e_valid filtering, the frozen folds, existing encoder/head architectures and optimization settings. Constant zero inputs still contribute to CNN BatchNorm statistics; the correction removes other-person image content, not all possible effects of missingness. A valid-frame-only BatchNorm/training redesign is a different experiment.

Original src/, data and final results remain unchanged. The corrected Bundle lives in this folder and is installed by the isolated runner. Never merge legacy and corrected fits.

## Execution and reporting

- Run vpres, vdrop, Image-CNN+TCN, GAP+TCN: all five subject-disjoint folds x seeds 0/1/2 = 60 fresh fits, in the same current environment. Do not import the six old GAP runs or the old Image-CNN fits, even though the mechanism test showed zero masked-placeholder influence for Image-CNN without encoder BatchNorm.
- Unchanged D=16, 19 frames, frame_standardize, Adam lr 0.001, batch 32, max 80 epochs, validation AP selection, patience 10, validation-accuracy threshold selection. No test-driven tuning or participant exclusion.
- Save CNN and EAR-head checkpoints, raw test scores, source hashes, environment and validation thresholds. Each fit has its own attempt folder and atomic verified record.
- Report per-fit AP/recall/precision/F1, pooled paired subject-bootstrap and subject-macro AP differences, 2,000 subject draws. Historical NI margin 0.02 remains primary; 0.01/0.005 are disclosed sensitivities, not retrospectively adopted margins.
- Compare corrected vpres vs corrected Image-CNN as primary, then GAP/vdrop tradeoffs. Assess old-vs-corrected differences only after complete outputs. Direction/magnitude of the bug's metric impact are unknown before refitting.
- All architecture selection remains exploratory: the source test data and historical results have been viewed. External data must not be used to tune thresholds/architectures and then claimed as untouched validation.
- Existing external pilot, trained ONNX files and landmark-provider audits concern legacy checkpoints. They demonstrate issues but are not corrected-model validation. Refresh prediction-dependent audits/exports when corrected models are complete.
- No manuscript/table/PDF edits, no new Pi measurements, no privacy claims.

## Stop/restart

Status and timestamp are in status.json. Resume at verified completed-fit boundaries. Incomplete attempts are preserved and restarted, not overwritten. A live previous child prevents duplicate execution. Any fit failure is explicit and stops the queue.
