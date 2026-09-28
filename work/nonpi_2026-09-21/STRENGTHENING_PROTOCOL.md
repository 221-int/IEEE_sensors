# Non-Pi submission evidence — 2026-09-21

User authorized continuing the remaining submission-strengthening work. Manuscript, tables and PDF remain frozen until experiments and analysis finish. Existing source code and original results remain unchanged.

## Fixed scope before the additional analyses

- Keep the existing GAP/vdrop queue running; no competing GPU training or test-guided hyperparameter changes.
- Audit final source score sidecars against the processed index, fold membership, labels, frame ownership and validation threshold provenance. Inventory RNN archives; summaries alone are not raw predictions.
- Post-hoc failure analysis: join the three held-out seed predictions per event for all 57 users, comparing ours, Image-CNN+TCN and EAR-head at their frozen validation thresholds. U1/U54 are requested cases; all-user context prevents presenting them as random cases. Majority votes are diagnostic summaries, not the original deployed classifier or a replacement primary metric.
- Review fixed example selection: first four seed-consistent errors of each error type by original event index, and first two correct events per label. Show all 19 frames, not only the center. No relabeling or exclusion from primary results on this basis; observations are hypotheses until verified with raw context/independent annotation.
- Deployment audit separates shared-code equivalence from changing landmark providers. Check actual missing-frame behavior and eligibility in the video runner. Synthetic masks are controlled stress tests, not empirical field dropout rates.
- Seek the official EyeBlink8 archive; preserve the source URL, response status and hash if acquired. Do not bypass a security challenge or contact third parties. Video 9 is already observed exploratory data. Other videos are unobserved at the time of this protocol, but subject overlap with video 9 must be documented. Freeze all evaluation rules and source thresholds before new inference; no external-test tuning.
- External event units are binocular blink IDs. Retain detection failures in recall denominators, report coverage and conditional window metrics separately. Include one-to-one interval matching at any overlap and IoU thresholds 0.1, 0.2, 0.5; none is substituted after viewing results. These are not directly comparable with published per-eye completeness benchmarks.
- Pi latency/power remeasurement is pending hardware and is not simulated by desktop timing. This work makes no privacy claims.

Completion is evidence-based, not simply scripts prepared: record completed, running, blocked and deferred items separately. Never claim the full external evaluation or new training comparisons complete before their validated outputs exist.
