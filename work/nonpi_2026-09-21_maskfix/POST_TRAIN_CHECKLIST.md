# Corrected model follow-through

The main queue automates the 60 fits and paired source comparisons. A separate `post_train.py watch` process waits for completion, then automates four-candidate subject diagnostics, replay of the saved provider samples, and the previously observed EyeBlink8 video9 pilot. It does not send chat notifications. Check `post_train/status.json`; a waiting state is not completed inference. The following review and remaining work still apply.

1. Inspect all completed-fit validation, convergence and matched-label checks. Report all 15 fits per candidate; investigate failures without selecting only good seeds.
2. Recompute source pooled, run-mean, subject-macro and NI margin sensitivities. Preserve historical results in a separate comparison, with input correction and environment differences disclosed. Do not use old vpres AP as the corrected model AP.
3. Recompute U1/U54/all-user diagnostics with corrected held-out models. Existing crop/index/labels are unchanged; old prediction summaries are obsolete for final claims.
4. Re-infer the fixed 8-person/160-event provider sample and the 17 diagnostic cases from ../nonpi_2026-09-21/submission_audit/*features.npz. Use matching corrected checkpoints, source-selected validation thresholds, explicit eligibility and abstention reporting. Preserve CPU-versus-saved-reference checks.
5. Re-infer EyeBlink8 video9 from saved features as exploratory data. Evaluate the full official archive when available, with subject mapping and annotation version recorded. Never tune on this external test and call it untouched.
6. Resolve the two source re-crop mismatches (U19/33969, U55/49892) before claiming universal crop equivalence. The observed pixel differences are documented and must not be silently ignored.
7. Refresh trained deployment exports and thresholds from the final selected corrected checkpoint. Legacy ONNX exports and Pi timings do not automatically become corrected-model measurements.
8. Align benchmark/demo/evaluation missing-frame rules and source validation thresholds in a separate reviewed implementation. A corrected training input does not itself fix all deployment-policy differences.
9. Obtain RNN raw outputs or transparently limit/reproduce those comparisons. The current handoff archives contain summaries/code only.
10. Finish Pi measurement if keeping broad real-time energy claims. Only then revise manuscript, tables, figures and PDF, consistent with the user's hold.
