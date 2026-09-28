# Evaluation protocol

`evaluation.json` contains all aggregate results, including metrics not selected for the overview. The public-English suite contains 18,098 examples; the prompt-family held partition contains 9,461. Think is held in full. WildGuard uses the English PolyGuard proxy; XSTest responses are omitted.

General-safety F1 uses original benchmark labels at threshold 0.5. v10 retains the validated shared backbone and general head, so the corresponding prior scores are reused after frozen-tensor verification. Red-line scores use v10 at its calibration-selected role thresholds. The same-example Qwen baseline uses its original threshold 0.5. Paper values in the summaries are reference values under the paper's own data setup, not directly comparable with held or proxy subsets.

Policy subsets use automated project-specific labels. The previously observed held suite is not a new independent confirmation set. Prompt checkpoint/threshold selection uses only calibration; the 734-example fresh audit is scored after selection. Fourteen of the 748 preselected audit records have unusable labels. Results are not an independent human gold standard.

The default response rule uses the maximum content-token cut score. Prompt scoring uses the final content token. Controversial probability is included in the cut score. The general and red-line heads are different operating modes and are never combined by picking the better head per test example.

For method details, sources and limitations see [technical_report.pdf](technical_report.pdf). Evaluation inputs and raw model outputs are excluded from this source release.
