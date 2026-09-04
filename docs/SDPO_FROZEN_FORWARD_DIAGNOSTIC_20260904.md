# Frozen-forward SDPO collapse diagnostic

Approved implementation only; root launches once after independent review. This is post-hoc mechanism analysis, not new training, a repair, a paper gate or independent validation.

Input is the verified v3 evidence whose MANIFEST SHA is `3765136f5452a3831c9a5620938a8b20bd28d0ad18c59465179b5f2bebc0fd41`. Its initial, step16 and final adapters are the only tested parameter states. No step4/5 adapter exists. Prepared v3 calibration bytes must match the archived preparation manifest.

## Frozen selection and probes

Select all eight originally correct calibration cases; then three JSON, three table and two bullet mismatches, each stratum sorted by case ID and taking the first required count. Assert32 source cases, eight original successes and enough records in each fixed stratum. No new-output-based selection.

For each case, checkpoint and context (16x3x4=192 forwards), teacher-force the original completion's native IDs. Contexts are original base, base with empty released hindsight block, original archived feedback-hindsight, and archived explicit-preference calibration. Thus no new feedback generation or simulator call occurs. Actual feedback and explicit contexts come from saved token IDs. Empty-block IDs come from the original prepared prompt and native tokenizer, with thinking disabled as in the source study.

Save full float32 vocabulary logits at offset0 and at the first non-special token after offset0 whose individual native-token decoding contains any of `: = ; | { } [ ] * -` or newline. Spaces are not delimiters. If no such second token exists, stop rather than choose another position. Selection uses token text only, before model loading/forwards; it is not a guarantee of a semantically sufficient action boundary. Save all completion-token log probabilities and both selected offsets.

## Derived quantities and scope

Using float64 CPU arithmetic, hold feedback and prefix fixed. Compute the categorical expectation and trace variance of `A_y(onehot(y)-p)`, where `A_y=logq_y-logp_y`, and compare the archived sampled token's local vector to this expectation. Separately compute the released student-top20-plus-tail reverse-KL logit gradient, with the same tail clamp. Norms and cosine comparisons use ascent/descent signs consistently.

These are LOCAL logit-space quantities conditional on one frozen feedback message and prefix. They are not globally unbiased SDPO policy gradients, do not marginalize the action-dependent feedback mechanism, and are not parameter/Adam replay. No vocabulary subset is labeled semantic-correct probability mass. Results can identify teacher displacement and saturation but cannot uniquely attribute the training collapse to LR, momentum, routing or estimator variance.

## Preservation and timing

Fresh output directory only. Hash input evidence, all three checkpoints, prepared calibration, all model/tokenizer files and executable sources before forwards. Validate model loading and exact loaded adapter tensors. Archive per-tensor loaded receipts, case IDs/context IDs/positions,192 small tensor files, JSONL records, environment and a final manifest. Preserve partial results on a time stop.

After12 forwards estimate remaining time from measured loop walltime, including CPU calculations and serialization, plus observed startup. The45-minute outer planning ceiling is not a forecast. Stop projected overrun at12 calls; thereafter check elapsed time before each forward with a60-second completion reserve. No training/generation calls or optimizer exist in the runner. A stuck individual CUDA call is not forcibly interrupted by an in-process check; root owns external timeout handling.

```sh
python -m interaction_sprint.sdpo_frozen_forward_diagnostic \
  /path/to/verified_v3_evidence /path/to/sdpo_format_control_v3 \
  /fresh/sdpo_frozen_forward_diagnostic \
  --model-path /path/to/Qwen3-4B/pinned_snapshot --threads 4
```

Independent verifier should reconstruct selection/context/position rules, compare logits and saved log-probabilities at the two positions, replay local moment/top20-tail calculations, verify hashes/loaded receipts, and classify completion versus partial diagnostic. It cannot claim independent neural-forward replay from saved logits alone.
