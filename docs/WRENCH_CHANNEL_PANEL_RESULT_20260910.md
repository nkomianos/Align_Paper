# Paired channel panel: verified result and transport correction

Qwen3-32B completed92 outputs on23 matched source tasks in148.93 inference seconds,
plus model loading and hashing. There were19 DEV tasks. No follow-up is admitted.

The original strict one-key JSON parser accepted only2/92 outputs, despite92/92
EOS. Its frozen result is INVALID_PANEL_QUALIFICATION. The other90 responses were
single JSON score objects inside Markdown fences. The model did supply scores;
this is an interface failure, not evidence of inability to assess the transcript.
We should have checked transport on a small preflight before all92 calls. Future
scoring protocols must define lossless wrapper handling before neural collection.

audit_wrench_fenced_transport.py removes exactly one outer JSON code fence and
then applies the unchanged strict payload schema. It rejects additional prose,
extra keys, booleans and nonfinite/out-of-range values. One targeted test covers
accepted/rejected wrappers. This recovers92/92 values without modifying a single
numeric answer. The old parser result is preserved, and the new analysis is
explicitly post-hoc; no threshold or numerical prediction was changed.

All19 DEV baseline-cohort paired differences are exactly zero. The mean hack
cohort stripped-minus-full score shift is-.3947368. Thus the proposed baseline
channel-shift hypothesis has no descriptive signal in this panel, even after
recovering every response. Do not expand it under the frozen.10 practical rule.
The hack-side shift is consistent with known reasoning-removal monitorability
effects and is not novel. These scores are not calibrated probabilities or a
claim that source labels independently establish honest/malicious intent.

Independent input reconstruction from all48 pinned raw traces verifies96 prepared
views, exact unchanged tool calls/observations, and one user step in every raw
trace. No later user context was accidentally dropped in this panel. The19 DEV
tasks remain the units; source/model repetitions are not additional task samples.
Reconstruction script verify_wrench_channel_inputs.py and receipt
artifacts/terminal_wrench_source_20260910/CHANNEL_INPUTS_VERIFIED_V1.json.

Stage5 archive remote/local SHA256:
32dd7d188071500d7646709956f4237c25802fb9793a39de97ad86cfb11d76f6.
Local root artifacts/gh200_research_20260910/retrieved_stage5/wrench_channel_qwen32_v1.
Both remote and local endpoint replay agree. Raw manifest SHA256:
d2613eaf24a4b666bf6f39878eca7beac23acf51b317d2e1a5aeabca862e905f.
The separate lossless-transport report is wrench_fenced_transport_v1.json in the
GH200 artifact root. No raw scores or manifests were overwritten.

Paper status remains NO-GO. This result closes the proposed baseline-shift screen
for this panel; it does not invalidate the source's asymmetric hiding experiment
or rule out channel effects for other monitors, task distributions or transforms.
