# Memory codec follow-up: v1 invalid schema assay

All192 Qwen calls completed. Remote/local manifest validation and scoring agree.
Reasoning25/48 correct with all schema/EOS valid. All three extraction arms0/48
schema-valid. Named-field EOS38/48; other arms48/48. Inference349.795seconds,
model loading2.444seconds; hashing and other setup are excluded.

Researcher error: the new extraction instructions did not explicitly require
the exact two top-level keys and integer-array values that the scorer required.
The original successful extraction prompt did. Raw responses include events
objects/lists and a threshold field. Do not reinterpret or normalize these into
passing results, or infer graph incompetence from the zero strict score.

V2 prospectively repairs that contract for all extraction arms equally. It adds
explicit two-key and integer-array requirements; retains every input,192-token
cap, decoding convention, reasoning control and scoring rule. No semantic
few-shot examples. This is a posthoc apparatus repair on already exposed cases,
not fresh confirmation. V1 is preserved. If the repaired contract still fails,
record the failure; do not silently change the parser or choose the winning arm.

The score mode and prepared oracles were replayed locally. This does not certify
every inference detail; next verify model hashes and exact prompts/token IDs.
Nemo remains unrun and no method-training campaign is admitted.

## V2: completed, schema improved, semantic qualification failed

All192calls completed in197.217 inference seconds,2.434loading seconds.
Local manifest replay agrees with remote; all11 recorded model/tokenizer hashes
match v1. ManifestSHA25694df2896782726a5b7f40fd8f57d2df3fc725bef8297c228c533ee587426b8b3.

| Arm | Valid schema | EOS | Correct graph | Correct action |
|---|---:|---:|---:|---:|
| Reasoning |48/48|48/48|not applicable|25/48|
| Earlier/later pairs |48/48|48/48|20/48|36/48|
| Named fields |44/48|48/48|16/48|34/48|
| Later/earlier pairs |47/48|48/48|28/48|40/48|

Every extraction encoding fails the frozen95%semantic gate. Named fields also
fail schema qualification. No Nemo download/run is admitted under the declared
stop rule; no third prompt repair or best-arm confirmation follows.

Posthoc raw diagnosis: correct actions despite incorrect graphs occur16,18,12
times respectively in the three extraction arms. Thus action accuracy alone
overstates faithful state extraction in this setting. This is not surprising
mathematically: many distinct graphs/values imply the same threshold answer.
No output was corrected. Numeric arrays are exact in35,38,34cases respectively;
complete reversal of nonempty gold edges occurs5,4,0times. Unlike old Nemo,
the broader Qwen failures are not solely edge reversal.

The earlier Qwen48/48 result remains a narrow observed result, but does not
generalize to this expanded synthetic generator. V2 changes graph size, ID
assignment and instructions relative to that old experiment; it cannot isolate
which difference caused the loss. The within-v2 codec comparisons share inputs.
The named-field primary contrast is negative descriptively (16versus20correct
graphs), not evidence for a universal encoding rule. Views are dependent; there
are24generated cases, not48independent tasks. No natural-memory or method-learning
claim survives from this diagnostic. Paper remains NO-GO.

Artifacts: artifacts/memory_encoding_followup_20260910/memory_encoding_qwen_v2
and QWEN_V2_DIAGNOSIS.json. Raw v1 and v2 are both retained. Combined recorded
inference is547.012seconds, excluding setup, transfers and idle rental time.
