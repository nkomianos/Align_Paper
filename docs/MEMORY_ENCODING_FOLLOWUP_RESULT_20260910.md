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
