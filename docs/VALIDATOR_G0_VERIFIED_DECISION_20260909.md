# Validator monoculture G0: verified decision, 9 September 2026

The registered decision is **INCONCLUSIVE_INSUFFICIENT_APPARATUS_POWER**. It is an invalid assay for a family-specific monoculture conclusion, not a valid negative and not a paper-qualified result.

All 460 completions were collected. The unchanged frozen verifier reconstructed the raw evidence and completed at approximately 09:03 UTC. Report: artifacts/validator_deployment_20260909/validator_g0_v3_verified.json, SHA256 dced9efa12c40e50efd3bc05cd5fb9d7c3e94dbd7083c491eb32609348cb6998, matching remote and receipt. Verification succeeded after preserving the virtualenv executable path and limiting BLAS/OpenMP threads to one; scoring source, memory cap, timeouts and thresholds were unchanged.

| Finding | Correct interpretation |
|---|---|
| 147 fully correct under frozen oracle; 35 qualifying incomplete patches; 10 other rejected generations | Task set often too easy for the desired incomplete-repair population; correctness is bounded by this oracle. |
| Common held-out support: 4 patches, 1 task, 1 CWE; required 30 patches, 4 CWEs | Insufficient independent support. More views of the same task do not repair this. |
| Planted defects: Qwen 13/16, Gemma 0/16 | Strong assay asymmetry; cannot attribute this to semantic security capability before resolving interface confounds. |
| Strict spec-only suites accepted: Qwen 58/64, Gemma 2/64 | Gemma fails the interface under this prompt/parser pair. Raw inspection identifies harmless extra case_id metadata in 41 rejected suites. |
| Registered crossed effect 0; directions +1 and -1 on the single shared task | Descriptive only. Reflects verifier asymmetry, not an identified self-family penalty. No calibrated interval exists with one CWE. |
| No indeterminate generated-test executions | Execution integrity survived; this does not rescue measurement power or task coverage. |

The prompt says vectors must have args, kwargs and expected but does not explicitly prohibit extra case_id metadata; the parser requires an exact key set. Public regression examples also include case identifiers. Preserve this as a prompt/parser mismatch, not proof Gemma cannot generate useful tests. Do not silently reinterpret the registered scores.

## Bounded next diagnosis

A CPU-only, explicitly post-hoc adapter removes optional string case_id fields only; it preserves every input, expected value, proposal slot, frozen reference and planted-mutant selection. It rejects other extra fields and malformed JSON. No primary endpoint is recalculated. The implementation is scripts/diagnose_validator_case_id_transport.py. This tests whether the observed power failure comes from discarded metadata. It is developmental and cannot become confirmation evidence even if detection improves. Initial replay recovers 41 Gemma suites, leaving 21 invalid; Qwen unchanged. Semantic planted-control replay is pending.

Do not repeat GPU generation on this corpus. If the adapter restores planted-control power, lock a clearer transport schema and qualify it on fresh DEV tasks before any expansion. Independently require a task source with enough naturally incomplete patches from both families and multiple independent defects. The single externally qualified PatchEval PoC establishes vulnerable/fixed separation only and does not yet meet that requirement. If these qualifications cannot be achieved, stop this formulation rather than manufacturing a paper from the current result.
