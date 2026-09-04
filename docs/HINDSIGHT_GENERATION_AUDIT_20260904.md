# Teacher generation diagnostic (frozen before execution)

Question: is the failed full-vocabulary teacher control partly a first-token
format artifact? The source remains the completed, unchanged64 truthful teacher
prompts from `hindsight_full_feedback_cpu_20260904_v1`. Retain every prompt;
no filtering to incorrect or low-mass cases. This is a post-hoc apparatus audit,
not a fresh confirmation set or a change to that run's stopping rule.

Generate greedily from the same pinned Qwen3-0.6B, CPUfloat32, up to24 tokens,
native EOS, thinking disabled. Compare full first-token log probabilities to
the saved teacher matrix; stop on max error>.001. Token IDs must replay exactly.
No training. Save complete outputs, capped/EOS status, source hash and manifest.

Primary diagnostic: entire decoded response after whitespace stripping must be
exactly A or B, correct relative to the same target. Compare with literal first
generated token. Also report a deliberately exploratory unique A/B-mention
parser. It cannot understand negation or rationale and is **not** semantic
correctness. Inspect decoded outputs before interpreting it. Report capped
responses, which may not represent completed answers. No model/prompt search,
automatic expansion, lowered threshold or paper greenlight.

Compute feasibility: only0.6B has cached language-model weights;4B/9B directories
contain metadata. The laptop had about5.1GiB free physical RAM at inspection.
No larger weights downloaded and no cloud instance contacted.
