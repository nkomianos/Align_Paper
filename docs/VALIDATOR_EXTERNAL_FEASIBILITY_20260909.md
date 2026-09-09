# Natural-repository follow-up feasibility

The current G0 has inadequate common-task support. This note investigates the
already proposed external benchmark path; it does not authorize changing G0.

[SEC-bench](https://arxiv.org/html/2506.11791) focuses on C/C++ memory safety,
using containerized builds and sanitizer-based vulnerability reproduction. It is
a possible external setting, but its interfaces are not interchangeable with
our small Python function/vector harness.

[PatchEval](https://arxiv.org/html/2511.11019) distinguishes patch-equivalence
evaluation from dynamic PoC validation. Our follow-up would need the latter,
not similarity to an official fix as a substitute for executable ground truth.
The paper also acknowledges test incompleteness and possible pretraining overlap.

Inspected the public [PatchEval repository](https://github.com/bytedance/PatchEval)
at b43285cdde80cc04608d5f1178a330b740c91c2d. Downloaded its README, evaluator README,
and patcheval_verified.json with source hashes in
artifacts/validator_external_sources_20260909/PATCHEVAL_DOWNLOAD.json. The actual
metadata contains 230 cases: 70 Python, 77 JavaScript, 83 Go; all have an image URL.
Fields include vulnerable/fixed function content and official patch references,
which must not enter a blinded test-writer input by accident.

The evaluator documentation says it executes case validation inside a CVE Docker
image and infers failure categories heuristically from logs. That is not yet
independently verified multi-defect ground truth. Before admitting a neural
experiment, inspect evaluator code, resolve immutable image digests, reproduce
vulnerable/fixed outcomes, and establish public regressions plus multiple
independently repairable hidden security cases. A single PoC cannot automatically
identify our repaired-one-defect/retained-another eligibility population.

No external container, patch generation or test-generation experiment has been
run. The 70 Python entries are a candidate source pool, not 70 qualified tasks.
