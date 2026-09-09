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

## First container inspection

Selected the lexicographically first Python CVE for infrastructure qualification,
CVE-2015-1326 (python-dbusmock), without selecting on model outcomes. Resolved
its public image to sha256:7084bf21accddef3980657933957bd2cf4792b8b03c329471361853ec9934840
and inspected files from a stopped container. The evaluation entrypoint applies
the supplied patch, copies a fixed test file from /workspace/poc, and executes
tests/test_api.py::TestTemplates::test_local. The test checks local template
loading and absence of bytecode cache files. It is one test method with several
assertions, not demonstrated independent repairable defects.

The source evaluator regards successful script exit as repair success; its
failure categorization is heuristic. Inspect each case's assertions and validate
both baselines before interpreting this as a security oracle. Do not execute
prepare.sh during evaluation: it resets the repository and the release warns
that preparation can alter the prepared baseline.

Started a CPU-only baseline check in a separate container with no network, host
mounts or GPU access, dropped capabilities, two CPUs, 2 GB RAM and a 256 PID limit.
It copies the released test file and runs that exact test without applying a
candidate patch. Container patcheval_baseline_2015_1326 remains running at the
last check; pytest has displayed F, but teardown is not complete. Do not classify
the result from that marker alone. Remote log: patcheval_audit/vulnerable_baseline.log
under align_run_20260909; live exec session 1125. Image pull session 63229 finished.

Baseline session 1125 is now terminal. Inspection found stop_dbus in
dbusmock/testcase.py sets timeout=50 but never decrements it inside while timeout
> 0. The daemon remained present and pytest stalled in cleanup. Sent SIGINT to
this diagnostic container; preserved exit 2 and the complete traceback. The test
itself failed at test_api.py:600 because cache_from_source(template) existed,
which is the advertised vulnerability assertion. KeyboardInterrupt occurred in
testcase.py:129 during cleanup. Distinguish reproduced assertion from a clean
completed assay: neither an exit-2 run nor the initial F alone qualifies the
benchmark. Original container and files remain preserved. A bounded, documented
fixture correction must be applied equally to vulnerable and fixed baselines
before admitting this case. No test assertion or security condition should be
changed to force baseline separation.
