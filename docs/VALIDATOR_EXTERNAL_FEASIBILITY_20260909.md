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

## Matched bounded-cleanup baseline result

At approximately 08:13 UTC, ran two fresh isolated containers from the same pinned image, both with exactly one fixture correction: decrement stop_dbus timeout after its existing 0.1-second sleep. No security assertions changed. Vulnerable baseline exited 1 with the expected bytecode-cache assertion failure at test_api.py:600 (11.45 seconds). Released fix.patch applied cleanly with git apply --check; fixed baseline exited 0 with one test passed (11.38 seconds). Cleanup remains capable of reaching its timeout and reporting a warning; it now terminates. This qualifies this one released PoC to distinguish its vulnerable and fixed baselines under the documented fixture amendment. It does not establish multiple independent defects, public regression coverage, or a validator-monoculture task. Classification: developmental apparatus positive only.

Preserved containers patcheval_bounded_vulnerable_2015_1326 and patcheval_bounded_fixed_2015_1326 are TERMINAL. Initial fixture-copy attempt failed on root-owned destination before any new container launch; retry used a separate user-owned amended file. Full launch commands, Docker states, logs, fixture and released patch are backed up in artifacts/validator_external_sources_20260909/bounded_baselines.tar.gz; local/remote SHA256 e3dde59ce2b539af0699c7d845f81b80c58df318340c193020bdfbcbdd194fef. No external neural run admitted.

## Snowflake two-defect component candidate

Inspected CVE-2024-49750 in the pinned PatchEval metadata, then checked the [upstream fix](https://github.com/snowflakedb/snowflake-connector-python/commit/dbc9284a3c0382c131b971b35e8d6ab93c46f37a). The secret_detector component changes two regex assignments: private-key matching and dotted-token matching. The broader commit also changes authentication logging; do not claim the component captures the entire CVE.

Downloaded full vulnerable/fixed component source and upstream tests at immutable commits; hashes in artifacts/validator_external_sources_20260909/snowflake_source/DOWNLOAD.json. Compared ASTs after excluding the two pattern assignments: otherwise identical. Executed the actual mask_secrets method on synthetic non-secret key/token probes, with four states: vulnerable, private-pattern-only repair, token-pattern-only repair, fully fixed. Each partial repair resolves exactly one probe and retains the other defect; fixed resolves both. All states preserve a benign string and redact the legacy non-dotted token control. Reproduction script scripts/qualify_snowflake_redaction_components.py; report SNOWFLAKE_TWO_DEFECT_COMPONENT_EXECUTION.json.

This is developmental evidence for ONE naturally sourced multi-defect component, not four independent tasks. It is more suitable than the first one-PoC case for incomplete-repair eligibility, but still lacks full regression qualification, container PoC replay, a blinded task contract, and any model outputs. Do not count it as part of an admitted 16-task corpus yet. No additional GPU work launched.

### Upstream redaction regression replay

Ran all ten zero-argument upstream component test functions from the pinned fixed revision, retaining their exact bodies, fixture values and mock decorators. Replaced only the package import with the hash-checked component class to avoid loading unrelated connector dependencies. Full fixed component passes 10/10. Vulnerable and each single-pattern repair pass the same nine regression functions and fail test_mask_token, whose body combines the two security assertions. This supports functional retention for the two partial repairs; the separate synthetic probes distinguish their remaining defects. It is not a full-package pytest/container run and ten functions are not ten independent tasks.

Reproduction: scripts/replay_snowflake_upstream_regressions.py. Evidence: artifacts/validator_external_sources_20260909/SNOWFLAKE_UPSTREAM_REGRESSION_REPLAY.json. No assertion changed, no model evaluated. The class uses regular expressions/imports and is not directly compatible with the existing builtins-only replacement-function sandbox. A natural-component execution interface must be qualified before presenting it as an executable GPU queue task.

## Requests redirect component candidate

Inspected the pinned PatchEval before/after methods for CVE-2018-18074. A network-free component replay uses the released rebuild_auth and should_strip_auth methods with local request/response stubs and trust_env=False. Original behavior fails same-host scheme-downgrade and port-change probes. Controlled ablations of the fixed final predicate show a port-only repair fixes only the port probe, a scheme-only repair fixes only the downgrade probe, and the complete helper fixes both. All retain same-origin auth, strip different-host auth, preserve standard HTTP-to-HTTPS upgrade compatibility, and preserve a non-auth header.

Reproduction: scripts/qualify_requests_redirect_component.py; report REQUESTS_REDIRECT_COMPONENT_CHECK.json. This is ONE additional developmental component, not independent HTTP integration or model evidence. The partial repairs are controlled ablations, not naturally collected model errors. Netrc behavior, full-package regression tests and official container reproduction are not covered. The fix methods are read from hash-bound dataset snippets; full upstream-source attestation remains to do.

Twisted redirect handling was also inspected and shares origin/header-stripping mechanisms. Do not count a second library implementing the same redirect predicate as a new independent security category. GitPython clone handling primarily adds a missing kwargs-option check in the inspected diff; it is not yet demonstrated to offer two independently repairable defects. The external pool therefore remains far short of the four-category, sixteen-task qualification requirement.

### Requests upstream attestation

Resolved the [upstream merge](https://github.com/psf/requests/commit/c45d7c49ea75133e52ab22a8e9e13173938e36ff) and its vulnerable parent dd754d13de250a6af8a68a6a83a8b4419fd429c6; downloaded full sessions.py and fixed test_requests.py with hashes in requests_source/DOWNLOAD.json. Benchmark snippets match upstream executable AST exactly. Initial strict AST equality failed only because snippet dedenting changed docstring indentation; the recorded comparison normalizes docstring indentation, not code.

Four pure upstream should_strip_auth test functions pass against the isolated fixed method with their assertion bodies unchanged. The upstream port-change test also changes scheme, so it cannot alone establish independent port coverage; the earlier same-scheme port probe supplies that component check. Reproduction attest_requests_component.py; report REQUESTS_UPSTREAM_ATTESTATION.json. This still excludes HTTP integration and full-package execution, and adds no new independent task or neural evidence.

## Django archive reference counterexample

The supplied fixed target_filename helper for CVE-2021-3281 uses string-prefix containment. Confirmed the same code in [the pinned upstream archive.py](https://raw.githubusercontent.com/django/django/02e6592835b4559909aa3aaaf67988fef435f624/django/utils/archive.py). Pure POSIX path arithmetic accepts ../dest_sibling/file.txt under /qualification/dest and returns /qualification/dest_sibling/file.txt, which is outside the destination by path-component containment. A normal child is accepted; an unrelated sibling is rejected. No file or archive extraction occurred.

Reproduction check_django_archive_reference.py and report DJANGO_ARCHIVE_REFERENCE_REPLAY.json. This refutes using this released helper as a comprehensive containment oracle. It does not prove a defect in current Django, nor reproduce the full benchmark container's behavior. Exclude this unamended helper from the candidate reference pool. A separately corrected implementation would require independent validation and explicit provenance; do not silently call it the released ground truth.

The h11 CVE-2025-43859 diff was also inspected. Its multiple edits coordinate one chunk-footer validation state machine; edit count alone does not establish two independently repairable defects. No second defect or component qualification was claimed. The pool remains insufficient for a new GPU experiment.
