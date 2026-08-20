# DID-v1 DEV-only deployment runbook

This runbook covers the post-hoc DID-v1 inference diagnostic only. It does not
alter the failed Stage-1 decision, authorize locked TEST access, or reveal the
hidden answer key. The bundle must not be created for a scientific run until the
diagnostic spec, public cases, source, and run command have been reviewed,
committed, and frozen.

## Deployment contract

Every archive has the single root `under_extinction_dev_diag/` and an exact
`DEV_DIAG_BUNDLE_MANIFEST.json`. The manifest binds every payload path, role,
byte count, mode, and SHA-256, plus the complete ordered inventory hash. It also
binds a named branch, exact tagged HEAD commit/tree, and a Git blob identity for
every `project/*` payload. Bundle creation requires a clean tagged worktree;
every selected project file must be Git-tracked and byte-identical to HEAD.
Ignored or untracked Python cannot enter through `src/**/*.py`. Untracked
retrieval artifacts under the explicitly recorded artifact prefixes may remain
on disk, but are never selected.

The only eligible payloads are:

- all runtime `src/**/*.py` files, `pyproject.toml`, `README.md`, the pinned
  `requirements/h100-cu12x.lock`, `scripts/run_dev_diag_remote.sh`, and
  `scripts/bootstrap_dev_diag.sh`;
- `configs/stage1_dev_diag_v1.yaml` and the original
  `configs/bridge_pilot.yaml`;
- model-visible `MANIFEST.json`, `cases.jsonl`, and
  `ANSWER_KEY_COMMITMENT.json`;
- the historical data `MANIFEST.json` and its hash-bound `dev.jsonl` only;
- checkpoint zero, genuine update 300, and proxy update 300, each reduced to
  `checkpoint_manifest.json`, `adapter_config.json`,
  `adapter_model.safetensors`, and `reload_probe.json` when present.

The bundle code rejects archive-path collisions, traversal, links, devices,
caches, secret-like material, hidden answer keys, `test.jsonl`, `train.jsonl`,
`bridge_state.pt`, optimizer/scheduler state, and every weight except the three
explicit adapter files. It validates the public-case and DEV manifest hashes,
checks that cases remain answer-blinded AUDIT/DEV records, checks checkpoint
manifests against their selected adapter files, verifies the prospective
inventory before writing, and re-reads every archived byte before publishing the
archive and checksum. The public manifest and commitment have exact schemas;
unknown fields and nested answer-bearing fields such as `expected_by_policy`
are rejected.

## Deployment and verified-analysis interfaces

Bundle construction remains an explicit Python operation so that it cannot be
confused with inference. Its exact local interface is:

```python
from pathlib import Path

from under_extinction.dev_diag_deployment import (
    DevDiagnosticBundleInputs,
    collect_dev_diag_results,
    create_dev_diag_bundle,
    verify_dev_diag_bootstrap_attestation,
    verify_dev_diag_bundle,
    verify_dev_diag_results,
)

inputs = DevDiagnosticBundleInputs(
    project_root=Path("."),
    diagnostic_spec=Path("configs/stage1_dev_diag_v1.yaml"),
    case_manifest=Path("<frozen-model-visible>/MANIFEST.json"),
    cases=Path("<frozen-model-visible>/cases.jsonl"),
    answer_key_commitment=Path(
        "<frozen-model-visible>/ANSWER_KEY_COMMITMENT.json"
    ),
    bridge_config=Path("configs/bridge_pilot.yaml"),
    historical_data_manifest=Path("<historical-data>/MANIFEST.json"),
    dev_data=Path("<historical-data>/dev.jsonl"),
    checkpoint_zero=Path("<checkpoint-000000>"),
    genuine_final=Path("<genuine-checkpoint-000300>"),
    proxy_final=Path("<proxy-checkpoint-000300>"),
)

# Do this only after the freeze review; do not run it from the current draft.
archive = create_dev_diag_bundle(inputs, Path("deployment/did_v1.tar.gz"))
verify_dev_diag_bundle(archive)
```

`create_dev_diag_bundle` refuses to overwrite either the archive or its
`.sha256` sidecar. `verify_dev_diag_bundle` accepts the tarball, its extracted
root, or the parent containing that root.

The paid-runtime verifier has the exact public interface:

```python
binding = verify_dev_diag_bootstrap_attestation(
    Path("<session>/evidence/logs/bootstrap_runtime_attestation.json"),
    Path("<extracted>/under_extinction_dev_diag"),
)
```

It returns the deterministic `did_v1_verified_bootstrap_binding` object. The
evaluator must embed that object verbatim in its run manifest and per-policy
provenance. Completed/formal analysis must call the verifier again with the
same attestation and deployment root and require exact equality. The binding
contains the attestation SHA, raw bundle-manifest SHA, bundle inventory SHA,
source-identity SHA, all-project-payload inventory SHA, tagged Git HEAD, GPU
UUID query, and exact pass booleans for hardware, dependency closures, ABI,
kernels, and source identity.

Result retrieval has a separate interface and deliberately accepts incomplete
runs:

```python
archive = collect_dev_diag_results(
    Path("<session>/evidence"),
    Path("retrieval/dev_diag_results_<run-id>.tar.gz"),
)
verify_dev_diag_results(archive)
```

The collector permits only JSON, JSONL, logs, text, and checksum files. Its
manifest records whether the evaluator claimed `COMPLETE`, but the collector
does not validate that claim. A complete retrieved inference directory must
still pass `verify_completed_dev_diagnostic_run` before analysis.

## Transfer and execute after freeze

Transfer both `did_v1.tar.gz` and `did_v1.tar.gz.sha256`. Treat the checksum as
an independently trusted value. On the remote host:

```bash
sha256sum --check did_v1.tar.gz.sha256
python3 - did_v1.tar.gz <<'PY'
import pathlib
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:gz") as archive:
    members = archive.getmembers()
    names = []
    for member in members:
        path = pathlib.PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not member.isreg()
            or not path.parts
            or path.parts[0] != "under_extinction_dev_diag"
        ):
            raise SystemExit(f"unsafe archive member: {member.name!r}")
        names.append(member.name)
    if len(names) != len(set(names)):
        raise SystemExit("duplicate archive paths")
PY
tar -xzf did_v1.tar.gz
bash under_extinction_dev_diag/project/scripts/run_dev_diag_remote.sh
```

The runner verifies the extracted inventory before creating a virtual
environment or loading a model. The diagnostic bootstrap reproduces the proven
paid-runtime checks: exact `aarch64` architecture; exactly one
`NVIDIA GH200 480GB`; compute capability 9.x and at least 90 GiB; provider
PyTorch `>=2.5,<3`; two complete reachable dependency closures; venv-origin
isolation for every non-Torch experiment dependency; an explicit, attested
allowlist for provider Torch's support runtime; exact Torch–NumPy ABI roundtrip; absence of
optional DeltaNet kernels; and live CUDA, BF16 matmul, and PyTorch SDPA probes.
It force-installs the exact binary-only lock roots into an isolated environment
even if matching global packages are visible, overlays any externally resolved
non-Torch transitive dependency into that environment, and uses only provider
PyTorch. The complete hardware identity, `nvidia-smi` identity,
package closure and origins, source hashes, and kernel-probe results are written
to `logs/bootstrap_runtime_attestation.json`, which is included in retrieval.

Only after that attestation passes and the strict bundle/runtime binding is
written to `logs/bootstrap_binding_preflight.json` does the runner make the
verified bundle read-only and invoke
`bridge-dev-diag-evaluate` with the immutable paths inside the bundle. The base
model remains pinned by the config/spec and may be downloaded into the separate
runtime cache. A Hugging Face token is normally unnecessary for the public
model; if one is needed, pass it only through the remote environment and never
place it in the archive or a command-line argument.

Optional safe location overrides are `DID_RUN_ID`, `DID_WORK_BASE`, and
`DID_RETRIEVAL_ROOT`. `DID_RUN_ID` is validated and every run requires a new,
nonexistent destination.

After inference (including evaluator failure), the runner re-verifies the full
bundle and attestation and requires its deterministic result to equal preflight
exactly. It writes `logs/bootstrap_binding_postflight.json`; any payload drift
changes the run status to failure. The runner exports
`UE_DEV_DIAG_BUNDLE_ROOT` and `UE_DEV_DIAG_BOOTSTRAP_ATTESTATION` so evaluator
and analysis code can bind the same evidence rather than trusting an operational
log.

## Failure and retrieval behavior

Normal evaluator success or failure is followed by result collection. The
retrieval archive and checksum are written outside the immutable bundle and
contain the run manifest, completed policy outputs, partial policy outputs, and
logs—but no checkpoint weights or runtime/model caches. If the process or host
is killed before the collector runs, invoke `collect_dev_diag_results` manually
on that session's `evidence/` directory.

There is currently **no evaluator resume implementation**. Preserved partial
outputs are forensic evidence and may avoid ambiguity about what completed, but
they cannot be supplied to a new evaluator invocation. A retry needs a new run
ID and repeats inference from the beginning. Do not delete the three source
checkpoints until the retrieved complete run has been checksum-verified locally;
if the run is partial, retain them because a full retry may be required.

After copying the result archive and sidecar off the GPU host, verify both the
transport checksum and `verify_dev_diag_results`. Only then is it safe to
terminate the instance. A claimed complete run additionally requires
`verify_completed_dev_diagnostic_run` before any analysis or resource decision.

Formal analysis must use the CLI finalizer below. Calling the low-level analysis
function directly produces an explicitly `UNVERIFIED_DIRECT_API_*` decision and
can never license E1b. The finalizer re-verifies the completed run, public corpus,
answer-key reveal, deployment bundle, source identity, and bootstrap attestation;
analyzes the exact prediction bytes it verified; repeats those verifications to
close file-swap and source-drift races; and only then writes the report atomically.

```bash
under-extinction --config configs/bridge_pilot.yaml \
  bridge-dev-diag-analyze \
  --spec configs/stage1_dev_diag_v1.yaml \
  --case-manifest <public>/MANIFEST.json \
  --cases <public>/cases.jsonl \
  --answer-key-commitment <public>/ANSWER_KEY_COMMITMENT.json \
  --answer-key <private>/answer_key.jsonl \
  --run-dir <retrieved-complete-run> \
  --deployment-root <verified-extracted-bundle>/under_extinction_dev_diag \
  --bootstrap-attestation <retrieved-complete-run>/logs/bootstrap_runtime_attestation.json \
  --destination <new-report-path>.json
```

All listed arguments are mandatory. A formally verified all-pass report can
license only a separately preregistered E1b/DEV2 experiment; it still cannot
reverse Stage 1, open locked TEST, authorize replication, or establish paper
viability by itself.
