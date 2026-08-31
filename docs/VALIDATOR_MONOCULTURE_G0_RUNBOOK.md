# Validator-Monoculture G0 Runbook

## What this run decides

This is a feasibility gate, not the paper's final experiment. It asks whether
plausible incomplete security patches are less likely to be exposed by tests
from the same model family than by tests from the other family. Qwen 3.5 9B and
Gemma 4 12B are fully crossed as patch generators and test generators. Entire
CWE families are held out from DEV selection.

The primary test writer sees only the function signature, natural-language
contract, and public regression vectors. It sees neither vulnerable nor patched
code. This makes the primary interaction a test of correlated task/security
interpretation rather than code-style recognition. A canonicalized patch-aware
arm is secondary.

There is no acceptance guarantee. A pass authorizes a larger three-family,
natural-repository study; a failure kills this formulation.

## Workload and expected time

The frozen run contains:

- 192 patch completions: 32 tasks × 2 families × 3 samples;
- 128 specification-only test-suite completions: 32 tasks × 2 verifier
  families × 2 suites;
- `4 × N` patch-aware suite completions, where `N` is the number of patches
  that pass every public regression, preserve every hidden behavior the
  vulnerable program already gets right, repair at least one frozen
  vulnerable-baseline defect, and retain at least one other such defect.

Thus the total is `320 + 4N`, with a hard maximum of 1,088 short generations.
After both models are cached, the planning estimate is 2–4 hours on one GH200;
the exact-runtime preflight gives the first trustworthy throughput measurement.
Four GPUs are unnecessary for G0 because the design has only two models and the
phase-separated evidence path is intentionally simple. Do not reserve a large
fixed dollar budget: run the bounded gate, retrieve it, and stop.

## Before enabling paid compute

1. Commit and push the final tree. Record the exact 40-character commit SHA.
2. Use a fresh or clean detached checkout at that SHA. The run script rejects
   tracked, staged, or untracked repository changes.
3. Create the existing virtual environment expected at `.venv`, using the
   platform's validated CUDA/PyTorch image, install
   `requirements/h100-cu12x.lock`, then install this checkout editable with
   `--no-deps`.
4. Export a fresh Hugging Face token through `HF_TOKEN`; never place it in the
   repository or command history. The token needs access to the pinned Gemma 4
   checkpoint. Any token pasted into chat should be rotated before the formal
   run.
5. Prefer pre-populating `.hf_cache` on a persistent volume. If both exact
   revisions are already cached, keep the default local-only mode. Otherwise,
   explicitly set `VALIDATOR_MONOCULTURE_LOCAL_FILES_ONLY=0`; the preflight will
   download and fully instantiate both models before any experimental output.

## Launch

From the pinned checkout on the GPU host:

```bash
export VALIDATOR_MONOCULTURE_PINNED_COMMIT='<40-hex-commit>'
export VALIDATOR_MONOCULTURE_RUN_ROOT='/home/ubuntu/validator_monoculture_g0_<UTC>'
export VALIDATOR_MONOCULTURE_VENV='/home/ubuntu/Align_Paper/.venv'
export VALIDATOR_MONOCULTURE_HF_HOME='/home/ubuntu/Align_Paper/.hf_cache'
export VALIDATOR_MONOCULTURE_LOCAL_FILES_ONLY=1
bash scripts/run_validator_monoculture_g0_remote.sh
```

The script acquires an exclusive run-root lease and creates an immutable,
run-unique `RUN_BINDING.json` (including a fresh 256-bit nonce) before any
stage. It then performs a dynamic 32-task oracle preflight, verifies Linux
Python 3.12, PyTorch 2.7.1/CUDA 12.8, exact package
versions and CUDA capacity, downloads/checks both pinned snapshots if explicitly
allowed, instantiates and samples both exact runtimes, then runs the seven frozen
evidence phases. Each generated record is durably checkpointed; an interrupted
phase is resumed only after validating its exact deterministic prefix, runtime
provenance, code/input binding, and preserved crash tail. Checkpoint files are
hints: every skipped phase is checksum- and closed-layout-validated. Never
delete a partial root or launch two resumes concurrently.

After an interruption, use the same checkout, commit, cache, and run root, set
`VALIDATOR_MONOCULTURE_RESUME=1`, and rerun the same shell script. Completed
stages are checksum-validated and skipped; the first partial phase is replayed
before any missing model call is issued. Resume never permits selecting or
dropping individual completions.

The only successful terminal state is
`generation_complete__offline_analysis_pending` in
`COMPLETION_MANIFEST.json`. It is not a scientific result.

## Retrieval

On the Windows laptop, retrieve to a fresh directory without altering the
remote root:

```powershell
./scripts/retrieve_validator_monoculture_g0.ps1 `
  -HostName '<ip-address>' `
  -UserName 'ubuntu' `
  -KeyPath 'C:\path\to\ece4150' `
  -RemoteRoot '/home/ubuntu/validator_monoculture_g0_<UTC>' `
  -DestinationParent 'C:\Users\nkomi\Documents\GitHub\Align_Paper\retrieved'
```

The retriever obtains the remote completion-manifest hash before copying,
verifies every listed file, rejects links and unlisted files, and writes a
retrieval receipt outside the immutable run root.

## Formal offline verification

Formal reconstruction is intentionally rejected on native Windows because the
sandbox resource contract differs. Create the pinned Linux/WSL2 CPU environment
once from the committed lock:

```powershell
wsl.exe --exec bash -lc 'cd /mnt/c/Users/nkomi/Documents/GitHub/Align_Paper &&
  python3 -m venv .venv-validator-offline-wsl &&
  .venv-validator-offline-wsl/bin/pip install -r requirements/validator-monoculture-offline.lock'
```

Then read `evidence_root_sha256`, `git_commit`, and `code_tree_sha256` from the
verified completion manifest and run the verifier from the exact clean pinned
checkout. The report path must be outside the entire retrieved run directory:

```bash
cd /mnt/c/Users/nkomi/Documents/GitHub/Align_Paper
export PYTHONHASHSEED=0
export PYTHONSAFEPATH=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"
run='<absolute-WSL-path-to-retrieved-run-root>'
report='<fresh-path-outside-retrieved-run>/validator_monoculture_g0_report.json'
.venv-validator-offline-wsl/bin/python -m validator_monoculture.verify \
  --evidence-root "$run/evidence" \
  --public-corpus "$run/corpus/public/tasks.jsonl" \
  --private-oracles "$run/corpus/private/oracles.jsonl" \
  --config "$run/corpus/FROZEN_CONFIG.yaml" \
  --expected-public-sha256 '<pinned-public-sha256>' \
  --expected-private-sha256 '<pinned-private-sha256>' \
  --expected-config-sha256 '<pinned-config-sha256>' \
  --expected-evidence-sha256 '<value-from-completion-manifest>' \
  --expected-code-sha256 '<value-from-completion-manifest>' \
  --expected-git-commit '<value-from-completion-manifest>' \
  --expected-run-binding-sha256 '<value-from-completion-manifest>' \
  --output-report "$report"
```

The verifier does not trust runner-side parses or labels. It replays prompt and
seed commitments, reconstructs patch eligibility, re-executes every generated
test reference-first, enforces exact crossed arms and common task support, and
checks that the evidence tree was not mutated during analysis.

## Decision and termination

- `EXPAND_VALIDATOR_MONOCULTURE`: run a separately frozen, larger study with a
  third family, more CWE clusters, and natural repository vulnerabilities.
- `KILL_VALIDATOR_MONOCULTURE`: same-family penalty is below five percentage
  points with adequate common-support capacity, balanced per-CWE valid-test
  reach, a clean execution record, at least 20% CWE-macro planted-mutant
  detection for both verifiers, at least one detection and at least 20%
  detection in every verifier-by-held-out-CWE cell, and a CWE-cluster heuristic
  upper bound below the ten-point expansion threshold; stop this paper.
- Either `INCONCLUSIVE_*`: do not claim success. Only the explicitly permitted
  patch-sample increase may be considered for insufficient patch capacity.

It is safe to terminate the GPU only after the process has exited, the complete
root has been retrieved, the remote/local completion-manifest hash matches, all
artifact hashes pass, and the offline verifier report has been written. Preserve
both remote and local evidence until then.
