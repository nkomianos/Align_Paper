# Reduced EndoPAHF execution integrity and locked-data boundary

5 September 2026. This is implemented, tested infrastructure for the new reduced
DEV protocol. **No neural experiment, model inference, GPU/host access, or new
private-data download was performed.** Legacy failed assays and verifier receipts
remain historical evidence; this document does not retrospectively qualify them.

## What the new verifier establishes

`src/interaction_sprint/hindsight_execution_integrity.py` supplies the new
runner's input, provenance, checkpoint and sealing helpers.
`scripts/verify_hindsight_pahf_reduced_dev.py` verifies a sealed completed or
failed DEV root. Its default path is read-only except for an optional fresh
verification receipt **outside** the sealed root.

| Layer | Evidence checked | Limits |
|---|---|---|
| Input identity | Fixed manifest, learning/DEV file hashes, exact row/base/rotation/display-name grids | Does not open or certify reserved outcomes |
| Source and configuration | Actual CLI, effective config, Git HEAD/branch/index digest/status/diff digests, transitive local Python source hashes, package/environment metadata | Git/source receipts attest saved state; arbitrary untracked user data is not copied |
| Model/tokenizer identity | Immutable model and tokenizer commit, local snapshot configuration/tokenizer/weight file hashes and index membership, effective loaded settings, backend tokenizer hash | Default verifier checks recorded weight receipts; optional local snapshot rehash checks weight bytes |
| Text and tokens | Exact source prompt and captured chat template, saved rendered text, tokenizer backend replay, native answer IDs, exact padded input IDs/masks, context ceiling | No neural inference is performed |
| Evaluation arithmetic | Saved full-vocabulary float32 logits, exact model vocabulary dimension, answer logits, log normalizer, full and conditional choice probabilities/mass | Does not establish logits by rerunning a model checkpoint |
| Training state | Common initialization, per-arm start adapter, fresh empty AdamW, final adapter and optimizer moments/steps/settings, exact logged batches and work counts | Does not recompute gradients or independently prove execution from a log alone |
| Decision | Exact interface/acquisition gates, early-stop arm count, deterministic readouts, all method comparisons, immutable recursive manifest | A method pass remains developmental; DEV is not confirmation |

The returned receipt explicitly reports `neural_inference_recomputed: false`
and `gradients_recomputed: false`. It separately reports recorded weight-receipt
validation, optional weight-file rehashing, tokenizer replay and saved-logit
arithmetic. Passing a checksum or arithmetic verifier cannot be presented as
independent neural replication.

## DEV-only access

`load_learning_dev` opens exactly these three members:

- `MANIFEST.json` (metadata);
- `learning.json` (2,520 rotations of 630 learning bases);
- `development.json` (384 rotations of 96 DEV bases).

It pins manifest SHA-256
`2ae32c119087d97de6f2e5a65959b4c94a7d81362732871ff806b157e000fab2`.
Reserved manifest entries are neither opened nor hashed. No directory-wide
dataset verifier is called, and there is no `--split`, `--unlock`, or
confirmation-reader branch. Traversal and symlink/junction member escapes fail
before content access. Future confirmation requires a separate entry point,
explicit reserved path and verified prerequisites under a newly frozen protocol;
this module intentionally supplies no bypass.

The exact DEV binding digest is
`10bff89f8c5acb39b78be590324854aa5d120fe81f279ba6f65d97e165ff1b90`.
Every prediction must bind to the expected `id`, `base_id`, integer
`label_rotation`, and `source_user`. The latter means the display name before
the prompt's first colon. The constructed release does not provide a separate
user-ID column; this grouping is not a newly validated human identity measure.
Learning/DEV base, exact prompt and surface overlap are rejected. User overlap
is allowed and must be reported as observed-cohort task transfer.

The reduced protocol's exact-interface panel now uses 16 bases from the shared
64 **learning anchors**, four rotations and two target contexts. DEV hindsight
text is not model-visible during qualification. The interface grid is rebuilt
from those permitted learning rows, not accepted from its own saved summary.
Sparse training functions receive restricted immediate/anchor views; the
full-delayed oracle is an explicitly separate extra-information control.

A poison-file regression makes any `confirmation.json` open fail. Both the
loader test and a sealed setup-failure verifier test complete without touching
that content. This is a software data-access boundary, not an operating-system
sandbox for arbitrary future code.

## Sealed runtime record

The runner requires a pre-existing local Hugging Face
`models--OWNER--MODEL/snapshots/<40-character-commit>` directory and uses
`local_files_only=True`. Floating revisions and model-cache links outside that
model's cache are rejected. All local weight shards and weight-index references
are hashed. The loaded model/tokenizer configuration and tokenizer backend are
captured separately from the requested configuration.

The source receipt contains Git HEAD, branch, index-entry digest, porcelain
status, staged and unstaged binary-diff digests, the complete configured local
Python import closure, package versions and a restricted set of runtime
environment variables. It does not copy credentials, remote URLs with embedded
tokens, arbitrary environment contents, or untracked `analysis/` contents.
Source files changed after launch fail current-source verification; use the
recorded source version for a historical replay.

Each arm saves its actual start adapter and empty optimizer state before
updates. The verifier uses only
`torch.load(..., weights_only=True, map_location="cpu")`. Canonical typed tensor
hashes avoid dependence on archive filename/serialization differences. Every
start must equal the shared initialization, and final adapter schemas must
match. AdamW parameter order must match the recorded trainable parameter names.
Its moments must have the exact parameter shapes/dtypes, be finite, and have
nonnegative second moments; every parameter's step tensor must equal the
executed step count. Warm optimizer states, missing trained parameters, extra
states, altered learning rates, reordered IDs and wrong step tensors fail.

All optimizer settings are explicit. Newer PyTorch versions serialize
`decoupled_weight_decay=True` for AdamW; this derived flag is recorded and
validated when present. Other undeclared defaults fail closed. The constant
35-step schedule and exact population/anchor IDs are checked, alongside the
saved per-step forward/backward work and residual/mixture loss composition.
An expected-count check alone is insufficient.

The saved tokenizer backend is instantiated directly through
`tokenizers.Tokenizer.from_str`; padding and truncation are disabled for replay.
The verifier reconstructs source text with the captured chat template, encodes
without extra special tokens, then constructs the declared left padding and
attention mask. It compares these arrays exactly to the actual saved model
inputs. Swapping two token rows while leaving IDs, masks and text hashes intact
is rejected. Saved logits must have the exact effective text-vocabulary width,
not merely enough columns to contain A/B/C/D.

For this forward-only study, `generation_receipt.json` explicitly records that
`generate` was not called. The separate `capture_effective_generation_config`
helper validates inherited defaults plus explicit overrides for future
generation-based runners; a test reproduces the historical `do_sample=True`
inheritance bug and requires an effective greedy override. Generation settings
are not falsely claimed to control forward-logit scoring.

`seal_artifacts` creates a recursive manifest once, after a COMPLETE or FAILED
marker exists. Verification rejects omitted, extra, changed or escaping members.
A failed setup can have no model receipt; its verifier receipt identifies that
limited scope. Completed trained arms and partially written failed-arm states
remain distinguishable. Failed or incomplete runs never receive a qualified
scientific endpoint merely because their saved files verify.

## PUPPET source-target binding remains explicitly unresolved

The legacy source layout is a single mixed
`artifacts/puppet_schema_20260904_v1/hidden_puppet_master_dataset.csv` containing
both DEV and reserved query groups. Its source checksum is
`6f5ac1b28de08c302abad8f25b2451df36d74006ab9b15e6dac6691722c0841d`;
the existing schema reports 1,151 rows. There is no DEV-only raw slice or
precomputed byte-offset index in the inspected local artifacts.

The historical runner reads the entire CSV through `csv.DictReader`, and
`records_from_rows` parses transcripts and survey values before filtering to
the DEV query split. Repeating that path would parse reserved content. The
old verifier checks the 72 saved DEV targets against themselves, so it cannot
close the source-target binding defect. It correctly reproduces arithmetic
for both the invalid v1 target binding and the repaired v2 prediction record.

This work did **not** reopen or parse the mixed CSV. Instead,
`artifacts/independent_audit_20260905/data/puppet_dev_source_conversion_request_20260905.json`
provides the precise 72 allowed zero-based source-row indices, record hashes
and seven query hashes. They were recovered solely from the existing DEV
record hash formula and the already released schema row count; no source
survey value or transcript was read. The file explicitly states
`source_targets_verified: false`.

Closing this gap requires a trusted split preparer/custodian to supply a sealed
DEV-only projection at that source checksum. It must preserve original row
indices, query IDs/hashes, condition/eligibility metadata and source
pre/post/delta values, with a source-provenance attestation or a trusted
source-to-byte-range index. A future DEV-only verifier can then require exact
record/query membership, compare each saved pre/delta to the source projection,
and verify `delta = post - pre`. It must never silently create the projection
by parsing the mixed source here. A self-asserted source checksum on a newly
constructed projection is not independent proof that its targets came from
the original CSV.

The absence of such a projection is a remaining data-format limitation, not
a reason to open the reserved material or call the PUPPET causal thesis false.
No PUPPET endpoint has been rerun, and existing audit classifications remain.

## Validation and use

At this revision, **38 integrity tests pass**, including exact locked-file
access, source/rotation/user binding, unsafe checkpoint objects, reset and
optimizer-state corruption, step/work/loss mismatches, manifest tampering,
inherited sampling configuration, immutable local model receipts, exact
source-to-token reconstruction, swapped input arrays, and vocabulary mismatch.
The verifier also compiles and its help path performs no model loading.

The full CLI integration test constructs a sealed toy artifact in a pytest
temporary directory: 128 interface rows, seven 384-row DEV evaluations, six
35-step tiny CPU AdamW checkpoint sequences, all source/input/tokenizer/logit
receipts and the complete method decision. The production CLI accepts it as
a structurally consistent **fixture**, with `fixture_only: true`; no Qwen
execution is implied. A second CLI call after swapping input-token rows and
updating the manifest still rejects the artifact at semantic token binding.
Thus a self-consistent checksum alone cannot hide that corruption. These
fixtures are software tests, not experiments or submission evidence.

After an authorized future run, ordinary offline verification is:

```powershell
$env:PYTHONPATH='src;.;scripts'
python scripts/verify_hindsight_pahf_reduced_dev.py `
  --input-root artifacts/hindsight_endo_pahf_external_20260904_v3 `
  --root <sealed-reduced-dev-root> `
  --receipt <fresh-receipt-outside-root.json>
```

Add `--model-snapshot <already-local-pinned-snapshot>` only to rehash original
model/tokenizer/weight files. It does not download them, run the model, or
unlock confirmation. A fresh neural recomputation is a different, currently
unrun validation step requiring its own declared scope and authorized compute.
