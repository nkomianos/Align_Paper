# Implemented pilot suite: execution contract

This supersedes the protocol-only readiness table in RESEARCH_QUEUE_READINESS_20260906.md.
All four candidate runners now exist. GPU execution is untested. The monitor is
data-blocked: the MALT raw-file endpoint returned HTTP 401, and no reviewed
normalized dataset has been fabricated to bypass that prerequisite.

## Scope changes frozen before first GPU execution

The first screens deliberately use the already pinned Qwen3.5-9B instead of
downloading a new 3B-4B model. Compensation uses a fixed additive rank-one edit
of one middle MLP output matrix, derived from authorized/unauthorized learning
contrasts. A separate rank-eight adapter at that site is trained under the edit;
the no-edit and norm-matched-random-edit arms reset to identical adapter tensors
and use fresh optimizers. The edit is reconstructed from a saved original weight
on every change, avoiding cumulative subtraction rounding errors.

This is a synthetic policy-choice apparatus screen, not the proposed completed
biological-compensation paper. Train/dev/exploratory-holdout each contain 64
option-balanced rows, representing 32 paired base scenarios. Arithmetic utility
has 32 rows. Six edit doses are selected on training data only. Three 16-step
training arms use four examples per optimizer step, full-vocabulary answer-token
CE, one clip per accumulated batch and no padding loss. No dataset/test endpoint
is selected after seeing GPU outputs. Independent seeds, effect-matched edits,
multi-site causal repair and external tasks are later work, not claimed here.

Reference robustness uses a prompted withholding organism, not a secretly
misaligned or fine-tuned model. It explicitly requires a privileged paired
honest reference, compares means and coordinate medians under 0/25/50% injected
reference replacement at budgets 4/16/64, and includes no-intervention/random
controls. It can reject this simple robustness setup but cannot establish
reference acquisition or improvement over SFT/CPE. Answer-token grafting is not
a demonstration of coherent free-generation recovery. No hazardous tasks are
used in either pilot.

CLARA exactly enumerates 6-bit scenes under independent/shared-error mixtures,
compares tie-complete joint credible sets to an oracle query-specific posterior,
and checks logical rewrite invariance. It is a Bayesian apparatus baseline,
not conformal calibration or a new method. Its current output must never be
presented as a neural or real-world safety result. No GPU expansion is automatic.

Monitoring compares task-only, trace-only, full-context, and specification-
grounded prompts. Both label rotations are averaged. Thresholds come only from
reviewed calibration negatives, with strict greater-than handling of ties. No
holdout is scored automatically. This is a prompt comparison, not yet the
proposed evidence-span method. Never label unreviewed MALT rows as clean negatives.

## Running

From repository root with a compatible environment:

```bash
export PYTHONPATH=src:.:scripts
python scripts/run_research_pilot.py clara --out /path/to/new/clara_cpu
python scripts/verify_research_pilot.py --root /path/to/new/clara_cpu
```

On the single-GPU Linux host, with an already cached pinned model:

```bash
python scripts/launch_research_suite.py --snapshot "$SNAPSHOT" --learning "$LEARNING" --out "$SUITE" --allocation-start-utc "$ALLOCATION_START_UTC" --previous-h200-hours "$PRIOR_H200_HOURS"
```

`--dry-run` prints the plan without running it. Without `--learning`, the old
Hindsight calibration is omitted. Without `--monitor-data`, monitoring is marked
blocked and omitted. Supplied monitor data must pass review/split checks before
any model loads. Input token lengths are checked without loading GPU weights;
overlength traces are rejected, never silently truncated.

Order: optional Hindsight (1h cap), CLARA CPU (3 minutes cap), compensation (3h),
reference (2h), optional monitor (2h). Maximum scheduled caps sum to 6.05 hours
including Hindsight without monitoring, or 8.05 with both optional stages.
Without Hindsight these sums are 5.05 and 7.05 hours. Actual runs should generally be shorter,
but throughput/fit remain unbenchmarked. All use the same original allocation
start, so time spent loading, verifying and between jobs reduces remaining budget.
No stage may expand into replication or paper confirmation automatically. An
execution or verifier failure stops the suite; a verified scientific stop may
continue to the next independent screen. Maximum budget remains 50 allocated
H200 hours. GH200 timing must be measured separately.

The supervisor terminates processes, not cloud billing. Retrieve evidence and
stop the rental through the provider when the suite ends; SSH disconnection is
not termination. Do not leave an instance waiting for implementation work.

## Monitor input contract

`--monitor-data` accepts a JSON array with fields: id, family, model_lineage,
split (calibration/dev/holdout), task, trace, label (integer 0/1), reviewed
(true), prompted (boolean), source_sha256 (64 lowercase hex digits). Require
at least 20 negatives and 10 positives per split, no duplicated transcripts,
and no task family crossing splits. This minimum is an apparatus requirement,
not sufficient statistical power for a 5% population false-positive claim.
Agent-lineage overlap is not silently advertised as lineage transfer. Real
source mappings/labels must be audited separately; user-provided `reviewed`
metadata is not independent authentication of a human review.

## Verification limits

Manifests cover all artifacts. Verifiers bind scores to input rows, recompute
choice probabilities/accuracy and routes, and reject missing/duplicated score
populations. They do not rerun neural checkpoints or replay tokenizer outputs.
Recorded forward inputs, logits summaries, original site weights, edit tensors,
training schedules, adapters and source/model provenance support later replay.
The pilots are single-seed, exploratory and synthetic except the optional real
monitor input. None may emit a paper green light.

## Local qualification and transfer

23 CPU regression cases pass, including the existing calibration cases and new
queue execution-failure/verifier-timeout stops. Archived tokenizer qualification
covers all 448 plain/withholding renderings of the 224 frozen synthetic rows;
maximum 77 tokens and distinct single-token A/B answers. Exact CLARA enumeration
and independent saved-output verification pass. Joint-set coverage is 61.42%
with 0.722% accepted posterior error; the query-specific oracle has 86.12%
coverage with 1.830% error. These are different operating points under a specified
finite Bayesian model, not a matched-risk advantage or empirical safety claim.

`package_research_pilots.py` creates a new commit-bound source bundle/ZIP,
frozen inputs, CPU receipts and hash manifest. It reuses hash-verified learning
data and wheel overlays from the original calibration package without changing
that package. Only transfer the wheel overlay matching the host architecture and
Python version. A working CUDA PyTorch image and cached pinned weights are still
required; overlays are not complete CUDA environments.

Check every transferred component against PACKAGE.json, then extract source.zip
into a fresh directory (or clone source.bundle), select the compatible environment,
and follow Running above. The default generated pilot inputs are checked against
the frozen packaged input hashes during packaging. No network downloads occur in
the experiment runners. No manuscript or claimed GPU result is generated by this
preparation. Monitor data must arrive and be audited before adding its queue flag.
