# Provenance-authority feasibility gate

This is a separate pre-paper feasibility study. It is not a continuation of
the failed reward-channel Stage 1 experiment.

The operational question is whether a fixed, goal-conflicting instrumental
plan receives more decision authority when it occurs as a prior assistant turn
than when the identical plan body occurs in an external or source-neutral
record. This manipulates conversational provenance and role; it does **not** by
itself demonstrate human-like self-recognition.

Each matched unit has four conditions: prior assistant turn, external record,
source-neutral record, and no injection. The same plan text, principal goal,
route facts, action-label assignment, and trajectory ledger are reused within a
unit. The source plan is either earlier or later than a principal reminder, and
the ledger has a short or long horizon. The outcome is the forced-choice
probability of selecting the plan-favoring route, where the alternative is the
programmatically principal-consistent route.

The initial corpus has 512 matched units and 2,048 prompts. The primary
estimand is the paired mean `P(plan route | self assistant turn) -
P(plan route | external record)`. It only licenses a dynamic agent study if the
effect is at least 0.10, its paired 95% bootstrap lower bound is positive, it
persists when the self record is less recent than the principal reminder, and it
does not decline with horizon. A failure stops this line; it is not tuned away.

The initial Qwen3.5-9B run is teacher-forced A/B scoring. It is intentionally
cheaper than a multi-turn agent run and tests whether there is any provenance
signal worth spending on. Any positive result must be replicated on an
independent model and in a genuinely dynamic tool environment before a paper
claim about long-horizon drift is made.

Before loading the model, the evaluator renders every complete prompt with the
pinned tokenizer, verifies distinct equal single-token A/B actions, and rejects
the run if any prompt-plus-action would exceed the frozen context cap. The
ordered token-length hash is retained in the prediction summary.

## Remote execution

After the DID run is retrieved or has otherwise fully stopped, transfer a
tagged checkout/archive of this feasibility study to the GH200. Reusing the
verified DID virtual environment and its public model cache is allowed because
this study is inference-only and separately records its source, configuration,
and corpus hashes. Set `PROVENANCE_RUNTIME_VENV` and `PROVENANCE_HF_HOME`, then
run `bash scripts/run_provenance_authority_remote.sh`. The script produces one
prediction JSONL and one continuation report; it never reads DID predictions or
the private DID answer key.
