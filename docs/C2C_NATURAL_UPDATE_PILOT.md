# Natural sender-update pilot: Stage A protocol

Status: implemented locally, not launched and not a scientific result. The user
has termination clearance. Confirm continued instance availability before any
new launch. No background queue is currently executing this protocol.

## Question and limits

Can an ordinary, useful sender update hurt a frozen latent interface more than
text transfer? Stage A only builds and independently qualifies the sender. It
must not measure bridge damage to select a seed, checkpoint or learning rate.
A generic observation that adapters break after input changes would not itself
justify an ICLR paper. Run the matched text comparator first: its 256 generations
measure the existing text alternative and both stages' costs.

## Frozen data and update

Prepared root: `artifacts/c2c_update_pilot_prepared_v1`. Manifest SHA-256:
`ad945cc2588809b6e94e43fdbd46f81159d8acc66dd3339621a0069c4ecd7af6`.
The preparation verifier reconstructs selections and labels from saved Parquet
rows. All partitions exclude the earlier 128 questions by ID and normalized
question/choice text. This is not semantic deduplication; public validation may
have appeared in model or fuser training.

| Partition | Count | Use |
| --- | ---: | --- |
| Update training | 1,024 | Supervised answers; labels available to trainer |
| Qualification | 128 | Sender-only generation/likelihood; key stays local |
| Repair calibration | 128 | Reserved; Stage A never opens it |
| Final evaluation | 256 | Reserved; Stage A never opens it |

Two runs use seeds **202609041 and 202609042**, each starting from the pinned
Qwen3-4B, never from the other run. One epoch; microbatch 2, accumulation 4,
128 optimizer steps. AdamW LR 5e-5, no schedule/warmup, betas (0.9,0.999), epsilon
1e-8, zero weight decay, gradient clipping 1.0. Attention q/k/v/o projections:
rank-16 LoRA, alpha 32, zero-initialized B, Kaiming-uniform A, no dropout. Frozen
BF16 base, float32 adapters. This is explicit standard LoRA, not a new method.

Use the pinned upstream C2C prompt and non-thinking chat template. Supervision
is `The correct answer is X` plus EOS. Prompt/padding have no loss. Loss is mean
completion-token NLL per example, then equal-example mean over an effective
batch. Exceeding 2,048 total supervised tokens fails; never truncate examples.

## Controls and qualification

1. Evaluate the unchanged sender on 128 qualification questions. Record greedy
   generation (64-token cap) and teacher-forced sequence log probabilities for
   all four answer strings, including EOS. The runner reads no answer key.
2. Install zero-effect adapters, save and reload them. Require identical tokens
   and option likelihoods on the first eight prespecified qualification cases.
3. Train exactly one epoch. Preserve initial, step-64 and final adapters, final
   optimizer/RNG state, training order and telemetry. Step 64 is archival, never
   selected for bridge damage.
4. Record eight wrapped-final outputs, merge into ordinary BF16 weights, preserve
   full model/tokenizer, then evaluate all 128 qualification cases. Record merge
   rounding differences; the merged model is the downstream deployment target.
5. Retrieve/hash-check all evidence. Local qualification uses the same explicit
   label scorer for old/new answers. Require >=95% parse in both, accuracy loss
   <=3.125pp (four cases), and >=5% relative reduction in normalized four-choice
   cross-entropy. **Both seeds must qualify**, with no seed substitution.

These are developmental screening rules, not confidence bounds or proof of
general capability preservation. Choice-probability improvement could reflect
format adaptation or domain-specific confidence. Report accuracy, loss and
truncation separately. Failure means the recipe did not create a suitable test
case, not that latent interfaces are robust.

Per seed: 264 greedy generations (128 base, 8 no-op, 8 wrapped, 128 merged),
264 batched four-choice likelihood forwards and 512 training microbatches.
Runtime is unmeasured; time the first 20 optimizer steps before giving a reliable
ETA. Require >=30GiB free per run for preserved checkpoint/export evidence.

## Launch instructions and boundary

Use a fresh committed checkout, not the original scientific checkout. Linux
`PYTHONPATH=src:.`; isolated C2C environment torch 2.7.1+cu128, Transformers 4.52.4.
No PEFT installation required. Source must match its recorded Git commit.

```sh
python -m scripts.run_c2c_sender_update \
  --train /path/to/public/update_train.json \
  --qualification /path/to/public/qualification.json \
  --upstream /path/to/c2c_upstream_113c3a9 \
  --seed 202609041 --output /fresh/seed_202609041
```

Use another fresh root for the other fixed seed. Never overwrite/resume a partial
run in place. Verify with the actual launch commit, not a placeholder:

```sh
python -m scripts.verify_c2c_sender_update --root /retrieved/seed_202609041 \
  --key /local/private_answer_keys.json --expected-commit COMMITTED_REVISION \
  --output /fresh/qualification_report.json
```

The verifier checks recursive manifest coverage, committed source, inputs,
generation grids, paired prompts, no-op agreement, epoch coverage, checkpoint
presence and qualification statistics. CPU tests do not replace hardware smoke
testing. No GPU experiment has run with this trainer.

## Stage B is not auto-launched

The [paired interface runner](C2C_PAIRED_UPDATE_PROTOCOL.md) is now implemented
and CPU-tested; the repair runner remains unimplemented. Finish and freeze the
remaining repair comparisons before Stage A launch so analysis choices are not
made after seeing the updates. Compare old/new sender alone, C2C, text transfer and disabled
fuser on the reserved 256 cases. Primary descriptive contrast: (new-old C2C
accuracy) minus (new-old T2T accuracy), with paired intervals and separate seeds.
This alone does not identify geometric drift. Include standalone behavior and
costs: the stronger sender already beat the fuser in the initial baseline.

Ridge, orthogonal, head-local/attention-weighted alignment, fuser retuning and
text fallback are required competitors, not methods to relabel as ours. Any
paper claim still needs independent tasks/families and substantive novelty.
