# Single-arm acquisition calibration — prospectively fixed

The previous 24-update positive control did not learn the preference task well.
Do not run another feedback sweep until direct supervision can demonstrate
acquisition. This follow-up starts from the original base model, not a selected
checkpoint. It preserves the prior semantic labels and uses fresh phrasings.
Previous cases and evaluations are developmental knowledge, not hidden test data.

Fixed changes: rank8/alpha16 LoRA, full-vocabulary first-answer-token cross entropy
(instead of loss conditioned on A/B), 96 AdamW steps, lr .0003, batch8, clip1,
no weight decay. Four training phrasings per 16 option-order contexts =64 rows;
two distinct evaluation phrasings =32 rows. All original eight domains retained.
Twelve complete shuffled passes; no adaptive extension or hyperparameter sweep.

Checkpoint adapter and optimizer at32/64/96, but evaluate held-out phrasings only
before training and at the final96-step point. No best-checkpoint selection.
This jointly changes schedule, rank, training wording diversity and objective;
it cannot identify which change is responsible for improvement.

Qualification: final full-vocabulary argmax accuracy at least90% on32 evaluation
rows, each domain at least75%, every evaluation row's A/B mass at least.95.
These are apparatus criteria, not statistical proof or a paper kill threshold.
Report all results if any criterion fails. No further feedback run launches
automatically from success. Evaluation covers new wording of acquired domain
preferences, not unseen people or domains, human welfare, or user-state dynamics.

CPU batching is checked against single-example full logits before training;
max difference must be <=.001. Right padding is masked and logits gathered at
the last actual prompt token. Native template, no sampling or truncation. All
initial/final and failure states preserved; no GPU involved.

## Independent pre-result review

A separate reviewer found no concrete defect in padding, target-token mapping,
schedule, split, loss or qualification. The numerical criteria amount to at
least29/32 correct and at least3/4 in each domain, with the stated per-row mass
requirement. Qualification concerns first-answer-token prediction, not generation
of complete responses or stopping after A/B. The two-example batching check is
not exhaustive equivalence testing across all batches and trained checkpoints.
Those limits must remain visible even if the final accuracy is high.

## Completed: acquisition qualified

Frozen commit `44cbf11`. The final96-step checkpoint scores **64/64** on training
phrasings and **32/32** on the two held-out phrasings. Every evaluation domain
scores4/4, and minimum evaluation A/B mass is **0.9999961**. Thus all prospectively
specified acquisition conditions pass. Final evaluation mean probability on the
true answer token is .99999977; full-vocabulary NLL is 2.285e-7.

The unadapted base scores20/64 on training and8/32 on evaluation using the same
full-vocabulary argmax metric. Some baseline prompts assign almost no probability
to A/B (minimum evaluation mass .0009185), unlike the original narrowly worded
probe. The gain therefore includes output-interface learning as well as learning
the eight domain preferences; do not attribute it entirely to preference memory.

Runtime353.3seconds on CPU. Counts:291 forward batches covering964 examples,
96 backwards and96 updates. All three saved adapters and optimizer states are
preserved. Read-only verification checks18 manifest files, exact data/schedule,
probability/NLL arithmetic, final criteria and finite compatible saved states.
The batch/single precheck maximum full-logit difference was1.979e-5. The verifier
does not replay backward passes or optimizer steps. Intermediate checkpoints
were not scored on evaluation or used for selecting the final model.

### What this licenses—and what it does not

This is a useful positive control: the small model and adapter can acquire this
synthetic personalization task and handle unseen phrasings. Earlier weak
acquisition is not grounds for rejecting the feedback hypothesis. A subsequent
matched feedback study should use an adequately trained regime and retain a
full-vocabulary supervised control, rather than declaring failure from a short
underfitting run or conditional A/B scores alone.

It is **not** a novel personalization method, human preference-shaping result,
unseen-domain generalization or paper greenlight. No noisy-feedback learning
was performed in this calibration. Changed schedule/rank/wording/loss were tested
together; their individual contributions are not identified. The next feedback
comparison must be prospectively frozen as a separate experiment, not appended
to the finished six-arm run or portrayed as its passing replacement.

Root: `artifacts/hindsight_acquisition_cpu_20260904_v1`.
Receipt: `artifacts/hindsight_acquisition_cpu_20260904_v1_verified.json`.
The process exited successfully. No active GPU/CPU run remains from this stage.
