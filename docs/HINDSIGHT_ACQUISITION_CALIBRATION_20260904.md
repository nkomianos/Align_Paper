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
