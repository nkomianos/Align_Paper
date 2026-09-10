# Representation alignment: a missing matched-preservation comparison

Status: source-verified research question, not an empirical finding about safety
or a GPU-ready reproduction. No dataset prompts, evaluation harnesses, model
weights or adversarial experiments were downloaded or run.

Primary sources: [September 3 paper](https://arxiv.org/pdf/2609.04022), Methods
pages 19-21; [author release](https://github.com/LingyuLi-Cogs/ReSO/tree/b7fbcf5b8ef61be8df74667e06b169795fb03955).
The paper compares representation training plus general-text forward KL against
DPO without a separate preservation term. Its shuffled representation control
also uses preservation. The methods describe full-decoder training on eight
H200s. The release confirms this objective distinction: DPO's replay data are
monitor-only, while the representation loss adds a weighted replay KL gradient.
Portable launch defaults differ from the paper's stated configuration; they
should not be treated as authenticated settings for the published tables.
The repository also acknowledges action-string overlap across annotation-level
splits. This does not itself establish overlap with external evaluations.

## Independently checked source evidence

Pinned commit: `b7fbcf5b8ef61be8df74667e06b169795fb03955`.
Eight downloaded source/configuration/documentation files passed Git blob checks
against the pinned tree. Receipts: `artifacts/reso_source_screen_20260910/`.
Reviewed training loss assembly and replay use, not merely README claims:

- `training/reso_train.py`: loss combines structure and beta times preservation;
  the resulting loss is differentiated.
- `training/dpo_train.py`: DPO plus optional chosen-response NLL is differentiated;
  replay is used during evaluation, not as a training term.
- Launch defaults: ReSO learning rate 1e-5, beta 0.1; DPO learning rate 5e-7;
  both default to 3,000 steps and eight processes.

These are facts about this release. We did not verify training logs, raw safety
scores, deployed checkpoints or the exact configuration used for the paper.

## Candidate question and necessary controls

Our inference: the comparison changes both the alignment objective and explicit
preservation. It does not isolate whether representational organization is
responsible for the difference between methods. This is not proof that
preservation explains the results. DPO's reference-dependent objective is also
not equivalent to having no regularization whatsoever.

The smallest interpretable follow-up is a 2x2 design: representation versus
behavioral objective, each with and without the SAME general-text preservation
term. Use the same base checkpoint, annotation groups, replay corpus, trainable
parameters, seed blocks and evaluation examples. Include the unchanged base and
shuffled-target representation control. Compare at both equal training budget
and matched held-out task performance; report actual reference drift rather than
assuming equal KL coefficients produce equal drift. Test whether the geometry
measure adds out-of-sample predictive information after accounting for drift
and task performance. Training checkpoints within a run are not independent
replications. Matching must use calibration data, not final target outcomes.

This question can first be studied with benign preference/decision tasks.
Such a study would test objective and preservation interactions, not establish
general safety or reproduce adversarial robustness. No security work is reopened.
Do not substitute benign results for the paper's safety conclusions.

## Admission and budget

A faithful full-parameter 8B reproduction is not a one-GPU drop-in job. Even an
optimistic allocation of BF16 policy weights and gradients, two FP32 Adam moments,
and a BF16 frozen reference costs about 112 decimal GB before activations and
buffers. Our AWS GPU exposes 97,887 MiB (about 102.6 decimal GB). Offloading or
sharding changes the resource plan; LoRA changes the experiment. The author
release's eight-GPU design cannot be assumed to fit or run cheaply on this host.

No honest neural runtime estimate is available without a representative preflight.
A smaller-model or adapter pilot would be a new developmental assay and must
first have a specified benign task, strong baselines and a novelty check against
existing preservation/continual-learning work. It cannot be counted as a faithful
reproduction. Do not launch a reduced implementation merely to keep the GPU busy.

Continue only if a concrete generalization outcome and differentiated contribution
survive that screening. Kill the paper pitch if matching preservation removes
the proposed new effect without yielding a useful independent result, or if it
amounts only to the established observation that replay reduces forgetting.
Current paper status remains NO-GO; this candidate has no empirical result yet.
