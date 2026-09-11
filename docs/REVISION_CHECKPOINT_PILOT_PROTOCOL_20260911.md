# Released-checkpoint evidence revision: exploratory admission

This pilot is authorized as bounded feasibility, not as a novel benchmark or
an already-qualified paper. Earlier screens rejected generic accuracy, entropy,
diversity and renaming theses. This endpoint is the interaction between adding
one released math adapter and response to new task evidence, conditional on
ordinary task capability. Neither a null nor a positive identifies the effect
of the training algorithm across runs.

Prior art is substantial: [Belief-R](https://aclanthology.org/2024.emnlp-main.586/)
studies revision; [CCOPD](https://arxiv.org/html/2605.30251v1) studies assistant
history contamination and a training remedy; [Resist and Update](https://arxiv.org/html/2607.12985v1)
is a further relevant lead whose full methods have not yet been audited. We
cannot claim evidence-responsiveness or history anchoring as a new phenomenon.
This modest checkpoint pilot supplies behavioral evidence that the existing
adapter release audit lacks; it does not resolve novelty by itself.

## Frozen design

Runner: scripts/run_revision_checkpoint_pilot.py. Eight newly generated DEV
parameter bases, two benign counting families: divisibility and nonadjacent
binary strings. Sixteen old/current gold answers are independently checked by
formula and enumeration. Parameters use seed 202609110317. These are eight
parameter instances, not eight independent task families.

Each base has a changed-parameter and unchanged-parameter final user message.
Each has four histories: clean current question; previous assistant old answer;
previous assistant current answer; neutral assistant placeholder. Old and current
numeric answers have equal decimal width. Initial and final user turns are
otherwise identical across the three history arms. On unchanged cases the two
numeric-anchor histories deliberately coincide; these are duplicate controls,
not independent samples. On changed cases the old answer becomes obsolete.
The current-answer assistant history is a constructed control, not a plausible
correct answer to the initial question. All histories are forced, not natural
on-policy conversations. The system explicitly prioritizes latest parameters.

64 inputs, both base and adapter: 128 outputs. Native thinking, greedy generation,
8192-new-token limit, batches of eight; no prompt truncation or continuation
repair. Use pinned local Qwen3-8B with the already hash-verified u-OPSD 8B thinking
adapter. The PEFT model is shared; disable_adapter gives the base arm. No training,
weight merge, new model download or GH200 access. Historical base revision is
unknown, so this is the adapter's effect on our pinned base, not exact reproduction.

Save all prompts, token outputs, completions, timing and base/adapter hashes.
Score only a final ANSWER: integer line; malformed and token-limited results stay
in the denominator and are also reported separately. No correctness filtering
of individual cases after seeing results.

## Gates and next action

Require EOS and parse rates >=95% separately for each model, clean accuracy
>=15/16 for each model, and unchanged old-answer accuracy >=7/8 for each model.
Otherwise stop this pilot as inadequate capability/interface for attribution.
Do not label an adapter's general capability loss as evidence-revision loss.

A candidate harmful interaction requires at least two additional errors out of
eight in the adapter's changed/old-answer arm relative to base, while neutral
changed and old-answer unchanged differences are each at most one error in
absolute value. The changed/current-answer control must reach >=7/8 in both
models. These are exploratory allocation rules, not statistical significance.
Report every arm even when the pattern is absent, reversed, or mixed.

No such pattern: stop this checkpoint-revision lead without a parameter/prompt
sweep. A qualifying pattern permits designing a fresh larger paired test,
natural-history controls, and a second independently trained checkpoint;
it does not authorize training or a paper claim automatically. Exact runtime
for those conditional experiments remains unestimated until this pilot runs.

Pilot planning estimate: 5–25 minutes if these short counting problems complete
normally. The conservative token ceiling is 1,048,576 generated tokens before
padding/early termination; pathological long traces could take substantially
longer (potentially hours). Let the fixed pilot finish; no wall-clock kill. Record
actual timing and do not convert program time to provider billing or H200 hours.
