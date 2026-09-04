# Video removal: novelty check and revised experiment

## Literature-driven decision

Do not pitch generic localization versus preservation as new. GenEraser already
decouples Locator and Preserver and ablates each while fixing the other
(Tables 6–7): https://arxiv.org/html/2605.30045v1 . EffectErase explicitly discusses
mask reliance and trains effect-aware localization:
https://arxiv.org/html/2603.19224v1 . ROSE predicts effect difference masks:
https://arxiv.org/abs/2508.18633 . BeyondMasks deliberately evaluates effects
outside object masks: https://arxiv.org/html/2608.20107v1 .

These are substantial overlaps, not proof that every controlled comparison is
already done. A simple oracle-mask improvement is insufficient for a standalone
ICLR claim. The remaining empirical question must identify a consequential
mechanism beyond these established observations.

## A sharper distinction: input information versus output permission

Changing a mask can simultaneously (1) hide more context from the generator and
(2) permit more pixels to change in final compositing. These are different
interventions. A single object-mask versus effect-mask comparison mixes them.

Proposed paired 2x2: conditioning mask is object-only or effect-inclusive;
output composite mask is independently object-only or effect-inclusive. Generate
twice per clip/seed, retain raw pre-composite output, and composite each output
twice. Also retain unrestricted raw output as a diagnostic, not a competing
method. All cells use identical source video and frozen generator parameters.
Effect-support annotation supplies locations only, never reference pixels.

This requires confirming that the chosen implementation exposes both operations
separately. If final compositing also occurs internally during denoising, the
two-factor interpretation is invalid until those operations are explicitly
accounted for. Do not silently edit a method into a different algorithm.

Primary contrasts: post-compositor restriction at fixed conditioning; conditioning
information loss at fixed output support; and their interaction. Report effect
restoration and unrelated-region changes separately. Compare area-matched
expanded support, not just a privileged oracle. Stratify localized reflections
versus global lighting; do not treat global relighting as unrelated corruption.

## Scientific decision rule

This remains a diagnostic candidate, not a paper greenlight. To justify expansion,
the factorial must explain an otherwise misleading performance comparison or
support a practically useful intervention across models, beyond the inevitable
fact that a hard compositor cannot change pixels outside its support. If all
results reduce to that deterministic identity, stop the paper direction.

Before GPU use: audit the full inference path, establish effect annotations and
alignment, pin clips and seeds, and implement the independent composite contract.
The three inspected clips are development data only; they cannot serve as an
unseen confirmation set. No full evaluation or GPU job has been launched.

## Executed source-path inspection

Pinned DiffuEraser `8e6f279ac7531e27ad1849c6f8dab5372a8597e7`,
`diffueraser/diffueraser.py`, reveals additional factors:

- `forward` requires a `priori` video, which supplies latent initialization.
  If it is regenerated under different masks, the conditioning contrast also
  changes the prior. Holding it fixed measures a conditional effect instead.
  Both choices must be named; neither identifies a pure information-loss effect.
- A minimum of 22 effective frames is checked after reading the three inputs.
  Development sample 175 contains only 20 decoded frames. Do not silently repeat
  frames to force eligibility; inspect reader resampling and predeclare any
  temporal adaptation before applying the model.
- Above `nframes*2` effective frames, pre-inference replaces sampled-frame latents
  and images and sets those masks to black. The original mask copy is retained
  for final compositing. Thus mask intervention propagates beyond a single input.
- Final blending occurs after `self.pipeline(...).frames`, using the preserved
  original mask copy. Saving that pre-blend output can isolate the final
  compositor effect without rerunning the generator. This does not isolate all
  internal support constraints; pipeline internals still require inspection.

Decision: keep this as a mechanistic pilot under design. Do not represent the
current two-generation proposal as a fully isolated 2x2 experiment yet. The prior
and temporal processing contracts are required parts of the frozen protocol.

## Pipeline inspection and implemented replay

Inspected the imported `pipeline_diffueraser.py` at the same pinned revision:
https://github.com/lixiaowen-xw/DiffuEraser/blob/8e6f279ac7531e27ad1849c6f8dab5372a8597e7/diffueraser/pipeline_diffueraser.py .
The inspected denoising loop uses encoded masked images and masks as BrushNet
conditioning, updates overlapping latent windows, and decodes to frames before
the wrapper's final composite. No source-pixel composite was found in that loop.
This supports a post-generation output-permission contrast, not a pure causal
interpretation of changing the conditioning mask. VAE conditioning sampling also
needs its RNG captured; the supplied scheduler generator alone is insufficient
for a fully paired stochastic run.

Implemented `interaction_sprint.video_composite_replay`: with fixed generated
pixels, independently vary output alpha and report region-specific reference MAE,
source change, and an unavoidable error contribution from exactly-zero alpha.
Four CPU tests pass for endpoints, fixed-support error floor, soft-alpha semantics,
invalid inputs, and nonmutation. This is normalized float-domain analysis; it
does not claim bit-exact reproduction of the wrapper's uint8 quantization or
video encoding. Reference pixels enter scoring only, never generation.

The hard-support floor is an elementary deterministic diagnostic, not a proposed
theoretical contribution. We still need actual generated pre-composite frames
and qualified region annotations before the tool can answer a scientific question.
