# Candidate: Patch-Phase Instability in Vision-Language Models

## PI status

**Independent backup candidate; frozen G0 is CPU-validated and awaits an
explicit GPU launch.** It ranks below effect consistency because translation
sensitivity of vision transformers is established; the paper is viable only
if modern MLLM primitive failures are strongly phase-locked and the mitigation
transfers across families.

## Paper question

Recent benchmarks show a large gap between frontier MLLMs and young children on
low-semantic visual primitives. [KidVis](https://arxiv.org/abs/2601.08292)
reports 67.33 for GPT-5 versus 95.32 for children and a scaling paradox;
[BabyVision](https://arxiv.org/abs/2601.06521) independently reports severe
deficits across 22 primitive subclasses. Neither result identifies whether a
substantial portion of these failures is locked to the phase of thin visual
features relative to the encoder's patch grid.

Patch embedding is a plausible causal bottleneck, but not a new observation by
itself. [Reviving Shift Equivariance in Vision
Transformers](https://arxiv.org/abs/2306.07470) and [Making Vision Transformers
Truly Shift-Equivariant](https://arxiv.org/abs/2305.16316) show that patching,
positional encoding, and subsampling can make ordinary ViTs shift-sensitive.
The proposed contribution must therefore be specific and stronger: phase must
explain a meaningful share of *MLLM visual-primitive* errors, its period must
track independently different vision encoders, and a compute-matched phase
ensemble must outperform repeated decoding of one image.

## Necessary causal prediction

G0 renders identical black-pixel sprites at integer horizontal translations.
The sprite pixels are copied without resampling; only their alignment to the
model's patch grid changes. Thin and thick versions have the same label. If
patch aliasing is a real cause, then:

1. thin primitives should flip answers across phases more often than thick
   controls;
2. answer agreement should recur at the frozen encoder patch period rather
   than neighboring lags; and
3. voting across four fixed patch phases should beat four stochastic samples
   of the canonical rendering at the same model-call budget.

The gate uses counting, closure, and path-crossing scenes, 60 base scenes, 32
integer phases, and both thicknesses. It tests
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), whose
official config declares a 16-pixel patch size, and
[Gemma-3-12B-IT](https://huggingface.co/google/gemma-3-12b-it), a distinct
multimodal family. Gemma access requires accepting Google's Hugging Face terms;
this is an infrastructure prerequisite, not a scientific degree of freedom.

## Frozen G0 decision

Every condition must pass in both families:

1. at least 15% of thin base scenes change answer across phase;
2. thin flip rate exceeds thick flip rate by at least 5 points;
3. agreement at the predeclared patch-period lag exceeds its two neighboring
   lags by at least 3 points; and
4. the four-phase TEST ensemble improves accuracy over four canonical-image
   stochastic samples by at least 5 points, with a paired bootstrap lower bound
   above zero.

The decision is `EXPAND_PATCH_PHASE_STUDY` only if all checks pass. A failure
kills this explanation; ordinary position sensitivity or one-model gains are
not enough.

## If G0 passes

Expansion must reproduce with a second renderer and real KidVis/BabyVision
subsets, include vertical and two-dimensional phases, inspect hidden vision
features where accessible, and compare phase voting with adaptive polyphase
anchoring, antialiasing, resolution scaling, and ordinary test-time
augmentation. The full claim cannot be made from synthetic shapes alone.

The gate contains 3,840 deterministic phase images plus 360 same-image
stochastic controls per family. The expected GH200 time is roughly 2--5 hours
for both VLMs, dominated by image prefill. The launch entrypoint is
`scripts/run_visual_patch_phase_g0_remote.sh`.
