# EP interface diagnosis: prospective 24-call protocol

Status: prepared and CPU/tokenizer-tested; no diagnostic model inference performed.
Root reviews and commits this protocol before authorizing exactly one run. No
automatic continuation, prompt revisions, example replacement, or paper expansion.

## Why this is separate

The earlier EP0 smoke returned stationary on all 24 calls. Native ordered RGB,
estimated joint fields, and oracle joint fields each scored 25%. Its formal result
was SMOKE_ONLY_NO_SCIENTIFIC_DECISION. The real ACaM/MotionBench hypothesis remains
untested. This new diagnosis asks whether an elementary version of the task is
understood with native video versus ordered still images. It cannot establish
global/residual decomposition utility, novelty, or a paper go/no-go.

## Fixed design

Use the unchanged original `efference_pair.pilot.scene` renderer and `decompose`
RGB-only estimator. Six cases: object left/right/stationary under a fixed camera,
and camera left/right/stationary with the object and background fixed in world
coordinates. Directions are restricted prospectively, not selected from new
outputs. Motion is two pixels/frame, sixteen 224x224 frames, 2 fps, 30 pixels
endpoint displacement. Camera right corresponds to background image flow left.
The prompt explicitly explains this sign and specifies the coordinate frame.

Each case receives four conditions, totaling exactly **24 greedy calls**:

1. Native Qwen video using all sixteen exact PNG frames and explicit metadata.
2. Ordered images using the identical sixteen exact PNG frames in the same order.
3. Only the first image: temporal-information-negative control.
4. Text of RGB-derived background and red-square horizontal displacement per frame,
   without images: instruction/sign-semantics positive control.

All conditions use the same task question and A/B/C options. Option order is
constant across directions within each task, changed across task families; each
correct letter appears twice among six cases. Neither case IDs nor expected
direction names enter the prompt. All six first frames are byte-identical.
The stationary camera/object cases reuse identical full pixels, with different
questions: **not six independent physical scenes**. No significance test or
population-performance inference is supported. Numeric oracle text directly
provides motion measurements and is not evidence of visual competence.

Native video and ordered images receive matched source bytes, **not matched
processed visual-token budgets**. Verified processor grids: native video
`[8,14,14]`, ordered image grids sixteen copies of `[1,16,16]`. For the object
prompt, total input tokens are 536 and 1134, respectively. Hence a difference
supports a presentation/processor-bundle explanation, not an isolated temporal
encoder causal effect. No processor settings are changed to force parity.

## CPU and runtime preflight

Prepared data: `artifacts/ep_interface_diagnostic_20260904_v1`.
MANIFEST SHA-256: `87bb923231812e17d08247307942b63d3b28dda0bbc7a1f1cd03ba81931495e8`.
The inherited manifest schema says EP0-v3 because the unchanged seal utility is
reused; the dataset preflight and new runner record ep-interface-diagnostic-v1.
All 96 PNGs and cases/preflight JSON are sealed. RGB background consensus plus
red-pixel centroid displacements classify every case correctly. First/last image
differences are zero only for the two stationary cases. Two CPU tests pass.

Qwen/Qwen3-VL-8B-Instruct revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` is already cached on the assigned GH200.
The existing `.venv-vision` has transformers 5.15.0 and native VideoMetadata and
Qwen3VL support. CPU-only remote processing of all 24 inputs passed; no model was
loaded for that preflight. Receipt:
`artifacts/ep_interface_processor_preflight_20260904_v1.json`.
Native grids confirm eight temporal patch groups from sixteen frames; image mode
confirms sixteen image grids. No frame subsampling is allowed.

Frozen inference: bfloat16, greedy, maximum eight new tokens, no training, no
retries. Fixed order shuffle seed 90904. Record raw strings, grids, input tokens,
timings, runtime versions, model revision, runner/helper bytes and sealed inputs.
Only exact stripped A/B/C parses are accepted; no post-hoc format rescue.
All native input IDs and complete generated sequences (including input prefix)
are retained. Preflight now binds exact input IDs as well as grids, and the
verifier checks every inference contract against that preflight. The earlier
CPU-only receipt predates addition of input-ID logging; the actual run generates
a new strict preflight before any forward pass.

Hash every file only within the pinned public model snapshot, and preserve a
portable tokenizer/config snapshot with matching file hashes. Never enumerate
other cache repositories, tokens or credentials. Record all model loading keys
and fail on missing/unexpected/mismatched keys or loading errors. Record the
model generation configuration, explicit deterministic overrides and decode
options. The independent verifier re-decodes generated IDs using the portable
tokenizer, without loading model weights. Remote weight hashes are recorded;
the portable verifier does not claim to rehash absent model weights.

## Interpretation rules fixed before inference

Report raw 0–6 counts and per-case outputs for every condition, including failed
controls. No statistical significance, confidence interval, or paper decision.

- Numeric text errors indicate instruction/sign/format problems, so visual
  contrasts cannot establish temporal capability failure.
- Native video success on all three directions of a task, with ordered-image
  failure, supports that presentation bundle as an interface repair candidate.
- Success in both visual modes means the simplified apparatus is readable;
  it does not explain the old joint-field failure (prompt and task complexity
  changed compared with EP0).
- Numeric success but both visual modes failing leaves visual task/interface
  capability unresolved. It does not refute camera/residual decomposition.
- First-frame success on individual cases is unsurprising; within each task,
  byte-identical images and identical prompts cannot reveal direction. This is
  a diagnostic negative control, not a six-case significance test.

No outcome automatically authorizes full EP0, real-video G0, or new repairs.
Verifier always reports DESCRIPTIVE_ONLY_NO_EXPANSION.

## Launch and retrieval

After committing sources, root stages an isolated source snapshot and prepared
data, and launches with a fresh output directory:

```sh
HF_HOME=/home/ubuntu/Align_Paper/.hf_cache PYTHONPATH=src \
 /home/ubuntu/Align_Paper/.venv-vision/bin/python scripts/ep_interface_diagnostic.py \
 run --prepared artifacts/ep_interface_diagnostic_20260904_v1 --output FRESH_ROOT
```

Estimated cached wall time **3–10 minutes** including model loading, 24-input
processor preflight, generation, and sealing. This is a planning estimate, not
the old forward-only timing extrapolation. No meaningful learned checkpoints.
Root retrieves the entire sealed directory into a fresh destination, compares
remote/local archive SHA-256. Before launch, root pins runner and pilot-helper
file hashes from the actual staged archive in an external receipt. After retrieval,
run `verify --root RETRIEVED --output NEW_JSON --expected-runner-sha RUNNER_SHA
--expected-pilot-sha PILOT_SHA`. The verifier requires those externally pinned
hashes, rejects output paths inside the evidence tree and refuses an existing
output file. Root separately checks the run model-identity receipt against the
remote snapshot as needed; no model weight download is required.
Old artifacts and old scorers remain unchanged. Model cache is preserved.
