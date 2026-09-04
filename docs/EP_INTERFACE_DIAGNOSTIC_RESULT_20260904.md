# Efference-Pair interface diagnosis: no expansion

The prospectively fixed 24-call diagnosis completed and was locally verified.
Decision: **DESCRIPTIVE_ONLY_NO_EXPANSION**. Changing ordered still images to
native video did not repair this elementary apparatus. This is not a rejection
of the original global-camera/residual-motion hypothesis, whose real-video gate
remains unrun.

## Results

| Presentation | Object direction | Camera translation | Total |
|---|---:|---:|---:|
| Native video, 16 frames | 1/3 | 1/3 | 2/6 |
| Ordered images, same 16 PNGs | 1/3 | 1/3 | 2/6 |
| First frame only | 1/3 | 1/3 | 2/6 |
| RGB-derived numeric displacement text | 2/3 | 2/3 | 4/6 |

All 24 responses parsed as exact A/B/C. Native video and ordered images answered
stationary for every case, including clear left/right motion. The first-frame
condition answered stationary for all object questions and left for all camera
questions; this is consistent with identical pixels and identical within-task
prompts, not recovery of temporal information.

Numeric text failed object-left (answered right) and camera-right (answered left).
The CPU RGB oracle itself classified all six cases correctly. Numeric failures
therefore leave instruction/sign interpretation unresolved; numeric text is not
a visual-comprehension test. We cannot attribute visual failure exclusively to
the temporal encoder. The native and image processors also use different grids
and token budgets despite identical source frames, as disclosed prospectively.

There are only three direction cases per task, shared scene texture, and duplicate
stationary pixels across tasks. These are descriptive apparatus checks, not
independent benchmark trials or evidence for a population effect. No significance
test, paper kill, or full-EP0 authorization follows.

## PI interpretation

Do not run the old 450-call synthetic gate or the unfinished real-video gate on
the strength of this result. Native video alone did not rescue the presentation.
Additional interface repairs require a new concrete rationale and authorization,
not repeated prompt/example adjustments until an oracle passes. Preserve the
original idea as untested/apparatus-unqualified rather than scientifically
disproved. Other independently qualified experiments may proceed.

## Provenance and verification scope

- Frozen scientific source commit: `42fe2d5`.
- Source archive SHA-256:
  `78015da58264c25be71ccb2313a65d60f1bfec038cd48ca96ab1c37e6293ecdf`.
- Executed runner SHA-256:
  `f28b7fd1bdad93e9d6a78587d20525c658fef604d7d9972863b1a7166ef395d2`.
- Executed unchanged pilot helper SHA-256:
  `6081fa23823e65a008dbeed5e8bbc6f21da76d1171422623876f5ce7dfb5414c`.
- Prepared manifest SHA-256:
  `87bb923231812e17d08247307942b63d3b28dda0bbc7a1f1cd03ba81931495e8`.
- Retrieved complete archive SHA-256, recomputed locally for this memo:
  `8b56aec8d01096b7c5f258560bf1cf2b6e2a5dd24da7911969c2b33d75609762`.
- Completed run manifest SHA-256, recomputed locally:
  `3fbf72900bd550a8b69e67d5db9612f8cca2b089e60714e54657afc648c1ea8e`.

Evidence is under `retrieved/ep_interface_20260904T1035Z`; completed run directory
is `ep_interface_cached_20260904T1034Z`, with `verified.json` beside it. The report
checks source hashes against externally supplied staged-archive pins, all sealed
files, prepared identity, output completeness, exact input-token/grid contracts,
generated prefixes/lengths, and independent re-decoding with the copied tokenizer.
Tokenizer files match the recorded pinned public snapshot hashes. Generation
configuration is preserved. Loading missing/unexpected/mismatched/error lists are
all empty. These checks do not independently recompute neural logits, and the
portable verifier does not rehash absent multi-GB model weights.

Model: Qwen/Qwen3-VL-8B-Instruct at
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, GH200, torch 2.7.1+cu128,
transformers 5.15.0, `.venv-vision/bin/python`.

Two setup failures are retained alongside the successful run:

1. T1030Z wrapper selected `.venv-interaction`, which lacks OpenCV; it stopped
   before model inference. Retry used the previously preflighted vision env.
2. T1032Z local-only snapshot resolution required absent public `.gitattributes`
   and `README.md`; it stopped before model loading. Only those two files were
   downloaded at the pinned revision. Every one of 14 preexisting snapshot files,
   including all four weight shards and tokenizer assets, was SHA-256 checked
   before and after and remained unchanged. Added metadata hashes were
   `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361`
   and `6d5d06e0c3f069097002445d30dce9ee107db3afaf15563f09e6df49b8dcb4d7`,
   respectively.

No scientific source, input scene, prompt, condition, or outcome was changed for
either retry. Timestamp-like directory suffixes are identifiers, not an exact
wall-clock chronology. See `GPU_WINDOW_20260904.md` for execution coordination.
