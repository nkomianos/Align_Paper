# Candidate: Efference-Pair for Frozen Video VLMs

**Status:** green-light for one bounded G0 only; vision backup, ranked below validator-monoculture security testing.

## One-sentence claim

A frozen video VLM should reason more accurately about camera and object motion when apparent motion is decomposed into two explicitly named, token-matched visual channels—a global camera-motion channel and a camera-stabilized residual-motion channel—than when it receives raw RGB, more RGB, or undecomposed optical flow.

## Exact causal hypothesis

Current video VLMs frequently confound camera ego-motion with object motion because both arrive as one apparent-motion field before vision-language alignment. A training-free **Efference-Pair** adapter will estimate the dominant global transform between adjacent frames, then expose:

1. a **global channel** rendering the estimated camera transform, and
2. a **residual channel** rendering motion after compensating for that transform.

The hypothesis is not merely that an engineered visualization improves accuracy. It predicts a double dissociation under a fixed visual-token budget:

- the global channel selectively improves camera-motion questions;
- the stabilized residual channel selectively improves object-motion questions; and
- the joint pair improves both strata more than either channel alone, extra RGB, or raw flow.

No model weights, adapters, prompts, or motion estimators are trained on the evaluation data. The VLM and motion estimator remain frozen.

## Mechanism to test

For each clip, decode a fixed set of frames and estimate dense adjacent-frame flow with a frozen public estimator. Fit one robust global transform per adjacent pair from background-supported correspondences, using a prespecified affine-or-homography estimator with RANSAC. Accumulate transforms relative to the first anchor frame.

- **Global camera view:** render the normalized global displacement field, trajectory, axes, sign, and magnitude on a sparse neutral canvas.
- **Stabilized residual view:** warp frames into the anchor coordinate system and render residual flow or short residual trails over sparse RGB anchors.
- **Joint view:** provide both named views beside the same raw anchors.

Estimator version, transform family, RANSAC threshold, frame rate, rendering scale, and all exclusion rules must be committed before any scored model output is inspected. Estimator failures remain in the intention-to-treat analysis; an estimator-success subset may be reported only as secondary diagnosis.

## Novelty boundary and closest-work collision check

The defensible contribution would be the **training-free global/residual decomposition plus a causal double-dissociation evaluation in a frozen VLM under token-matched controls**. The paper must not claim that optical flow, motion visualization, temporal tokens, stabilization, or camera-motion supervision are themselves new.

Closest primary work:

- [Time Blindness / SpookyBench](https://timeblindness.github.io/) already shows that a training-free Farneback motion-boundary overlay can substantially improve some temporal judgments, while dynamic scenes remain difficult ([paper](https://arxiv.org/abs/2505.24867)). This directly kills any generic claim that “showing flow to a VLM” is novel; its overlay is a mandatory baseline.
- [4DP-QA](https://research.nvidia.com/labs/lpr/4dpqa/) introduces True-Motion Point Tracking with a fixed reference camera and trains a VLM for 4D reasoning. It is the closest conceptual collision. The proposed distinction is a frozen, dual-view interface and an explicit camera/object selectivity test rather than post-training.
- [ACaM](https://1yuwen.github.io/ACaM-Project-Page/) documents severe camera-motion confusions and improves them through supervised fine-tuning. It motivates the failure mode but does not establish the proposed frozen double dissociation.
- [MASS](https://arxiv.org/abs/2511.18373) serializes tracked 3D motion profiles into language and then post-trains the model; it explicitly identifies robust ego-motion handling as unfinished work. Efference-Pair instead supplies paired visual evidence without training.
- [Delta Dynamics](https://arxiv.org/abs/2605.20576) trains a Qwen-based model on synthetic rigid-body scenes with optical-flow inputs. It collides with flow-conditioned reasoning, but not with a frozen, general-video camera/residual intervention.
- [MASH-VLM](https://openaccess.thecvf.com/content/CVPR2025/papers/Bae_MASH-VLM_Mitigating_Action-Scene_Hallucination_in_Video-LLMs_through_Disentangled_Spatial-Temporal_Representations_CVPR_2025_paper.pdf) learns disentangled spatial and temporal representations, including frame-difference information. It makes a broad “disentanglement” claim unsafe; the candidate is narrower and training-free.
- [IG-VLM](https://arxiv.org/abs/2403.18406) is a training-free image-grid interface for video understanding. It makes extra-RGB and layout-matched controls essential.
- [MotionBench](https://openaccess.thecvf.com/content/CVPR2025/papers/Hong_MotionBench_Benchmarking_and_Improving_Fine-grained_Video_Motion_Understanding_for_Vision_CVPR_2025_paper.pdf) and [VLM4D](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_VLM4D_Towards_Spatiotemporal_Awareness_in_Vision_Language_Models_ICCV_2025_paper.pdf) already benchmark fine-grained motion and spatiotemporal reasoning. A benchmark-only contribution would therefore be weak.

As of the 2026-08-30 collision search, no primary paper located here tests this exact frozen dual-channel intervention with the predicted double dissociation. That is an evidence-limited search conclusion, not proof of novelty. A fresh Semantic Scholar/arXiv/OpenReview collision audit is mandatory before expansion or paper positioning.

## Frozen G0 design

### Model and decoding

- Primary model: `Qwen/Qwen3-VL-8B-Instruct`, frozen.
- Resolve the model repository to an immutable commit SHA before the first scored forward pass and record it in the run manifest.
- Greedy decoding, temperature 0, one answer token/letter where the benchmark is multiple choice, and no self-consistency sampling.
- One frozen prompt template across conditions. Only the truthful channel legend changes; question text, answer order, and response format do not.
- Exactly 16 visual canvases per item, each rendered at 448 x 448. The processor-reported vision-token count must be logged and equalized across conditions before scoring.

### Six conditions

Every item is evaluated in all six conditions, in deterministically shuffled order:

1. **`native_rgb`** — 16 uniformly sampled RGB frames.
2. **`rgb_layout_control`** — eight common RGB anchors plus eight additional RGB frames occupying the same labeled auxiliary slots used below; this controls visual budget, layout, and channel labels without derived motion.
3. **`raw_flow_control`** — eight common RGB anchors plus eight undecomposed dense-flow renderings. This is the direct “flow helps” baseline.
4. **`global_camera_only`** — eight common RGB anchors plus eight global-transform renderings.
5. **`stabilized_residual_only`** — eight common RGB anchors plus eight camera-compensated residual-motion renderings.
6. **`efference_pair_joint`** — eight common RGB anchors plus four global-transform and four stabilized-residual renderings.

The raw anchors, timestamps, canvas count, resolution, prompt length, answer order, and maximum output tokens are fixed per item. If exact processor token equality cannot be achieved, conditions are padded down to the minimum common token budget before the run; a post-hoc budget correction is forbidden.

### Sealed evaluation slice

The G0 contains 96 multiple-choice items, selected with seed `43117` and sealed before inference:

- **48 ACaM items:** eight static clips, eight object-centric track/arc clips, and 32 non-static camera-motion clips balanced across rotation, translation, and focal-length change, with inverse directions balanced where labels permit.
- **48 MotionBench development items:** 24 Camera Motion (`CM`) and 24 Motion-related Objects (`MO`) questions, balanced over answer position and available source domains. Only answer-accessible development data may be used; the hidden test split is excluded.

Primary strata are `camera_motion` (dynamic ACaM camera classes plus MotionBench-CM) and `object_motion` (ACaM track/arc plus MotionBench-MO). ACaM static clips form a prespecified negative-control stratum. Overall accuracy is the macro-average of dataset-by-stratum cells so that the larger camera stratum cannot dominate.

VLM4D is reserved for confirmation after a pass; it is not a dependency of G0.

### Causal perturbation controls

A sealed 24-item subset with directionally interpretable labels receives three additional probes:

1. **Sham pairing:** attach a magnitude-matched global transform and residual field from another clip in the same source/category.
2. **Sign reversal:** reverse the relevant global or residual vectors while leaving raw anchors unchanged. On eligible questions, the alternative answer corresponding to the reversed direction is specified before inference.
3. **Temporal shuffle:** permute only the derived-channel time order while preserving every canvas and the raw-anchor order.

These controls test whether the model uses the represented motion direction and temporal structure, rather than merely benefiting from color, layout, extra edges, or a “motion analysis” cue.

## Metrics and expected signature

The primary endpoint is exact multiple-choice accuracy. Report paired item-level differences, dataset-stratum macro-averages, and percentile paired-bootstrap 95% confidence intervals with 10,000 resamples. The two selectivity contrasts are:

```text
global_selectivity = delta(global_camera_only, native_rgb | camera_motion)
                   - delta(global_camera_only, native_rgb | object_motion)

residual_selectivity = delta(stabilized_residual_only, native_rgb | object_motion)
                     - delta(stabilized_residual_only, native_rgb | camera_motion)
```

The hypothesized signature is not a uniform lift. It is positive global selectivity, positive residual selectivity, a joint improvement on both primary strata, no meaningful benefit from sham channels, directional sensitivity to sign reversal, and loss of benefit after temporal shuffling.

## Frozen pass, kill, and invalidity rules

Return **`EXPAND_EFFERENCE_PAIR`** only if every criterion below is met:

1. `efference_pair_joint` improves overall macro-accuracy by at least 8 percentage points over `native_rgb`, with the paired-bootstrap 95% interval excluding zero.
2. The joint condition improves both the camera-motion and object-motion strata by at least 10 percentage points over native RGB.
3. `global_selectivity` and `residual_selectivity` are each at least 5 percentage points, and each paired-bootstrap 95% interval excludes zero.
4. The joint condition beats `rgb_layout_control` by at least 5 percentage points and `raw_flow_control` by at least 3 percentage points overall.
5. On direction-eligible controls, sign reversal reduces canonical-answer accuracy by at least 10 percentage points and increases the prespecified reversed-direction answer rate by at least 10 percentage points relative to sham; temporal shuffle removes at least half of the joint condition's gain.
6. On the ACaM static negative controls, the joint condition degrades accuracy by no more than 2 percentage points relative to native RGB, and sham pairing improves accuracy by no more than 3 percentage points.

If any criterion fails, return **`KILL_EFFERENCE_PAIR`** as a paper direction. The result may still motivate an engineering note, but thresholds, strata, transforms, or prompts must not be relaxed post hoc to manufacture a pass.

Return **`INVALID_G0`**, not pass or kill, only for a predeclared integrity failure: wrong model revision, unequal visual-token budget, corrupted or mislabeled source media, missing outputs, use of hidden-test answers, or inability to verify artifact hashes. Estimator failure on difficult clips is part of the scientific result and is not an invalidity excuse.

## Dataset and licensing constraints

Benchmark availability does not imply permission to redistribute source videos.

- ACaM aggregates clips from several prior benchmarks and curated YouTube material. Before sealing, inspect the ACaM dataset card and every upstream component's current research-use and redistribution terms. Store source IDs, checksums, timestamps, and transformation manifests; do not commit or republish media unless its license explicitly permits it.
- MotionBench uses heterogeneous video sources. Use only the official answer-accessible development split under its stated terms, preserve original identifiers, and never infer or expose hidden-test answers.
- Derived flow, stabilized frames, and motion canvases may inherit restrictions from their source media. Keep them out of the public repository unless redistribution is permitted; publish deterministic generation code and manifests instead.
- Pin and record licenses for the dense-flow estimator, transform/stabilization dependencies, and model weights. A research-only estimator is acceptable for G0 but may constrain artifact release and later paper claims.
- If a clip's provenance or license is ambiguous, replace it before the slice is sealed. No replacement is allowed after model outputs are inspected. If too few legal clips remain, use a separately preregistered synthetic controlled set rather than silently changing the target population.

## Runtime and resource estimate

With model weights and data cached on one GH200:

- preprocessing and motion decomposition for 96 clips: approximately 10–25 minutes;
- 576 main forwards plus approximately 72 perturbation-control forwards: approximately 35–85 minutes;
- hashing, verification, and analysis: approximately 5–10 minutes.

Expected wall time is **about 1–2 hours cached**. A cold model/data download or slow video decoding can raise this to **2–4 hours**. The gate requires no training and produces no learned checkpoint; only immutable evidence, manifests, derived features, and logs need preservation.

## Why this ranks behind validator-monoculture security testing

Efference-Pair is the strongest vision backup in this slate, but it should not displace the validator-monoculture candidate yet:

1. **More crowded novelty neighborhood.** Flow prompting, fixed-reference motion, learned spatial/temporal disentanglement, and camera-motion supervision already exist. The remaining novelty is specific and could be judged as a classical-vision wrapper.
2. **Less clean ground truth.** Camera/object labels, source-video artifacts, and motion-estimator quality introduce ambiguity. Validator-monoculture security experiments can use executable outcomes and provenance with sharper causal attribution.
3. **Higher licensing and reproducibility friction.** Public video benchmarks often inherit heterogeneous media terms, whereas programmatic security tasks are easier to package and independently reproduce.
4. **Narrower first-order impact.** A pass would establish an important interface failure and a cheap mitigation, but not yet broad spatiotemporal intelligence. It would still require a second model family, VLM4D confirmation, estimator ablations, and likely human-error analysis.
5. **Slower and more failure-prone G0.** Video decoding, optical flow, stabilization, and multimodal token accounting create more ways for a sub-two-hour gate to become invalid.

The candidate remains worth one bounded run because the predicted double dissociation is mechanistic, falsifiable, and potentially useful without pretraining. It becomes a paper program only on the frozen pass signature above.
