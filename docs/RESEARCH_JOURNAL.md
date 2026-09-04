# Researcher journal

Last consolidated: **2026-09-02**. This is the current experiment index; historical
protocols and the thesis scout retain their original context. The short ledger
is in the [README](../README.md). Outcomes are reconstructed from the saved
reports below, not from memory or optimistic chat summaries. The
[machine-readable evidence index](RESEARCH_EVIDENCE_INDEX_20260902.json) contains
SHA-256 fingerprints for all 15 source reports. It identifies the report bytes
used here, **not** a claim that all historical inferential verifiers were rerun
today. `scripts/journal_fingerprints.py` rebuilds it from an explicit allowlist
without reading raw trajectories or private keys.

## Recording rules

- Give each hypothesis and empirical stage a stable ID. Retried launches,
  snapshots, software tests, and model families are not independent paper ideas.
- Record the hypothesis, comparison, population/model, frozen gate, effect size,
  uncertainty where available, validity failures, evidence path, and PI decision.
- Separate `valid negative`, `invalid assay/model organism`, `developmental`,
  `apparatus only`, and `not run`. A stopped project is not automatically a
  disproved scientific thesis. An interesting effect is not automatically a paper.
- Preserve raw completions, inputs, configs, dependency attestations, partial
  failures and checkpoints. Never overwrite retrieved evidence. Derived reports
  belong outside sealed roots. Retain original reports when an audit corrects them.
- Do not tune thresholds or drop failed cases after inspecting scored results.
  A genuinely different experiment needs a new dated protocol and fresh evidence.
- Before a long run: test formatting, comprehension, visual-token accounting,
  intended training response, and required comparison matching on a **separate
  small setup slice**. Benchmark throughput on the exact model. Stop at failed
  prerequisites. Do not pay for a second family merely to confirm a broken assay.
- Code tests and simulator-oracle success establish apparatus correctness only.
  Keep restricted media, private answer keys, credentials and large model files
  out of Git. Original Under Extinction locked TEST remains unopened.
- Future entries must include actual start/end times and billed-resource data
  when available. Do not reconstruct precise GPU-hours from chat estimates.

## Completed empirical stages

### UE-1 — Under Extinction Stage 1 (2026-08-19)

**Hypothesis:** passive transition/value revaluation distinguishes genuinely
rewarded policies from proxy-rewarded policies despite matched ordinary behavior.
**Run:** paired Qwen3.5-9B reward-acquisition arms, seed 11, checkpoints at
0/30/75/150/225/300 optimizer updates; 1,792 DEV cases per checkpoint.
The unchanged base and checkpoint-zero controls were retained.

**Finding:** software smoke checks passed; both arms learned their respective
late-TRAIN objectives. Registered Stage-1 DEV and replication gates failed on
comprehension, channel specificity, reversal and within-cell robustness.
Decision: `STOP_OR_DEBUG_WITHOUT_OPENING_LOCKED_TEST`.

**Interpretation:** the learned-policy assay was not established. Do not claim
latent objectives were identified or that the general controller hypothesis was
refuted. No locked-test replication or predictive Gate C was run.
Evidence: [public DEV report](../results/stage1_20260819/stage1_dev_report.json),
[run log](RUN_LOG_20260819_STAGE1.md), release/tag
`stage1-dev-20260819-failed`; full local archive `retrieved/stage1_20260819/`.

### UE-DID — post-failure diagnostic (retrieved 2026-08-25)

**Hypothesis:** separate objective retention, static causal parsing, passive-update
integration, planner composition and response-label effects.
**Run:** four frozen policy conditions, 19,200 new DEV cases each: 76,800 score
rows plus 1,024 unconstrained format probes. No new training or original TEST.

**Finding:** `LABEL_INTERFACE_CONTAMINATED_REDESIGN_RESPONSE_INTERFACE`.
Exact-format parse rate was 100%, but mean label-equivariance error was 0.189686;
semantic choice agreement under label swaps was 75.09375% (worst policy 68.6875%).
The worst cell equivariance error was 0.431215. Thus parse success alone did not
make the A/B measurement valid. The broader diagnostic gates also did not pass.

**Decision:** retire the current Under Extinction assay; retain the checkpoints.
Evidence: `retrieved/did_v1.1.1_20260825/analysis/did_v1.1.1_report.json`;
[preregistration](DEV_DIAG_PREREGISTRATION.md). This is a diagnostic stage of UE,
not another independent successful/failed paper.

### PA-0 — provenance authority (2026-08-25)

**Hypothesis:** self-authored instructions exert more authority than matched
external instructions, with the predicted horizon/recency pattern.
**Run:** forced-likelihood Qwen gate; 512 paired units.
**Finding:** self-minus-external probability = 0.00027724 = **0.027724 pp**,
95% CI **[0.023908, 0.031702] pp**; required effect was 10 pp. All three gates
failed; horizon gain was negative.
**Decision:** `STOP_PROVENANCE_AUTHORITY_LINE`; tiny nonzero effect is not a
material mechanism. Evidence:
`retrieved/provenance_authority_g0_20260825/analysis/report_local.json`.

### RI-0 — response-interface invariance (2026-08-25)

**Hypothesis:** semantically identical actions encoded as single tokens, JSON,
Python calls or XML materially change choices.
**Run:** 128 matched units, Qwen forced-likelihood gate.
**Finding:** selection disagreement **0**; within-unit probability spread
**0.105105 pp**, CI **[0.090372, 0.120507] pp**. All three continuation checks
failed. **Decision:** `STOP_INTERFACE_INVARIANCE_LINE`.
Evidence: `retrieved/interface_invariance_g0_20260825/analysis/report_local.json`.

### HM-0 — hybrid-memory safety carrier (2026-08-25)

**Hypothesis:** Qwen3.5's recurrent state is a material causal carrier of a
retained behavioral constraint, beyond attention K/V storage.
**Finding:** baseline accuracy 100%; authorized margin 7.311 logits. Recurrent
carryover **0.168 logits**, CI **[0.144, 0.193]**, below the 0.50 bar. Attention
K/V carryover **14.449 logits**. **Decision:** `STOP_HYBRID_MEMORY_LINE`.
This was a valid, small mechanistic effect, not a loading or parsing failure.
Evidence: [result](HYBRID_MEMORY_G0_RESULT.md),
`retrieved/hybrid_memory_g0_v1_1_20260825/report_local.json`.

### REC-0 — recency-gated switching (2026-08-28)

**Hypothesis:** a learned recency direction causally mediates policy switching.
**Run:** two seeds, readout/steering/erasure and matched controls.
**Finding:** every registered gate failed both seeds. The later non-retroactive
matched-cue audit gave switching differences **−0.072 pp / +0.704 pp** and
steering contrasts **0.147 pp / 0.096 pp**. No specific necessity signature.
**Decision:** `KILL_CANDIDATE`, no G1.
Evidence: [final result with hashes](RECENCY_G0_FINAL_RESULT.md);
`retrieved/recency_g0_20260828T0432Z/run/gate_report.json` and
`retrieved/recency_g0_corrected_audit_20260828T0619Z/`.
The audit applied interventions to the cue-only control as well; it did not
replace the original report or rescue its decision.

### J-0 — recipe-invariant intervention selection (2026-08-28)

**Hypothesis:** intervention directions agreed upon by two training recipes
transport causally to a held-out third recipe.
**Run:** seeds 9201/9202; layer/direction selected using A/B only.
**Finding:** source agreement **0.9872 / 0.9477**, but held-out signed steering
**−0.095 / −0.224 pp** (lower bounds **−0.667 / −0.661 pp**). Necessity,
specificity and baseline margin failed; preservation passed.
**Decision:** kill the selection rule as this paper's central method, not the
existence of all possible causal mechanisms. No external replication.
Evidence: [final result](RECIPE_INVARIANT_J0_FINAL_RESULT.md),
`retrieved/recipe_invariant_j0_20260828T0710Z_verified_20260828T0815Z/`.

### SA-DEV — semantic ancestry developmental run (2026-08-28–29)

**Hypothesis:** distinct-looking rewritten sources may inherit one answer's
history and cease providing genuinely diverse evidence.
**Finding:** completed Qwen family snapshot has **6,720 rows**; saved family
status is `AWAITING_SECOND_INDEPENDENT_MODEL_FAMILY`, not a formal pass.
Role reuse and style-control confounds were identified in the sealed design.
Mistral was stopped at the user's request so the corrected role-separated run
could start. **Decision:** developmental evidence only; neither result can make
a candidate go/no-go decision.
Evidence: `retrieved/semantic_ancestry_rag_g0_20260828T0901Z_qwen_family_snapshot_20260828T142825Z/`.
Do not mislabel this interrupted development stage as a failed replication.

### SA-0b — role-separated semantic ancestry (2026-08-29)

**Run:** serving Qwen/Mistral crossed with SmolLM3/Granite rewriter/shadow roles;
60 questions in each of four cells; style and independent-summary controls;
history-aware retrieval compared to MMR.

| Serving / rewriter / shadow | Cross-ancestry collapse minus baseline | History-aware mitigation minus MMR |
| --- | ---: | ---: |
| Qwen / Granite / SmolLM3 | +53.33 pp | +8.33 pp |
| Qwen / SmolLM3 / Granite | +28.33 pp | −3.33 pp |
| Mistral / Granite / SmolLM3 | +50.00 pp | −5.00 pp |
| Mistral / SmolLM3 / Granite | +21.67 pp | −10.00 pp |

**Finding:** collapse survived all four cells, but Mistral/SmolLM3/Granite failed
specificity and independent-summary checks. **All four cells missed the frozen
history-beats-MMR criterion**, including the positive +8.33 pp point estimate.
**Decision:** `KILL_SEMANTIC_ANCESTRY_CANDIDATE`; retain an interesting observation,
but do not expand this mechanism-plus-mitigation paper or retune its selector.
Evidence: `retrieved/semantic_ancestry_rag_g0b_20260829T0128Z_complete_20260829T1054Z/recovered_aggregate_20260829T1054Z/gate_report.json`.
The recovered aggregate and its verification are separate artifacts from the
original aggregate, not silent edits to family outputs.

### EC-0 — effect-consistency uncertainty (2026-08-29–30)

**Hypothesis:** agreement between executed successor states is a better
uncertainty signal than textual/action agreement and improves routing.
**Run:** Qwen3.5 and gpt-oss, calendar/inventory environments, alias controls.
**Finding:** **3 of 10** checks passed. Neither family provided the required
cross-domain discrimination-plus-voting improvement.
**Decision:** `KILL_EFFECT_CONSISTENCY`. Evidence:
`retrieved/effect_consistency_uq_g0_20260829T2225Z_retrieved_20260830T0134Z/effect_consistency_uq_g0_20260829T2225Z/GATE_REPORT.json`.

### VP-0 — visual patch-phase instability (2026-08-30)

**Hypothesis:** patch-grid phase specifically destabilizes thin visual primitives;
compute-matched phase ensembling mitigates the effect.
**Run:** Qwen3-VL-8B and Gemma4-12B.
**Finding:** Qwen thin/thick flip rates **16.67% / 16.67%**; Gemma **25% / 41.67%**.
Periodicity and specificity gates failed. Ensemble accuracy differences:
Qwen **−7.5 pp**, CI **[−17.5, 0]**; Gemma **−17.5 pp**, CI **[−30, −7.5]**.
**Decision:** `KILL_PATCH_PHASE_STUDY`. Instability alone did not validate its
mechanism, and the proposed remedy did not work. Evidence:
`retrieved/visual_patch_phase_g0_gemma4_20260829T2300Z_retrieved_20260830T0210Z/visual_patch_phase_g0_gemma4_20260829T2300Z/GATE_REPORT.json`.

### FL-0 — Critique Oracle feedback leakage (2026-08-30)

**Hypothesis:** diagnostic feedback enables policy gaming under pressure;
action-oriented feedback preserves legitimate recovery while reducing gaming.
**Run:** Qwen/Gemma, 384 trials each; deployment/export/refund; neutral/coercive
conditions; four feedback types.
**Finding:** fabricated authorization evidence **0 in every cell**. Actionable
versus opaque safe-success gains (pooled pressure conditions) were **+35.42 pp**
Qwen, **+45.83 pp** Gemma. Under coercion alone: Qwen **29.17% → 100%** and Gemma
**0% → 91.67%**. These are useful recovery results, not evidence of a leakage
tradeoff. **Decision:** `KILL_FEEDBACK_LEAKAGE`.
Evidence: `retrieved/feedback_leakage_g0_20260830T212423Z_retrieved_20260830T225635Z/verification_report.json`.

### EW-DEV — reward-hack early-warning screen (2026-08-30, CPU)

**Hypothesis:** ecological variance/polarization indicators anticipate hacking
before ordinary observable hack rates do.
**Run:** one public Qwen3-4B training trajectory, **10,240 rollouts / 40 checkpoints**;
sustained onset at step 120 under the frozen rule.
**Finding:** candidate and simple baselines both AUROC **1.0**; advantage **0**
vs required 0.15; false alarms **26.7%**; **0/9** robustness cells passed.
**Decision:** `DEVELOPMENTAL_FAIL`. No new RL training justified.
Evidence: [result](REWARD_HACK_EARLY_WARNING_DEV_RESULT.md),
`artifacts/reward_hack_early_warning_dev_20260830/REPORT.json`.

### PR-0 — Phantom Rollback (2026-09-01)

**Hypothesis:** ambiguous local checkpoint availability makes an agent take
irreversible external actions; precise scope metadata removes that excess.
**Run:** Qwen3.5/Gemma4, 1,024 trajectories and 16 comprehension probes per family;
four arms and four domains; safe authorized twins.
**Finding:** availability contrast **0** in both families (pooled CI **[0,0]**).
Scope reduction pooled **0.5208 pp**, CI **[0,1.5625] pp**, versus required 8 pp.
Qwen comprehension **12/16 = 75%**, below 85%; Gemma strict comprehension
**0/16**, with Markdown-fenced responses rejected by the frozen parser.

**Correction to an earlier chat summary:** Qwen was **not** 0/16. Gemma's strict
0/16 is not proof that it understood zero semantic answers. No retrospective
parser change can turn this assay into a registered pass.
**Decision:** `INVALID_ASSAY_COMPREHENSION_OR_CAPABILITY`; do not expand this
assay, but do not claim to have disproved risk compensation in capable agents.
Evidence: `retrieved/phantom_rollback_g0_20260901T0930Z/VERIFIED_REPORT.json`.

### RED-0 — Reward Extinction Debt (2026-09-01)

**Hypothesis:** a formerly shortcut-rewarded, subsequently aligned model retains
faster reacquisition than a clean policy matched on current behavior;
reactivation counterconditioning reduces that debt.
**Run:** instruction-tuned Qwen3.5-9B with frozen backbone/rank-8 LoRA; exact
expected two-choice policy gradient and KL, not a general autonomous RL agent.
Seeds 2718/5779; 18,304 scores. Both training history and the required matching
condition matter; the use of an instruction-tuned model is not itself invalid.

**Finding:** shortcut induction succeeded, **+97.690 pp** over clean,
CI **[96.672,98.480] pp**. But subsequent alignment did **not** create the
required matched starting policies. Initial shortcut probabilities were clean
**2.310%**, ordinary-aligned **23.438%**, counterconditioned **46.875%**.
Only one of the four seed-by-alignment-arm DEV selections achieved matching.

Descriptive debt AUC difference: **−0.01736**, CI **[−0.09674,0.06206]**, not the
required +0.12. Both clean and ordinary-aligned crossed the reacquisition bar at
dose 16 (speed ratio **1**, not 2). Ordinary/counterconditioned utility losses
were **26.00 / 52.56 pp**. Only **1/4 contexts** and **1/2 seeds** had positive debt.

**Decision:** `INVALID_MODEL_ORGANISM_FORMATION`. These unmatched trajectories
do not identify hidden debt or establish its absence. Plateau-like scores need
an optimization/scoring diagnosis before any biological interpretation. No
more compute on this formulation without a new, successful formation pilot.
Evidence: `retrieved/reward_extinction_debt_g0_20260901T1100Z/` and separate report
directory `retrieved/reward_extinction_debt_g0_20260901T1100Z_VERIFIED_REPORT.json/VERIFIED_GATE_REPORT.json`.
The directory suffix `.json` is historical: it is a directory, not one file.
265 evidence files / 4,489,112,701 bytes were previously secured; adapters are
preserved. Manifest SHA-256:
`6e97d78f6e0505e5d0bb8e2875f0ba0789f7ecf48bded7e4c2190c1ce658f120`.

## Runtime attempts and non-results

- Original symbolic/SFT oracle corpora and bridge smoke runs: engineering
  calibration, not independent evidence for learned goals. Prepared variants and
  stale-before-freeze directories remain under `artifacts/`.
- Hybrid-memory v1.0 stopped before predictions due to an unbound paired cache
  variable; v1.1 corrected it. Both archived. See HM-0 result.
- Recency/J0 timestamped retries and snapshots under `retrieved/` are preserved.
  The verified final result documents above, not folder count, determine the
  number of scientific runs. Do not guess a cause for every retry from its name.
- Semantic ancestry initial runtime-failure root and Qwen snapshot are retained.
  The confounded long run and G0b are distinct stages, not interchangeable data.
- Visual patch-phase `0204Z` retrieval precedes the completed `0210Z` retrieval;
  it is not another independent replication.
- Phantom's first `0524Z` attempt failed before generation on metadata handling.
  The `0530Z` run completed Qwen; missing torchvision stopped Gemma before its
  generations. Installing the matching torchvision and using a fresh `0732Z`
  Gemma root was a runtime repair, not a changed scientific treatment.
- SENTRY had runtime/preflight work but no verified covert-transfer scientific
  gate in the recovered index. Its transparent-style setup was rejected as an
  inadequate assay for the proposed hidden-transfer claim.

## Prepared or screened, not empirically established

| Candidate | Status as of this update |
| --- | --- |
| Validator-monoculture / correlated security testing | Code/oracle preparation exists; no completed GPU result found. Generic cross-family verification has substantial prior work. |
| Visual hindsight leakage | Code and frozen proposal exist; no completed GPU result found. Keep available as an independent bounded pilot, not a result. |
| Efference-Pair | Previously protocol only. EP0 code, corpus and CPU testing added September 2; see next entry. Real-video G0 unrun. |
| Outcome-blind process verification; effect-only world models; verifier-aware transport; SENTRY | Scouting/scaffolding, not completed GPU evidence for their paper claims. Consult dated candidate/scout documents. |
| Mostik-inspired latent compatibility | September 2 LC0 now has a channel-validity runner; CUDA and real task accuracy untested. Update/repair study and published baseline reproduction not implemented. |
| Other literature-screened ideas | Listed in [thesis scout](ICLR_2027_THESIS_SCOUT.md). Literature rejection is not experimental falsification. |

## 2026-09-02 — EP0 apparatus development (no GPU)

The initial 28-pixel low-contrast object/1.5-pixel motion fixture failed 3 of 14
CPU tests: the optical-flow residual did not meet its numeric-error tolerance.
This caught two apparatus problems before any VLM scoring: weak texture at the
flow window scale and rounded fractional object motion inconsistent with the
constant-step oracle. Initial generated artifacts remain at
`artifacts/efference_pair_ep0_20260902_v1/`.

The new **v2** fixture uses a 40-pixel textured object and exactly 2-pixel steps;
tolerances are unchanged. This is pre-experiment apparatus development, not
post-result threshold tuning. The separate EP0 protocol explicitly limits its
scope to planar translation. It does not silently replace the 448-pixel,
real-video ACaM/MotionBench gate with easier synthetic data.

Final **v3** adds neutral, truthful time stamps so the extra-RGB layout cannot
be mistaken for a single uninterrupted chronological sequence. The same vector
checks pass. All 28 targeted CPU tests passed (17 new motion, 3 latent fixture,
8 existing visual-hindsight); there was **no GPU or LLM inference**. The 64-pair
latent fixture has a perfect executable join oracle by construction, which is
only a dataset/control check and not evidence for latent communication.

See [EP0 runbook](EFFERENCE_PAIR_EP0_RUNBOOK.md) for the frozen workload and
actual validation results. Any future GPU result must be appended here with
the evidence manifest and a new decision; do not replace this CPU entry.

## 2026-09-02 — Data acquisition and independent experiment queue

User asked whether video data actually existed and requested more prepared
experiments. Synthetic EP0 frames already existed; real-video bytes did not.
We acquired **12 MotionBench DEV MP4s (36,001,980 bytes)** at pinned revision
`f099db892172a015c489507c9abe56b036d960ef`; upstream LFS SHA-256 and local
manifests agree. Eleven decode fully; preview screening identifies ten live-action
and one cartoon. The other clip decodes 119/182 advertised frames despite a
matching upstream checksum. It is quarantined, not silently dropped/replaced.
The initial failed acquisition and completed diagnostic inventory are both kept.

The sample is for media/label audit, not held-out scoring. Category labels were
not sufficient to find pure camera-versus-object cases: cuts, blur, action
counts, animation and pose questions are present. ACaM inventory/card was checked
but its archives/test annotations were not downloaded. Full label review and a
deduplicated, source-separated natural-video protocol remain necessary.

Prepared independent visual-hindsight corpus: 48 pairs / 240 calls, exact prefix
identity and matched suffix checks pass. Corrected a stale launcher code digest
to match committed source; no scientific conditions or thresholds changed.

New LC0 code adds a six-arm contextual-prefix transport preflight to the prior
nonce-join fixture: text, norm-text, latent, counterfactual text/latent and no
message. First smoke is 48 receiver forwards and eight teacher-forced sender
prefills. It uses a StateBridge-style mathematical alignment, not the original
four-agent benchmark pipeline. Model pins, budget audit, saved prefix tensors,
raw records, immutable snapshots and read-only verifier are implemented.
No sender update or scientific latent-compatibility result exists. A random tiny
Qwen CPU integration test verifies tensor/generation contracts only.

Additional brainstorming rejected generic retry/idempotency and durable
authorization replay as too close to current primary papers before spending
GPU time. These are literature decisions, not new failed experiments.

See [data and executable queue](DATA_AND_EXPERIMENT_QUEUE_20260902.md) and its
manifest for exact prepared paths, hashes, models, workloads and launch limits.
No GPU instance was contacted or launched during this preparation.

Validation: **43 focused CPU tests passed** (EP0, original latent fixture, LC0,
MotionBench selection and visual hindsight), all four prepared data roots passed
read-only checksum checks, both shell launchers passed syntax checks, and changed
documentation links resolved. CPU integration environment: torch 2.11.0+cpu,
Transformers 5.6.2, huggingface-hub 1.11.0, NumPy 2.4.2, Pillow 12.1.1,
pytest 9.0.3. These versions do not replace the older frozen GPU environments.

## 2026-09-03 — User shortlist: mathematical audit and two bounded assays

Full rationale, proofs, source links and limitations:
[five-idea PI review](USER_IDEA_REVIEW_20260903.md). The official abstract/full
deadlines are September 18/25 AoE. This is preparation, not a new GPU result.

**Hindsight:** a rational binary construction gives identical immediate feedback
under any common randomized one-step logging policy (channel probabilities
7/40 and 19/25), despite opposite rankings under initial-preference utility.
This is an elementary independent-episode result, not a claimed new general
theorem. The source SDPO paper already models action-dependent follow-ups;
the proposed research must concern a separately defined anchor estimand.

The suggested self-fulfilling polarization result is not automatic. In an ideal
Bayesian symmetric channel, the exact logit gradient is +0.109177 at policy
probability 0.1 and -0.109177 at 0.9: it restores balance. A simple linear
population-feedback model likewise has eigenvalues 1 and 0.42, not unstable
polarization. These counterexamples narrow the claim, not kill all endogenous
feedback research or establish real SDPO safety.

Sparse-anchor CPU calibration uses 12,000 independent simulated people in each
of four regimes. Cross-fitted delayed-anchor ATE estimates are -0.0449, 0.0044,
0.4925 and 0.5065 for truths 0, 0, 0.5 and 0.5 respectively. This includes
sampling error and is not a trained-policy welfare result. Assumptions and
anchor-contamination sensitivity are explicit. Sealed audit:
`artifacts/interaction_sprint_theory_20260903_v2`, manifest SHA-256
`d3dbfb7b258a804aaf0755ae62bcf318cd761e9fe397fa8c19f8e476ca10e163`.
Earlier unsealed v1 output is preserved.

**UNDO:** executable register semantics require a fresh-register guard for
set/clear cancellation. Clear removes a value; it does not restore a stack.
The generic local-to-global Lipschitz/telescoping bound is conditional and can
be vacuous. It is not itself a theorem-level novelty claim; CCOPD and state
tracking are necessary baselines.

New prepared roots are `artifacts/endo_signal_20260903_v1` (64 smoke/512 full
forwards) and `artifacts/undo_algebra_20260903_v1` (48 smoke/768 full). Source
is under `src/interaction_sprint`. Both runners score a pinned Qwen3-4B model
without weight updates. They retain raw probability mass, token budgets,
counterfactual controls, private keys, manifests and read-only analysis.
Smoke cannot yield a paper-go decision. Full SDPO/Anchor-SDPO and algebraic
local-relation training are **not yet implemented**; a positive first assay
would justify designing that next stage. CUDA compatibility is untested.

Transport is parked pending a target-linked inequality: exact inverse and
correct marginal distribution cannot identify a particular coupling (Gaussian
2D rotation counterexample). Succinctness is parked until a precise learning
lower-bound formulation exists. The steering extension overlaps the cited
paper's own identification assumptions. No GPU experiments for these three.

Updated queue: `configs/research_queue_20260903.json`; old queue retained.
No GPU connection, model training, expansion, remote push or human-data download
occurred in this preparation. Talk2AI is a publication-level data lead only.

The actual pinned Qwen tokenizer was checked locally (tokenizer files only, no
model weights): A/B/C/D map to single IDs 32/33/34/35. All 512 hindsight inputs
fit 37–75 tokens (27,136 total); all 768 UNDO inputs fit 105–3,022 tokens
(847,648 total). The runner performs this audit before allocating model weights.

Validation: **69 focused CPU tests passed** (26 new interaction-sprint tests and
43 existing motion/latent/video tests). All seven queue/data/audit roots passed
read-only integrity checks; the new remote launcher passed Bash syntax checking
and `git diff --check` passed. GPU model loading and the new inference runner
remain untested on hardware; neither test counts nor checksums are scientific
evidence for either proposed paper.

## 2026-09-03 PDT / 2026-09-04 UTC — GH200 text smoke assays completed

User supplied `192.222.57.245` and requested estimates. Transferred source
`cb1bac1` and verified both prepared text corpora. Created an isolated environment
using server CUDA torch 2.7.0; corrected old inherited SciPy/Jinja2/Pillow inside
that venv. Initial pre-inference failure is preserved; retries use fresh roots
and unchanged scientific inputs. No historical environment was changed.

Hindsight: 64 forwards, 2.313 seconds summed synchronized forward time; anchor
and truthful-correction comprehension essentially perfect. Wrong-action hindsight
advantage 1.2500 nats; anchor restoration 6.724 percentage points on the conflict
subset. Preliminary weak correction signal, not a full gate or SDPO result.
UNDO: 48 forwards, 1.832 seconds summed forward time; all arms correct at depth
4, no stale choices. Long histories not tested. Both decisions `SMOKE_ONLY`.
No full assay, training run or expansion followed; GPU idle after completion.

Complete evidence, failed attempt, logs and environment freeze retrieved to
`retrieved/interaction_smokes_20260904T0340Z`. Remote/local archive SHA-256 agrees;
committed local verifiers pass. See [timing note](GH200_TIMING_20260904.md).
These are two new apparatus smokes, separate from the 15 historical stages—not
two accepted/rejected papers. Setup and loading are excluded from forward times.

## 2026-09-04 — Full interaction assays and fresh-history robustness

User authorized running beyond the smokes. Frozen full assays used source
`cb1bac1`; complete evidence under `retrieved/interaction_full_20260904T0348Z`
passes remote/local archive SHA-256 and committed read-only verification.

Hindsight (512 forwards): comprehension prerequisites pass, but simple anchor
restoration averages only 0.003142 percentage points, failing the joint signal/
correction criterion. Ordinary wrong-action hindsight advantage remains positive
at 2.7653 nats. Decision `NO_SIGNAL_IN_THIS_PROMPT_ASSAY_DO_NOT_INFER_THESIS_FALSE`.
No SDPO training or preference-transition inference was performed. Overlapping
smoke probability records reproduce exactly; new cases explain the mean change.

UNDO (768 forwards): edited-history accuracy 100%, 97.92%, 68.75%, 66.67% at
4/20/60/100 updates; canonical and padding controls remain 100%. Counterfactual
accuracy is 97.92% at 60 and 100% elsewhere. The initial full gate shows a
residue signal. This is not yet a novel method or paper green light.

Implemented, tested, froze and ran a separate 864-forward developmental audit
at source `18e6207`: fresh wording, field names/noise, update positions, reminder
and explicit-update baselines. Prepared-manifest pin and complete evidence verify
locally at `retrieved/undo_audit_20260904T0354Z`. At long depths history scores
87/96 versus padding 96/96, a 9.375-point gap below the frozen 10-point criterion.
Reminder also scores 87/96; explicit update 91/96. Formal decision
`NO_FRESH_HISTORY_SIGNAL_PARK`, interpreted as insufficient registered robustness,
not zero effect. The cutoff misses by one case, so do not exaggerate it into a
universal negative or change the cutoff. Multiple apparatus factors changed;
their individual effects remain unidentified.

Further literature checking identifies ICF-Bench (ICLR 2026) as direct related
work for instruction forgetting and subtask revision, alongside CCOPD. The
remaining contribution must exceed a benchmark of stale-information errors.
The conditional queue now records a factorial diagnostic to design, then
independent-family replication and baseline-matched training only if justified.
These later stages are not implemented/launched. See
[full results, sources and queue](INTERACTION_FOLLOWUP_20260904.md).

Total this turn: 2,144 additional Qwen3-4B forwards, 31 passing targeted CPU
tests, all results checksum-secured locally, no weight updates, GPU idle after
completion. All roots/logs/private keys and previous attempts are preserved.

## 2026-09-04 — Independent queue resumed after idle scheduling gap

The independent motion, native-video hindsight and latent-channel candidates
were not blocked by the text candidate outcomes. Leaving them waiting was a
scheduling error. LC0 has now completed; complete evidence is locally secured
with matching remote/local SHA-256 and the committed verifier.

LC0 smoke: four pairs, 48 receiver calls; text/normalized-text/latent each 12.5%
accuracy, no-message 37.5%, parse rate 68.75%. Communication controls fail;
formal decision `SMOKE_ONLY_NO_THESIS_DECISION`. No weights were updated. This
invalid interface cannot judge latent communication or update compatibility.

Created an isolated CUDA vision environment without changing the text venv;
transferred and checksum-validated the EP0 corpus. Launched its frozen 24-call
smoke. Native-video hindsight is the independent next candidate after evidence
preservation. See [run roots, hashes and operational queue](INDEPENDENT_QUEUE_20260904.md).

EP0 subsequently completed: all 24 answers parse, but native, joint and oracle
each score 25% on both strata. Complete evidence and 3,615-file manifest verify
locally; archive SHA-256 matches remote. Formal smoke-only result, unusable
oracle prerequisite; no full-pilot expansion. Median forward 0.13033 seconds.
After verification, launched the independent 240-call native-video hindsight
run at `/home/ubuntu/visual_hindsight_g0_20260904Tqueue`, preserving all EP0 data.

## 2026-09-04 — Video gate complete; diagnose before interpreting failed controls

The 240-call native-video hindsight gate completed. All readability and format
prerequisites pass, with 100% prefix-past/future accuracy and zero endpoint-
following or paired-assignment effects in 48 pairs. Committed decision:
`KILL_VISUAL_HINDSIGHT_HYPOTHESIS`. Stop this frozen candidate, without claiming
all models or harder natural videos are immune. Evidence retrieved to
`retrieved/visual_hindsight_20260904T0441Z`; remote/local archive hashes,
3,081 outer-manifest hashes and the committed verifier pass. No active model
process remained. All three independent candidate runs are now secured.

Added a read-only post-hoc interface diagnostic and passing parser unit test.
EP0's 24 predictions all decode to stationary; LC0 leading-letter extraction
only recovers text to 3/8, equal to no-message. Neither issue is rescued by
lenient scoring. Original reports and thresholds remain unchanged.

Literature refresh also found [CREDIT](https://arxiv.org/abs/2605.11613), whose
posterior-compatible self-distillation analysis already derives pointwise
mutual information and an input-contrastive correction. We must not promote
our earlier mutual-information calculation into a novel theorem. New adjacent
camera work [CamChoreo/CamDistill](https://arxiv.org/abs/2608.10932) covers
temporally grounded compositional camera-motion recognition and geometric
distillation. Both tighten the novelty boundary; neither justifies relabeling
the failed controls as a positive paper result.

Started a separately labeled 24-call LC0 text diagnosis: identical token IDs vs
embeddings without reasoning, then token IDs with reasoning/256-token budget,
eight existing DEV cases. The runner never reads answers, pins the model and
records all generated tokens, input IDs, truncations and source. Two CPU parser
tests pass. Reasoning and budget change together in the third arm; no separate
causal attribution to either is allowed. No latent prefixes or training are used.

## 2026-09-04 — Text interface diagnosis verified; fresh channel test frozen

The 24-call diagnostic is complete and checksum-secured locally. All eight
token-ID and embedding generations are identical, reproducing old text outputs
exactly. Both arms score 1/8. Reasoning scores 5/8, with the other three runs
unfinished at the 256-token cap. This rules out the tested embedding-generation
API path as the cause and supports a separately specified reasoning-budget test.

Froze and launched 96 calls on previously unused pairs 4–11, original six channel
controls, reasoning on, fixed 512-token cap. No labels in runner, no updates,
strict final extraction, no automatic further budget increase. This is DEV
apparatus repair, not a replacement gate or paper validation. See
[design](LC0_REASONING_DEVELOPMENT.md) and [live queue](INDEPENDENT_QUEUE_20260904.md).

Staged official [C2C](https://github.com/thu-nics/C2C) code for read-only review at
commit `113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a` under ignored
`artifacts/c2c_upstream_audit_20260904`. It pins torch 2.6.0/Transformers 4.52.4;
never install it into the active environments. Upstream LICENSE is Apache-2.0
while package metadata says MIT; preserve upstream notices and resolve before
redistributing adaptations. No upstream code executed and no fuser weights
downloaded yet. Small published-model reproduction would be a baseline, not a
claim about frontier-model communication.

## 2026-09-04 — C2C release provenance and conditional update design

While the 96-call reasoning DEV process was verified live, audited the released
C2C model metadata without loading weights. Evidence is preserved in
`artifacts/c2c_release_metadata_20260904T0450Z`: pinned fuser revision
`f01fc3258b305e280e04c7238f4f2cf31b7dc70d`, 28 projectors, 59 selected files,
1,058,841,119 bytes. Selected sender Qwen3-4B matches the cached snapshot;
receiver Qwen3-0.6B has 28 layers, and the release maps sender layers 8–35 into
receiver layers 0–27. The original config lacks model revisions, so this is
not a claim of exact original-paper numerical reproduction.

Added metadata-only preflight script and a
[conditional update-study plan](LATENT_UPDATE_BASELINE_PLAN.md). The plan
requires useful independently selected updates, retained sender performance,
paired text/C2C comparisons, separate calibration/evaluation data, and strong
alignment plus version-check/text-fallback baselines. A broken adapter alone is
not a strong contribution. Existing compatible-representation learning further
limits novelty. No training or new GPU job was launched during this audit.

## 2026-09-04 — Published-task DEV data and strict projector load verified

Prepared 128 source-pinned validation examples (64 OpenBookQA, 64 ARC-Challenge)
with label-independent hashed selection and separated answer key. No TEST split
accessed. Eight selection/inventory tests pass. Cases digest
`a279178264c7b2e66c65d852193723925b42482d532ef6dc568a5bf3d0ce046b`.
These are ordinary public validation examples, not novel benchmark data.

Loaded released C2C projector 0 strictly on local CPU after matching its public
LFS digest. Synthetic-tensor forward shapes, finiteness and eval determinism
pass. This removes a checkpoint-schema concern for one module but does not
validate full cache fusion, the ARM64 dependency environment, or task accuracy.
The active reasoning DEV GPU process continued unchanged (53/96 calls at the
last process check); no outcome peeking or adjustment was performed.

## 2026-09-04 — Baseline inference implementation and isolated setup

Created `.venv-c2c-baseline-v1` on the GH200 without altering either active
environment. Transformers 4.52.4 cache API and dependency checks pass; torch
2.7.1+cu128 is an explicit ARM64 compatibility deviation from upstream 2.6.0.
Setup evidence `/home/ubuntu/c2c_baseline_setup_20260904_v1` is preserved.
The setup launch's PID file contains a literal `$!` due to quoting; its actual
process was confirmed exited and successful via the readiness marker, import
check and dependency log. Do not use that malformed PID file as a live handle.

Download-only process 14631 completed and checksum-verified seven receiver
assets and 58 fuser assets at pinned revisions. Evidence root:
`/home/ubuntu/c2c_assets_20260904_v1`. No model inference was performed by setup
or staging. Implemented and froze the 512-call four-arm baseline and read-only
verifier before inspecting any outcomes from that baseline. Fifteen CPU tests
cover data selection, inventory and analysis; GPU wrapper integration is still
untested. The active 96-call reasoning study remains unchanged.

## 2026-09-04 — Reasoning channel result secured; published baseline launched

Fresh reasoning DEV completed: 96 calls / eight independent pairs, generation
time 1,046.71 seconds. Retrieved complete evidence and prefixes to
`retrieved/lc0_reasoning_dev_20260904T0459Z`; remote/local archive digest
`95895efa41f9334a740857fb00935f3cea7a07c2b23ec64da6d43f782d5b47a8` matches.
The frozen verifier reports text/normalized/counterfactual-text target accuracy
100%, latent and counterfactual-latent target accuracy 93.75%, no-message 0%.
All 16 no-message and two latent responses are token-limited; global parse rate
81.25% fails the frozen validity prerequisite, so `INVALID_CHANNEL_ASSAY` stands.
Do not discard this requirement after seeing the data. Equally, do not call the
15/16 latent answers a negative communication result. This is a useful working
apparatus clue, not a model-update result or paper green light.

After process exit and verified local preservation, launched the already frozen
released-C2C baseline (512 calls; 128 public validation examples, four arms;
PID 15457). Upstream imports pass in the isolated environment. No update
training is launched. Follow-up design remains conditional on functional
controls and a meaningful baseline signal. See the live queue for exact roots.

## 2026-09-04 — Stronger nearby cache-transfer baselines located

Primary-source search located [cross-model cache transfer](https://arxiv.org/abs/2608.03893)
and the September 1 [CacheBridge paper](https://arxiv.org/abs/2609.00891).
These occupy closed-form unlabeled alignment and attention-sensitive repair;
do not claim either as a new corrective mechanism. CacheBridge explicitly
leaves open-ended multi-turn quality outside its evaluation, but that gap alone
does not green-light a new paper. Added the collision and interface distinction
to the conditional update plan. The frozen C2C baseline remains unchanged.

## 2026-09-04 — C2C completed; scoring mismatch diagnosed, not hidden

All 512 calls completed and checksum-verified locally at
`retrieved/c2c_baseline_dev_20260904_v1`, archive
`5ede9533332d891a0ba6626fb06b3783e40d342f1af4bdd223068a35239fd34f`.
Strict scoring failed because the small receiver commonly appended option
text, while our regex required the output to end after its letter. That was
our scoring-design error for a baseline reproduction. The original report and
criteria remain unchanged, and no generation was rerun to repair it.

A separately labeled post-hoc audit uses the inspected upstream parser from
the pinned source: receiver 51/128 (39.84%), sender 113/128 (88.28%), fused
72/128 (56.25%), disabled fuser 50/128. Conservative explicit-prefix scoring
gives fused 65/128 (50.78%), even treating other formats as wrong. Thus the
fused gain is not wholly explained by the upstream parser's loose fallback.
Twelve fused parser disagreements remain; manual inspection found mostly
letter-plus-option outputs and a few prose-only answers. Do not present every
upstream fallback as unambiguous. The exploratory paired bootstrap CIs are
[8.59,24.22]pp (upstream) and [2.34,19.53]pp (conservative), stratified by dataset.

Disabled-fuser text agreement is 99.22%. Sender generation is both more accurate
and faster in this setting (35.32s total versus fused 60.80s), so the baseline
does not establish communication's superiority over simply using the sender.
The mechanism helps the smaller receiver; this is an existing-method result,
not novelty or a deployment-update finding. No update training is justified
by this result alone. Next work is the text-transfer baseline and a controlled
natural-update protocol. All GPU jobs have exited and evidence is secured;
the completed-run heartbeat is paused while offline preparation continues.

## 2026-09-04 — Offline comparator and natural sender-update implementation

No GPU job was launched. The user has termination clearance, and the paid host
was confirmed idle before this work. The earlier wording of a "queue" conflated
executable jobs with conditional design entries; these are now explicitly
separated in the README and pilot protocol. No paper viability claim follows.

Implemented the prospective two-stage text comparator: 128 questions, 256
generations, pinned sender/receiver and upstream prompt structure. It records
the generated background and exact transfer conversation and charges both
generation stages. A fixed explicit-label scorer accepts letter-plus-option
answers without interpreting arbitrary prose. It will rescore the preserved
historical arms identically; the original strict report remains untouched.
Full verifier-path tests use synthetic T2T records against the real saved
baseline, clearly marked test fixtures, not model results. They catch duplicate
outputs, altered message transfers, bad caps, NaN timings and wrong datasets.

Prepared and independently reconstructed four disjoint public-data partitions:
1,024 update-training, 128 qualification, 128 repair-calibration, 256 final-eval.
The 128 old baseline questions are excluded by ID and normalized content.
Preparation manifest SHA-256:
`ad945cc2588809b6e94e43fdbd46f81159d8acc66dd3339621a0069c4ecd7af6`.
Read-only report `artifacts/c2c_update_data_audit_v1.json` confirms exact
reconstruction from saved Parquet rows and answer-key alignment. No model
evaluated any new partition; public pretraining contamination remains possible.

Implemented Stage A of the natural-update pilot, including ordinary attention
LoRA training, prompt/padding masking, independent fixed seeds, no-op save/load,
archived initial/middle/final adapters, merged model export, and sender-only
qualification. Selection depends on sender retention and choice cross-entropy,
never bridge degradation. Both seeds must qualify; there is no automatic
learning-rate/checkpoint sweep. See `docs/C2C_NATURAL_UPDATE_PILOT.md` for exact
recipe, limits and commands. The paired interface/repair Stage B is still
unimplemented and must be frozen before Stage A GPU launch.

Engineering evidence: 35 relevant tests pass on local torch 2.11.0+cpu /
Transformers 5.6.2. Seven training-core tests also pass in an isolated local
Transformers 4.52.4 environment, including a tiny native Qwen3 train, adapter
merge, safe-weight export and reload. Base weights remain frozen during adapter
training; accumulation matches full-batch gradients; all examples are covered.
An initial bit-exact output reload assertion exposed a 3.7e-8 float32 difference
from attention backend selection. The test now fixes the backend, checks saved
weights bit-exactly, and compares outputs at float32 numerical tolerance. This
is a test correction, not a changed scientific threshold.

JUnit reports: `artifacts/c2c_implementation_checks_v1.xml` and
`artifacts/c2c_tf452_cpu_checks_v1.xml`. These tests do not exercise GH200 CUDA
training or constitute an update-effect result. The preserved isolated CPU
environment is `artifacts/c2c_cpu_compat_v1`; older environments are unchanged.

## 2026-09-04 — Paired natural-update measurement implemented, not run

Added Stage B (`scripts/run_c2c_paired_update.py`) and its local verifier. For
each of 256 reserved public DEV questions it compares the original and merged
updated sender: sender alone, frozen C2C, disabled fuser, and two-stage text
transfer, plus a shared standalone receiver. This is 2,816 generations per
training seed, 5,632 total. Both predetermined seeds are required. No sender
has yet been trained and none of these model calls has run.

The local release-ticket builder first re-verifies the text comparator and both
Stage A archives. It refuses to release final DEV for an unqualified or missing
seed. The GPU runner checks the separately recorded ticket digest and every
merged-model file before loading the model. It never receives private keys.
The same frozen projectors/receiver are used across versions; ordering is
counterbalanced by case index, not outcomes.

Frozen descriptive analysis: old/new accuracy changes and the C2C-minus-text
difference in changes, paired question bootstrap stratified by dataset, and
separate seed reports. Sender retention, a usable original bridge, parsing and
disabled-fuser controls are necessary before interpreting extra latent loss.
Both seeds must show >=10pp extra loss with upper interval below zero to advance
to repair investigation. A failed seed cannot be pooled away. A positive result
is still not geometric causal identification or a paper go. Costs include both
text generations and the standalone sender comparison. Exact rules and caveats
are in `docs/C2C_PAIRED_UPDATE_PROTOCOL.md`.

Verification: 55 C2C tests pass in the main CPU environment, with one intentional
skip for the older wrapper API. Eight tests pass in the isolated Transformers
4.52.4 environment, including the previously skipped tiny native Qwen3/C2C
integration test. It executes both source models, switches old/new/old sender
membership, checks repeated-old generation equality, and verifies disabled
fusion equals the unchanged receiver before/after switching. Synthetic tests
reject changed checkpoint files, missing/failed seeds, corrupted transfer
messages, duplicates and invalid timings; they distinguish shared channel
degradation from the proposed extra latent loss. These are engineering tests,
not measured paper effects or a full GPU runtime validation.

Reports: `artifacts/c2c_paired_implementation_checks_v1.xml` and
`artifacts/c2c_paired_tf452_cpu_checks_v1.xml`. GPU launch remains on hold after
termination clearance. Repair comparisons are the remaining implementation
work before the full pilot is ready; no automated run or expansion was started.

## 2026-09-04 — Conditional repair baselines complete; no GPU experiment yet

Implemented Stage C: identity-wrapper control, head-local diagonal/ridge/
orthogonal cache alignment, and small projector-output-matching retuning. These
are explicitly existing baselines, not a new claimed method. Calibration uses
the frozen unlabeled 128 examples, split by a fixed ID hash into 48 fit and 16
check examples per dataset. It collects matched old/new/receiver post-RoPE cache
rows from the exact source-prefill prefix, with at most 32 positions per case.
The 384 backbone forwards, selected caches, fitted maps, retuned float32/BF16
weights, sample indices, loss curves and preparation costs are all preserved.

The conditional language comparison adds 1,280 generations per seed only after
both paired runs show actual bridge damage. Ridge is the declared primary
repair; all alternatives are reported without winner selection. Verification
rechecks the paired parents, hashes, source, cache splits and output grids and
independently reconstructs closed-form maps on CPU. For nonunique orthogonal
fits, it checks feasibility and objective optimality instead of demanding a
particular SVD basis. Retuning is not independently replayed; this limitation is
explicit. Successful existing repair still does not authorize a paper go.

Pre-run logical audit caught a false-positive possibility in the Stage B rule:
unchanged latent accuracy plus improved text accuracy could satisfy a negative
difference-in-differences threshold. Before any training/paired results, added
an explicit >=10pp absolute C2C decline requirement. A synthetic regression test
now rejects the counterexample. This is a prospective protocol correction, not
a post-outcome threshold change.

Tests: 75 pass with two intentional old-API skips in the main CPU environment;
24 pass under Transformers 4.52.4, including those compatibility checks. Tests
cover fresh-sample affine/rotation recovery, rank deficiency, nonunique-optimum
verification, teacher/gate preservation, actual C2C projector retuning, native
Qwen3 cache equivalence, and identity-repair generation through the real wrapper.
An initial cache-collector test exposed Transformers 5's non-subscriptable cache;
an explicit adapter handles that local test API while GPU runners remain strictly
pinned to 4.52.4. No dependency on the GPU host was changed.

JUnit: `artifacts/c2c_repair_implementation_checks_v1.xml` and
`artifacts/c2c_repair_tf452_cpu_checks_v1.xml`. The execution plan is
`configs/c2c_natural_update_queue.json`, explicitly marked as a plan, not a running
service. All stages now have code/protocols and CPU checks. Full CUDA runtime
validation, actual experiments, and empirical evidence remain outstanding.
GPU access must be reconfirmed after the previous termination clearance.

## 2026-09-04 — Existing-result utility audit and closer prior work

Verified the previous status request against the live host: GH200 idle, no
scientific process, all completed evidence already local. This was not an
active experimental queue. No subsequent GPU job was launched.

Primary-source reading of the final DroidSpeak paper confirms that its
motivation explicitly includes models updated over time, not only unrelated
specialists. PrefillShare section 3.2 directly treats compatibility failure
after fine-tuning and trains cache-compatible decoders. XKV addresses frozen
heterogeneous private-context communication. These do not prove our exact
frozen-fuser update test already exists, but substantially weaken its current
novelty argument. Sources and distinctions are recorded in
`docs/C2C_NOVELTY_UTILITY_REVIEW_20260904.md`.

Implemented a read-only paired utility audit, with five passing focused tests,
and ran it on the complete baseline after immutable-evidence re-verification.
Both parsers give four C2C-only correct answers; sender-only correct answers
number 45 under the published parser and 52 conservatively. A label-using
oracle union reaches 117/128, only 3.125pp over sender alone; this is explicitly
not an achieved selector. Recorded generation is 35.324s for sender versus
60.797s for fusion. The paired result is saved separately at
`artifacts/c2c_sender_utility_audit_v1.json`. Original reports are untouched.

PI action: park the current same-input natural-update/repair paper route on
novelty and utility grounds before training. Added a top-level hold to the
execution-plan JSON; preserved every stage, protocol and package. A different
private-information or long-context deployment could have value, but requires
its own rationale and comparisons and is not automatically authorized by this
finding. This is a literature/portfolio decision, not evidence that updates
are harmless. The broader paper-finding goal remains unmet.

## 2026-09-04 — Hindsight estimator audit yields a sharper learning question

Read SDPO section 3 and appendix B and inspected pinned released online/offline
losses. Distinguished own-response sampling under p(a)K(o|a) from a full-KL
update under p(a)M(o), with teacher and sampling weights stopped in both. An
exact two-action example gives opposite ascent directions at p=0.1: sampled
+0.1091770192, full KL -0.1265601357. Actual released loss methods executed on
mocked logits reproduce both; frozen-target finite differences check full KL.
The earlier balanced-attractor result remains correct for the sampled update,
not both losses. Earlier reports are preserved.

Reverified the 512-forward Hindsight archive and reused its 20 unique base and
feedback prompts. Under imposed action-copying channels, saved A/B distributions
yield opposing scalar-logit directions in 7/8 surface/wording contexts for copy
probabilities 0.5 through 0.9. Independent feedback gives zero mismatches. This
post-hoc calculation is not LM training or measured real-user behavior. Shared
parameter gradients, learning trajectories and utility effects remain unknown.
Reports: `artifacts/feedback_gradient_audit_v1.json` and
`artifacts/feedback_gradient_audit_v2.json`.

Close prior work, Privileged Likelihood Is Not Automatically Value, already
discusses own-rollout feedback, cross-fitting and partial/upstream gradients.
The mismatch identity alone is not a novel method. Next is a tightly controlled
learning diagnostic design separating estimator fidelity from utility, not
another anchor-prompt gate. See `docs/HINDSIGHT_SAMPLING_LAW_AUDIT_20260904.md`.
No GPU work was launched. The paper-finding goal remains unmet.

## 2026-09-04 — Actual pretrained CPU parameter probe completed

Implemented and froze a restricted-A/B, norm-matched one-step comparison on
public Qwen3-0.6B, with rank-4 attention LoRA and four CPU threads. Four task
types in two option orders form eight training contexts; four analogous fresh
scenarios form eight heldout contexts. The designed feedback channel mixes
truthful original-preference messages with action-copying expression. This
does not model or measure real persuasion. Paid GPU use remained zero.

The first launch failed before scientific forwards because Transformers 5's
chat-template default returned a BatchEncoding. Explicit `return_dict=False`
fixed token formatting without changing prompts or science. Failed root
`artifacts/hindsight_parameter_probe_cpu_v1` remains intact. Fresh retry at
commit `085479e`, root `artifacts/hindsight_parameter_probe_cpu_v1_retry1`,
completed 66 forwards, eight backwards and both one-step updates in 26.78s.

Result: primary rho=.5 parameter-gradient cosine is 0.999156 (float64 replay);
rho=.9 gives 0.995805; the independent-feedback null matches exactly. Both
updates improve heldout accuracy from 6/8 to 7/8 and train accuracy from 4/8
to 7/8. Heldout NLL is .724127 before, .359969 after own-response scoring and
.356621 after full KL. These are near-identical beneficial one-step results,
not the opposite parameter directions suggested by extrapolating the earlier
logit example. No substantive learning-level split is demonstrated here.

Independent read-only verification checks checksums, cases, metric arithmetic,
stored directions and actual adapter deltas; it does not rerun the model.
Float32 cosine/norm reductions were slightly inaccurate; the separate verifier
recomputes in float64 and reports actual .1000033 step norms. Original artifacts
remain unchanged. Report `artifacts/hindsight_parameter_probe_cpu_v1_verified.json`;
manifest SHA `72e2b3422393fa9d8ecc8b39a4460b63a011fb369ad930957f561388e535c7d6`.
All adapters, directions, predictions, prompts and source remain on this laptop.

45 relevant tests pass. Limits: tiny handcrafted split, low starting task
accuracy, only one normalized step, restricted vocabulary, no teacher mass
record, and changed model/prompts/channel relative to the earlier saved-score
study. Do not attribute the difference to any one changed factor. PI decision:
no expensive expansion yet; isolate those factors before making a practical
claim from the exact estimator mismatch. The paper-finding goal remains unmet.
