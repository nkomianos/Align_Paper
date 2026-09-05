# Independent audit: corrections, paper decision, and next experiment

**Audit date:** 5 September 2026 UTC / 4 September Pacific.
**Audited starting checkout:** `1cfe22124955dee56c8683cbf0ea0f51137e146a`,
`codex/iclr-2027-thesis-scout`.

**Recommendation: pivot Hindsight specifically toward persistence measurement;
do not launch the existing GPU queue unchanged. No current candidate is
submission-ready.** Several mathematically correct and useful developmental
results survive, but the current neural endpoints, simple baselines and external
measurement bridge do not support the proposed causal paper.

The complete handover was read first. The audit then inspected code, raw records,
protocol chronology, manifests, source bindings, saved statistical outputs and
primary literature. Three parallel independent subaudits covered theory/prior
art, historical experiments, and Hindsight data; the primary audit separately
checked neural code, decision functions, data units, repository/bundle integrity,
local runtime receipts and manuscript readiness. No GPU host was contacted,
no model experiment was launched, no private token was used, and no current
reserved confirmation records were opened by this audit. Existing read-only
verifiers and seeded computation replays are audit work, not new experiments.

## 1. Correction and audit ledger

The detailed claim-level ledgers are in the rebuilt
[evidence index](RESEARCH_EVIDENCE_INDEX_20260905.json) and the linked subaudits.
They include `claim`, `evidence_root`, `protocol_commit`, `unit_of_analysis`,
`verifier_result`, exclusive `classification`, `error_or_confound`,
`corrected_conclusion`, and `next_action`. A verified artifact can still be an
invalid assay: checksums and internally consistent arithmetic do not establish
correct targets, measurements or conclusions.

| Correction | What the independent evidence says | Decision |
|---|---|---|
| Deterministic minimax claim | `.2` applies to a fixed population action. A deterministic rule using random historical data attains worst expected regret `.16`, contradicting the literal universal `.2` statement. General expected-risk lower bound `2/15` survives. | Correct the theorem statement; retain the restricted result. |
| Deployment information | Immediate mechanism nonidentification survives observed historical baseline state. But if the current user's baseline state is available at deployment, `A=Z0` succeeds in both witness worlds. | Remove general unavoidable-personalization-regret language. |
| Theoretical/empirical objective mismatch | The theorem evaluates `P(a=Z1(a))`; G1 mimics a trained oracle and EndoPAHF predicts fixed old/new labels. Those are different estimands. | Declare each explicitly; no causal utility bridge has been established. |
| Robustness baseline omission | Ordinary equal-anchor regret `.00380385` beats advertised augmented-IPW `.00417692` under clean random sampling. At strongest selection the corresponding regrets are `.00000769` and `.00218077`. | IPW bias correction survives; broad policy superiority does not. |
| Earlier finite-state correction | At 16 anchors per action, ordinary-anchor `.01377017` versus clipped augmentation `.00934317` survives, raw `.08628067`. | Positive developmental method screen; clipping precludes a blanket exact-unbiasedness claim. |
| Under Extinction | Comprehension `.3906` versus `.90` prerequisite; failed reversal. | Invalid controller assay/capability, not a qualified scientific negative. Preserve operational stop. |
| Broad Hindsight prompt test | Its frozen language/anchor prerequisites pass; restoration is only `.00314161` percentage points. | Valid negative for that prompt-restoration endpoint, not invalid comprehension and not disproof of causal learning. |
| UNDO | Local loses four long-history pairs to canonical; exact two-sided paired sign/McNemar `p=.125`. Protocol calls the one-seed final cases developmental. | Preserve descriptive negative; remove confirmatory/significant-inferiority implication. |
| C2C | Original strict parser missed the gate; repaired upstream/conservative scores are useful retrospective diagnostics. Fused still loses to sender. | Developmental utility comparison, not a prospectively qualified positive. |
| PUPPET target bug | v1 saved all 72 targets as `+7`, not all zero. Current numeric verifier passes v1 and repaired v2. | v1 invalid; v2 weak-reader developmental evidence, with source-target binding still requiring a dedicated check. |
| Teacher versus parser | Hindsight full responses have 44/64 first-token correctness, 55/64 unique correct label mentions, and 25/64 strict full-response A/B correctness. SDPO v3 loses preservation despite semantic content retention. | Separate format, semantics, acquisition and scientific endpoint. |
| Generation configuration | The single-profile run executed sampling despite an intended greedy configuration; the passed-config receipt alone passes. | Invalid intended assay; preserve as-executed diagnostic, no SDPO training conclusion. |
| G1 endpoint | Distance to trained-oracle probability is not monotone in persistent correctness. Position-gap averaging can hide complete per-task label-swap reversal. | Repair/supersede before endpoints; existing tests passing does not validate scientific measurement. |
| G1 generalization and panels | 128 rows, 32 unique prompts, 16 reused domains; all evaluation text matches after removing the prefix. G0 panels overlap; G1 panels share one model seed/population. | No independent domain transfer or eight-seed replication. |
| EndoPAHF cohorts | 630 learning bases across 20 users; all 19 DEV users occur in learning. Rotations must be clustered; user dependence remains. | Held-out tasks within the observed cohort, not unseen-user transfer. |
| Scheduled label balance | Four-rotation input balance does not imply balance in the actual 630-row schedule: old labels A156/B142/C167/D165. | Report actual exposures; mild imbalance is not a between-arm confound by itself. |
| Power claims | G1 is a deterministic nine-cell surrogate. G2's simulated NLL threshold is not the full joint NLL/accuracy/multiple-comparison rule. | No claim of neural statistical power or full-study false-positive control. |
| Published baseline fairness | Prepared raw/FIX/SPEC SLIFT sees immediate feedback; proposed correction additionally sees delayed labels. G2 also compares panel ensembles with single controls. | Passive identification comparison only; require pooled same-label and matched-compute baselines for method superiority. |
| Transition sanity | G2 literally subtracts a tensor from itself and repeats raw training. | CPU identity check; not separate empirical transition validation or useful GPU science. |
| Locks | Some deterministic preparation/verifier paths read reserved content while reporting no confirmation evaluation. Reward Debt's historical TEST was used for induction/evaluation before matching qualified. | Distinguish content access, model evaluation and selection; document exceptions. No current confirmation endpoint was found. |
| Reproducibility | Recency/Recipe runners discard per-case endpoint scores. Some verifiers merely recompute decisions from saved aggregates. Historical Windows line endings can break source hashes. | Raw recomputation is unavailable for those scores without new forwards; do not claim universal re-verification. |
| Early-root census | Native decoder qualification actually ran 16 completions/124 CPU forwards but strict correctness was Qwen 1/8, Smol 0/8. Twenty-four bridge/oracle files collapse to five constructed probability payloads. | Add one invalid capability assay; never count oracle fixtures or copied roots as learned-model positives. |
| Offline agent monitoring | Public traces, calibrated selection, ScopeJudge expert votes and BeyondMasks source/video checks are reanalyses or apparatus. Calibrated v1 fails current source replay while its cells equal v2. | Preserve measurements and the failed receipt; no new model-monitor, intervention or editor result. |
| Prior-art narrowing | PUMA learns inferred state dynamics; SLIFT already decomposes feedback. Privileged Likelihood and human preference-expression intervention work also directly collide. | A generic causal-personalization or gradient-validity fallback is not novel. |
| Runtime/source state | GitHub branch remains at `6d8c61b`; local audited HEAD is 196 commits ahead. Latest handover/SLIFT already committed. SSH timeout never proved that paid processes ended. | Preserve bundles; no current host/billing status inferred. |

The detailed sources and qualification limits are recorded in
[theory/literature](AUDIT_THEORY_LITERATURE_20260905.md),
[neural queue](AUDIT_NEURAL_QUEUE_20260905.md),
[historical portfolio](AUDIT_PORTFOLIO_20260905.md), and
[Hindsight data](AUDIT_HINDSIGHT_DATA_20260905.md), with
[early-root](AUDIT_PORTFOLIO_EARLY_SUPPLEMENT_20260905.md) and
[offline-monitor](AUDIT_OFFLINE_MONITORS_SUPPLEMENT_20260905.md) supplements.

### Census and evidence limits

The filesystem census contains **34,587 files in 3,197 directories**,
with 211 root groups. Filename-role counts include 159 protocol/runbook files,
336 result/summary files, 357 manifest/receipt files, 609 checkpoint-related files,
and 15 Git bundles. These are **file counts, not experiment counts**; source
snapshots, model caches, duplicate retrievals, checkpoints and rotations cannot
be counted as independent runs. User-owned `analysis/`, environment caches and
the audit's artifact outputs were excluded; new audit documents are included.
The first inventory accidentally excluded nested directories also named
`analysis`; the preserved initial receipt is superseded by this corrected census,
which excludes only the user-owned root `analysis/` and includes retrieved analyses.

The final index has **144 claim/status records**, with the following units kept
separate. Its **64 logical measured assay versions contain zero qualified
scientific positives, 11 scoped valid negatives, 21 invalid assays and 32
developmental checks**. Numerical improvements within developmental checks are
not thereby denied; they are not qualified scientific confirmations.

| Record type | Count | Classification |
|---|---:|---|
| Logical measured assay versions | 64 | 0 valid positive; 11 valid negative; 21 invalid; 32 developmental |
| Formal claims | 3 | Valid positive, partly dependent exact results |
| Finite-model longitudinal mechanism test | 1 | Scoped valid negative |
| Preparation / computation versions | 47 | Developmental |
| Post-hoc reanalysis groups | 7 | Developmental; six reuse released data, one reanalyzes C2C |
| Superseded calibrated artifact | 1 | Developmental duplicate; never an extra study |
| Explicit Hindsight pre-inference aborts | 5 | Invalid launch attempts, no measured endpoint |
| Known never-run protocol groups | 16 | Not run |

These are not physical process or billing counts. The separate
[attempt register](../artifacts/independent_audit_20260905/portfolio/excluded_attempts.json)
also preserves historical partial/asset failures and duplicate retrievals; some
partial failures consumed forwards. Never-run entries group known stages, not
every unexecuted script. The proposed new experiment is not historical evidence.
Each row states its unit and evidence limit; unresolved protocol commits are
explicitly identified. Mathematical positives are not empirical neural positives.

All 15 existing bundles pass `git bundle verify`. The handover's `d890e05` hash
matches exactly. All ten present adjacent deployment sidecars match; two older
queue bundles have no adjacent sidecar. The later `iclr_handover_1cfe221.bundle`
also verifies and hashes to
`7e03e33489e471c212c0ae5a00966fecb727b3890ec3fa1d255c1efddf94a52a`.

Local runtime audit finds 40 relevant log paths but only 32 distinct contents,
39 runtime receipts, and 15 exit receipts (11 zero, 4 nonzero); eight log paths
contain tracebacks. It confirms failed launches/retries around environment,
interface and SDPO setup as well as successful retrieved runs. Copied logs are
not extra executions. No continuous utilization/billing trace permits a reliable
total GPU-hours or idle-time reconstruction. Nor can this local audit prove that
no unrecorded remote output existed. Report that gap instead of inventing hours.

The audit could reconstruct deterministic effect-consistency and feedback-leakage
private keys byte-for-byte against their original hash commitments, closing those
verifier prerequisites without changing original evidence. It could not recover
discarded Recency/Recipe raw probabilities. It deliberately did not run verifiers
that would parse current reserved confirmation during preparation replay.
Model weights and historical neural trajectories were not rerun.

## 2. Conclusions that survive or change

Surviving positives are the restricted exact nonidentification, policy-ranking
witness, `2/15` minimax result and known-probe matrix identification; the latter
is standard linear identification. Sparse residual correction beats ordinary
anchors in the first finite-state experiment and at random selection with its
unweighted version in the robustness grid. Those simulations are developmental,
not neural evidence or prevalence estimates.

The operational decisions to park the failed historical candidates mostly
survive. Their reasons need precision: some qualified endpoints are negative;
others never had a valid scientific assay. Under Extinction, DID, Phantom,
Reward Debt, failed teacher/reader interfaces and video-oracle failures cannot
falsify the underlying hypothesis. UNDO and C2C remain unpromising descriptive
method comparisons. The symmetric drift calculation and natural-initialization
LoRA negative still oppose the automatic-polarization story.

No capable Hindsight gradient/policy result, successful natural-surface neural
transfer, fair SLIFT comparison, second-family replication or independently
validated human persistence measurement exists. PUPPET/PAHF cannot be relabeled
as evidence that assistant responses caused real persistent user change.

## 3. Strongest thesis and reviewer objection

The best remaining thesis is **persistence-specific identifiability and the
value of independently validated delayed measurements for learning**, under an
explicit objective and equal information budgets. It is a research thesis with
some formal support, not a claim that the empirical conjunction is established.

The strongest objection is: *The worlds and correct delayed labels were engineered,
and the correction is standard. Why is this more than supervised learning with
better labels?* A scalar gradient cosine, extra labels withheld from a baseline,
or a natural-language wrapper does not answer it.
The [one-page pitch](HINDSIGHT_PAPER_PITCH_AFTER_AUDIT_20260905.md) and
[rejection memo](HINDSIGHT_REJECTION_MEMO_AFTER_AUDIT_20260905.md) spell out the
intended contribution and the evidence that could change this assessment.

The collision comparison must include
[SLIFT](https://arxiv.org/html/2608.09109v1),
[PUMA](https://arxiv.org/html/2605.24647v1),
[Privileged Likelihood](https://arxiv.org/html/2608.09263v1), and
[Influencing Humans to Conform to Preference Models](https://arxiv.org/html/2501.06416v3),
alongside SDPO, ThoughtTrace and standard causal/PPI estimators. The remaining
distinction requires a credible persistence measurement and a consequential
learning result; a generic measurement paper is not a safe novelty fallback.

## 4. Continue, pivot, or kill Hindsight?

**Pivot.** Keep the persistence-identification question, kill unsupported broad
claims and the current queue's interpretation. Do not start another portfolio
search. Give the narrower path one bounded decisive method screen and a hard
deadline for external measurement feasibility. If that route is not real, choose
a later venue; do not infer paper viability from theorem/apparatus successes.

## 5. Smallest decisive next experiment and queue

The smallest useful neural decision is a **reduced direct EndoPAHF DEV policy
comparison**, prospectively superseding the old G0/G1 path. Use all 630 learning
bases, the same pooled 64 delayed-anchor bases for every sparse learner, and
96 DEV bases with rotations averaged within task. Train six single-model arms:
raw, full-delayed oracle, pooled SFT, pooled SDPO, residual correction and a
fixed nonnegative immediate/anchor mixture. Qwen3.5-9B, one seed, 35 updates each:
**210 updates** on the full pass path. Score full-vocabulary delayed-label NLL
and correct native answer-token probability (other tokens score zero), not
distance to the trained oracle; report normalized-choice metrics separately.

The exact interface must qualify first. Raw and full-oracle acquisition must
qualify before sparse arms. Proposed method gate: at least `.03` lower NLL and
`.02` higher expected correct-choice probability against every named simple
neural comparator, plus `.01` higher A/B/C/D-conditional semantic correctness,
with nonnegative rotation and leave-one-user-out gains for both probability
metrics. The conditional gate prevents formatting-only improvements from passing.
Residual correction must also beat each prespecified cheap profile/memory
readout by more than `.005` on both probability metrics; a matching or better
readout stops the neural-superiority path. These are practical routing thresholds,
not statistical equivalence or power claims.
Serialize and check this complete rule before launch; no current power simulation
validates it. Qualified failure parks this estimator. Interface/acquisition
failure is an invalid assay, with at most one isolated prospective repair.

Provisional budget: **2–4 cached GH200 hours**, plus fresh setup and interface
qualification; early acquisition failure stops after 70 updates. This is an
unbenchmarked estimate, not a host quote. No GPU is requested now.

Only a pass permits independent seeds, matched-compute simple controls, a second
family, same-information SLIFT/released multi-token SDPO and then one locked
confirmation. Independently validated persistence measurements remain a separate
requirement. This DEV screen addresses method usefulness; it cannot establish
human persistence or repair the theoretical/empirical estimand gap by itself.
The [full plan](ICLR_2027_PAPER_COMPLETION_PLAN_20260905.md) specifies all controls,
outcomes, stopping rules, runtime limits and the conditional queue.

## 6. Completing an actual paper

Begin an evidence-linked nine-page manuscript and theorem appendix immediately.
Use a real claim table, all-comparator plots and explicit blank cells for unrun
work. The report and implementation are not substitutes for the manuscript.

By **September 8**, require a credible independently validated persistence-data
route. By **September 11–12**, require a replicated useful neural result, fair
baselines, an honest external estimand and a complete evidence-grounded draft.
Without that package, recommend a later venue. On a scientific go decision,
reserve September 13–17 for confirmation, reviewer scrutiny, proof checking,
anonymization and locking the actual title/authors/abstract; September 19–24 for
final reproducibility and PDF/supplement QA.

Official deadlines remain **September 18** for a genuine abstract and **September
25** for the full paper, both **11:59 PM AOE**; initial main text is nine pages.
New authors cannot be added after the abstract deadline. Follow the explicit
initial-submission rule despite inconsistent later ten-page FAQ wording.
[ICLR 2027 author guidelines](https://www.iclr.cc/Conferences/2027/AuthorGuidelines)

Substantive AI-assisted research design, theory, code, synthetic data and
interpretation must be disclosed, not described only as language editing.
[ICLR AI policy](https://www.iclr.cc/Conferences/2027/AIPolicyForAuthors)

This audit supports neither a promised acceptance probability nor submission of
the present evidence. The recommended path is shorter, more demanding about the
actual contribution, and explicitly allowed to end without an ICLR submission.
