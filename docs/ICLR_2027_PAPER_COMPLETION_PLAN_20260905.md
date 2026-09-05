# Paper decision and completion plan after the independent audit

Date: 5 September 2026 UTC / 4 September Pacific. Audited starting commit:
`1cfe22124955dee56c8683cbf0ea0f51137e146a`. This is a proposal and schedule,
**not a frozen or executed experiment**. No GPU access is requested by this note.

## Thesis, pitch, and rejection memo

The strongest remaining research thesis is:

> Immediate linguistic agreement does not identify whether an assistant changed
> transient expression or a persistent preference. For an explicitly measured
> delayed target, sparse independent measurements can be useful only if they
> are valid measurements under explicit stability and reactivity assumptions,
> and improve decisions beyond strong learners given the
> same measurements.

The first sentence has a restricted exact witness. The second is a proposed
empirical contribution, not an established neural finding. The intended paper
would connect a precisely scoped identification result, an honest measurement
assumption/sensitivity analysis, and a useful equal-information learning result.
It would call the estimator standard difference estimation and distinguish
simulated persistence from an observed persistence intervention.

A skeptical reviewer can currently reject the paper for a straightforward
reason: the hidden worlds are constructed, delayed labels reveal the desired
world by assumption, the correction is old, and no successful neural policy or
validated real persistence instrument has been shown. SLIFT occupies selective
feedback learning; PUMA occupies action-conditioned user modeling; Privileged
Likelihood occupies generic likelihood/gradient/utility mismatch; human
expression-versus-reward elicitation effects also have direct prior art. Neither
renaming these ideas nor running the entire prepared queue resolves the objection.

**Decision: persistence-specific pivot; current submission is no-go.** Preserve
the formal witness and useful negative/developmental record. Kill the automatic
polarization headline, universal deterministic `.2` learner claim, broad IPW
policy-superiority claim, and a generic gradient-validity fallback. Do not revive
another historical candidate simply because this pivot is difficult.

## Smallest useful next experiment

The original 48-backward G0 can measure proxy-gradient fidelity, but cannot decide
the now-critical utility or novelty question. Do not spend a separate hour on
that gate followed by 26 same-domain synthetic training arms. The smallest useful
**method screen** is a prospectively superseded, reduced EndoPAHF DEV policy
comparison. Bypassing the old G0/G1 prerequisites requires an explicit new version
with the changes below recorded *before* endpoints; it must not silently alter
or relabel the existing protocols.

This screen cannot itself validate real persistent preference change. A separate
persistence-measurement route is a necessary paper decision, described below.

### Data and estimand

- Use the already constructed 630 learning bases and 96 DEV bases; retain all
  four option rotations at evaluation and average within base. Do not inspect
  or use the 256 reserved confirmation bases to prepare this screen.
- Use the union of the four already outcome-blind disjoint 16-base panels:
  **64 unique delayed-labeled learning bases total**, shared by every sparse
  method. Each method is one model, not a four-model ensemble.
- Every method may access the same 630 immediate logs. Every sparse comparator
  may access the same 64 delayed labels and metadata; none may access the
  full delayed oracle or latent keys for the other 566 learning bases.
- The estimand is **next-query prediction of a declared delayed label for the
  observed cohort**, measured by full-vocabulary target-token NLL and expected
  correct-choice probability, defined here as the full-vocabulary probability
  of the correct native answer token (all other tokens receive zero credit).
  Also report the A/B/C/D-normalized probability, constrained choice NLL, argmax
  accuracy, answer mass and paired option-order sensitivity.
- Full-vocabulary correctness equals answer-set mass times conditional semantic
  correctness. Improvement in both full-vocabulary probability and NLL can
  therefore come from formatting alone. The method decision additionally requires
  a positive gain in A/B/C/D-conditional correct-choice probability against each
  comparator, as specified below. A format-only gain cannot pass.
- This differs from the theorem's `P(a=Z1(a))`, in which the evaluated action
  changes the target state. Do not claim this offline experiment empirically
  validates that endogenous-action policy regret theorem. A bridge requires a
  separately specified sequential environment/measurement study.
- DEV users overlap training users. Report held-out tasks for that cohort, not
  unseen-user generalization. DEV is used for routing, not confirmatory inference.

### Qualification and six trained arms

First use the exact downstream hindsight serialization on the existing 16-base
capability panel, all four rotations and both target contexts. Retain the existing
per-cell capability thresholds from the exact-interface preflight. Before launch,
repair evaluation ID binding, use paired per-base position gaps, and capture
effective model/tokenizer/environment/source pins and initialization hashes.
Preserve raw logits/semantic probabilities so parser and mass failure remain
separate from target comprehension. No role classifier should guess hidden state.

Use Qwen3.5-9B for this first screen, with an immutable resolved model/tokenizer
revision and the declared LoRA configuration recorded in the superseding
protocol. One fixed initialization, 35 updates per arm, fixed final checkpoints,
fresh AdamW per arm; freeze the exact schedules and optimizer settings before any run:

| Arm | Training information and purpose |
|---|---|
| Raw SDPO-style learner | All immediate logs; passive-feedback reference |
| Full delayed oracle | All delayed feedback through the same SDPO-style teacher/loss; extra-information acquisition control, never a fair baseline |
| Pooled anchor SFT | Same 64 delayed bases; direct simple supervised comparator |
| Pooled anchor SDPO | Same 64 delayed bases; teacher-based comparator |
| Standard residual correction | Immediate population term plus paired delayed-minus-immediate term on those 64 bases |
| Nonnegative mixture | Prespecified equal-weight immediate and anchor-delayed losses; tests whether ordinary mixing suffices |

The proposed pooled schedule visits all 64 anchor bases once in the first 32
updates (two bases, each under four rotations, per update), then revisits six
bases selected by a frozen outcome-blind hash in the final three updates.
Every sparse arm uses the identical resulting 280 anchor-row presentations.
The existing helper assumes 16-base panels, so a separate pooled schedule must
be implemented and verified; this proposal is not a ready launcher.

All arms use the same declared learning-rate schedule; the mixture weight is
fixed at `0.5` for each mean loss, not selected on DEV. Run no-update and direct
profile/memory readout as inexpensive descriptive controls with the same allowed
labels. Freeze their construction and answer-to-probability mapping before
launch; deterministic readouts use their actual one-hot action distribution,
not post-hoc confidence smoothing. Do not provide them unobserved delayed labels.
Count actual unique labels, tokens, forward/backward work and elapsed
time. Equal labels do not imply equal compute; if the residual wins, the next
stage must give simple baselines matched-compute optimization under a newly
frozen schedule, without tuning on confirmation.

The transition null in the existing data has identical immediate/delayed text.
Verify its residual-zero identity on CPU. Do not train a duplicate raw arm and
call equality a new transition result. The eventual paper needs nontrivial
truthful/persistent cases and independently varied probe contamination; this
screen is insufficient for that claim.

### Outcomes, stop rules, and runtime

These are prospective **proposed** criteria; serialize them and test the complete
decision rule before launch. Do not call the old marginal power simulations
power calculations for this new rule.

1. If the exact semantic/interface preflight fails, classify invalid assay and
   do not train. Repair one isolated documented issue at most once; another
   failure parks this model/interface.
2. Train raw and full-oracle controls first (70 updates total). Continue only if
   the oracle reduces old-target DEV NLL by at least `0.10` versus both no-update
   and raw, increases expected correct-choice probability over both; raw must
   reduce new-target NLL by at least `0.10` versus no-update and increase new-target
   expected correct-choice probability. Require conditional semantic probability
   to increase in each corresponding control comparison as well; acquisition
   cannot qualify solely through answer-format mass. All prescribed format/mass/paired-position
   diagnostics must qualify. Otherwise stop:
   the learning assay has no demonstrated useful headroom. This is not a negative
   causal finding.
3. If qualified, run the other four arms. A promising method screen requires the
   residual to lower mean delayed-target DEV NLL by at least `0.03` versus **each**
   of raw, pooled SFT, pooled SDPO and the mixture, with at least `0.02` paired gain
   in expected correct-choice probability against each. Require nonnegative gains
   in every rotation stratum and every leave-one-source-user-out aggregate.
   Additionally require at least `0.01` gain in A/B/C/D-conditional semantic
   correct-choice probability against each comparator, with nonnegative
   conditional gains in every rotation stratum and leave-one-user-out aggregate.
   Do not select a favorable
   comparator, panel, endpoint or checkpoint after viewing results. All simple
   readout controls must be reported. To advance the neural-method path, residual
   correction must also improve both full-vocabulary and conditional correct-choice
   probability by more than `0.005` over each prespecified cheap readout. A readout
   within that half-percentage-point margin, or better, stops the neural-superiority
   claim and its follow-up queue. This is a prospective practical routing margin,
   not a statistical equivalence conclusion. All these thresholds remain proposed
   effect floors, not validated power guarantees.
4. A qualified assay that misses the utility/method rule is a negative for this
   correction at this budget. Park the neural-method claim; do not move a threshold
   or switch to oracle distance to rescue it. Improvement only in the surrogate
   is a developmental diagnosis.
5. A pass is developmental support for a controlled learner. It authorizes
   replication and fair published baselines, not a causal-human claim or submission.

Cost: **210 optimizer updates** on the full pass path, versus old G1's 832 and
old G2's 525. Provisional cached GH200 budget **2–4 hours**, plus approximately
15–40 minutes for a fresh environment/model and 10–30 minutes for the exact
interface preflight. A failed preflight stops before updates; a failed acquisition
control uses only 70. These are extrapolations from the old implementation budget,
not measured throughput for this proposed runner. Time the preflight and initial
control steps on an authorized current host, then revise the estimate before
the remaining arms. No current host or price is assumed.

## Conditional follow-up queue

1. **Before compute:** repair endpoints/provenance/lock handling; freeze the
   reduced protocol and full decision-rule simulation. Define an independent
   persistence measurement and data-access path. Stop if no distinctive question
   survives the primary literature comparison.
2. **One qualified DEV screen:** execute the six-arm study above and retrieve
   immutable evidence immediately. No downstream automatic launch on invalidity
   or a qualified negative.
3. **Only after a method pass:** two further independently reset training seeds,
   pooled equal-information and matched-compute controls, and a second model
   family. Prespecify one aggregate endpoint; panel/rotation/seed counts are
   not exchangeable independent observations.
4. **Only after replicated DEV utility:** official raw SLIFT and SLIFT supplied
   the same sparse delayed information, plus faithful released multi-token SDPO.
   Raw/FIX/SPEC sensitivity alone is insufficient. A budget-adjusted adaptation
   must be described as such rather than an exact published reproduction.
5. **Only after those prerequisites:** frozen checkpoint confirmation once, with
   rotation-averaged paired analysis and cohort/user dependence addressed.
   Use full joint operating-characteristic checks, not the old single-NLL
   simulation; do not count 1,024 rotations as 1,024 independent tasks.
6. **Independent persistence validation:** separately randomized exposure,
   comparable immediate and delayed measures of the same construct, neutral
   reference exposure, repeated-measure stability/elicitation-reactivity controls,
   prespecified delay and attrition analysis. Data from a capable PUPPET reader
   cannot substitute. If human data must be newly collected, consent, appropriate
   review, recruitment and follow-up time must be planned before launch; none
   is presently secured. No human-study power/runtime claim is justified without
   an instrument, effect threshold and user-cluster variance.

The proposed full pass path has no defensible 9–18-hour all-in guarantee. Three
seeds, a second family, published full-sequence baselines and measurement validation
may dominate the reduced DEV cost. Budget each only after measuring its predecessor.

## Actual manuscript and calendar

The repository presently has research plans and memos, not a complete current
nine-page paper. Begin writing the claim-evidence skeleton immediately; leave
unrun numerical cells explicitly blank. Do not write a results abstract before
the empirical claims exist.

| Date, Pacific | Required deliverable / decision |
|---|---|
| Sep 5 | Final corrected claim ledger, exact estimand, primary-prior matrix, nine-page outline and theorem appendix. Repair/freeze the reduced study; resolve author list and OpenReview profiles. |
| Sep 6 | Exact-interface and first reduced DEV result if access is authorized; otherwise continue manuscript/theory and resolve access without claiming progress on experiments. |
| Sep 7 | Verified method decision and simple-control comparison. Qualified failure stops the method path; do not start another portfolio screen. |
| Sep 8 | A credible independent persistence-measurement dataset/protocol/access route must exist. If none, abandon the causal ICLR promise; explicitly judge whether a narrower controlled benchmark has enough differentiation. Current evidence says no. |
| Sep 9 | Independent training seeds and second-family result, conditional on DEV. Write results/limitations directly from sealed evidence. |
| Sep 10 | Equal-information published/simple baseline comparison; complete the first full manuscript with figure placeholders replaced only by verified results. |
| Sep 11–12 | Hard scientific go/no-go: useful replicated neural result, defensible external estimand, competitive fair baselines, complete narrative and enough time to resolve review. Otherwise target a later venue; no placeholder abstract. |
| Sep 13–14 | Only on go: one frozen confirmation, complete theorem/proof review, independent claim-to-artifact reproduction. |
| Sep 15–17 | Anonymous nine-page draft reviewed by skeptical readers; final figures, bibliography verification, code/data permissions, reproducibility and substantive AI-use statements; lock real title/authors/abstract. |
| Sep 18 | Genuine abstract only if its claims are already supported; official cutoff 11:59 PM AOE. |
| Sep 19–22 | Resolve review findings without tuning on confirmation; finish appendices, full settings, ablations/negative results and artifact instructions. |
| Sep 23–24 | Clean-environment reproduction, anonymization/link checks and final PDF QA; obtain approval of the concrete final submission. |
| Sep 25 | Full paper and supplement by 11:59 PM AOE, only with explicit user authorization to submit. |

Main-text allocation: introduction/contribution 1 page; estimand/setting and
closest work 1; restricted identification plus assumptions 1.5; standard estimator
and controls 1; validated experimental design 1; actual results 2; sensitivity,
limitations and conclusion 1.5. Four principal figures: observational equivalence
and measurement intervention; every-comparator finite-state correction; neural
label/compute/utility trade-off; held-out persistence/contamination validation.
One experiment table must identify unique users/tasks/seeds, disclosure/access
sets, model revisions and status. Full proofs, all negative/developmental attempts
relevant to the thesis, protocol history and reproducibility receipts belong in
the appendix, with clear distinctions from confirmatory evidence.

The [official author guidelines](https://www.iclr.cc/Conferences/2027/AuthorGuidelines)
confirm the dates, genuine abstract, author-list lock, double-blind materials and
nine-page initial main-text limit. They also require an AI-use statement outside
that limit. The [AI policy](https://www.iclr.cc/Conferences/2027/AIPolicyForAuthors)
requires disclosure of substantive AI contributions: this project includes
research design, theoretical work, code, synthetic data and interpretation,
not just editing. Check authorship/profile/reviewer eligibility now; publishing,
pushing, contacting participants and submitting are separate unperformed actions.
