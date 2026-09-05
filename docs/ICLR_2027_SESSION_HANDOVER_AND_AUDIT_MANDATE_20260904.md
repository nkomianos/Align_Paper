# ICLR 2027 research handover and independent-audit mandate

**As of:** 4 September 2026 (America/Los_Angeles)

**Repository:** `C:\Users\nkomi\Documents\GitHub\Align_Paper`

**Working branch:** `codex/iclr-2027-thesis-scout`

**Immediate mission:** audit the entire research program, correct any invalid conclusions, choose one defensible contribution, and turn it into a submission-ready ICLR 2027 paper.

> Give this entire document to the next session. It is a handover, not a claim that the present conclusions are correct. The next session must independently inspect raw evidence, code, manifests, and primary literature before accepting any scientific conclusion below.

## 1. The actual goal

The terminal deliverable is **one scientifically defensible, submission-ready ICLR 2027 paper**, with a crisp thesis, honest positioning, validated experiments, reproducible code and evidence, a complete manuscript, and a genuine abstract. The goal is not to maximize the number of ideas or experiments, and it is not complete when a benchmark, protocol, or promising toy result exists.

Acceptance cannot be guaranteed. Optimize acceptance probability by doing correct, differentiated science rather than by manufacturing a positive result. A valid negative should kill or redirect a claim. An invalid assay should be repaired, not described as a disproof. Do not weaken thresholds after seeing results.

Official deadlines and constraints, which must be rechecked before submission:

- Abstract: **18 September 2026, 11:59 PM AOE**.
- Full paper: **25 September 2026, 11:59 PM AOE**.
- A genuine abstract is required; no new authors may be added after the abstract deadline.
- Double-blind submission; nine main-text pages at submission; required AI-use statement.
- Primary sources: [ICLR 2027 Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers) and [Author Guidelines](https://www.iclr.cc/Conferences/2027/AuthorGuidelines).

The compressed deadline makes an audit and one hard selection urgent. Do not submit a placeholder or an under-supported story merely to meet the date.

## 2. Current truth in one page

- **No GPU experiment is currently known to be running.** The last GH200 address, `192.222.57.245`, timed out, and the user said the instance would be terminated. Never assume a paid host is alive; verify before reporting otherwise.
- **No candidate is presently paper-qualified.** Several mechanisms or apparatus checks are positive, but none yet supplies the full package of novelty, valid neural effect, strong baselines, external validity, replication, and manuscript.
- The strongest live lead is **Hindsight identifiability**, but it is conditional yellow, not a green light.
- The narrow Hindsight claim is that a user’s next message is post-treatment feedback: the assistant response can affect transient expression or persistent preference state. The same immediate logs can therefore support opposite policies for a declared persistent-preference objective. Sparse delayed neutral probes may identify and correct the update.
- The strongest Hindsight evidence is exact finite-state non-identification plus a positive finite-state sparse-anchor correction. The decisive Qwen3.5-9B gradient and policy-learning experiments have **not run**.
- The original automatic-polarization/bifurcation framing is not robust enough: symmetric dynamics can restore balance, and a natural-initialization shared-LoRA test was negative. Do not revive this as the headline without new prospective evidence.
- The **UNDO/local-rewrite** method is a valid negative on the current synthetic task: it lost to canonical-state distillation and is parked. Its originally stated `add(x); retract(x) = identity` algebra was also overgeneralized and has been corrected.
- Most earlier candidates were either valid negatives, invalid assays, or developmental apparatus results. They are useful as a failure map, not as a paper portfolio to keep expanding indiscriminately.
- A current literature collision, **SLIFT** (August 2026), already addresses selective self-learning from heterogeneous user feedback. It does not appear to address causal influence/persistent preference transitions, but it must be a primary baseline and sharply narrows the novelty claim.
- GitHub is **not current**: at handover start the branch was 195 commits ahead of its remote. Do not assume the public repository contains the latest work.
- `analysis/` is untracked and user-owned. Do not stage, delete, or rewrite it.

## 3. Evidence vocabulary: enforce this rigorously

Every result must receive exactly one of these labels before it affects a paper decision:

1. **Valid positive:** the assay qualified prospectively, the planned endpoint passed, artifacts verified, and the result supports the stated claim.
2. **Valid negative:** the assay qualified, the planned endpoint failed, and the result weighs against the stated claim or method.
3. **Invalid assay/capability:** comprehension, parsing, target matching, model-organism formation, integrity, or another prerequisite failed. It is not evidence for or against the scientific endpoint.
4. **Developmental/apparatus only:** a smoke test, power audit, data construction, code check, synthetic positive control, or exploratory analysis. It may justify a next experiment but cannot green-light a paper.
5. **Not run:** implemented or queued code is not evidence.

Past discussion sometimes blurred these categories. The new session must reconstruct the ledger using them rather than trusting prose labels.

## 4. Repository, reproducibility, and security state

At the start of this handover:

```text
branch: codex/iclr-2027-thesis-scout
HEAD:   d890e05 Repair EndoPAHF full-learning gate
remote divergence: ahead by 195 commits
untracked user-owned path: analysis/
uncommitted SLIFT work:
  scripts/fetch_slift_snapshot.py
  scripts/prepare_hindsight_slift_g2b.py
  scripts/verify_hindsight_slift_g2b.py
  src/interaction_sprint/hindsight_slift_baseline.py
  tests/test_hindsight_slift_baseline.py
```

Recheck `git status`, `git log`, remotes, tags, and bundle integrity immediately. Preserve unrelated changes. Do not push without explicit user authorization.

Local test invocation used during the latest work:

```powershell
$env:PYTHONPATH='src;.;scripts'
C:\Users\nkomi\miniconda3\python.exe -m pytest
```

The latest committed deployment bundle before the uncommitted SLIFT work was:

```text
artifacts/deployment/hindsight_gpu_queue_d890e05.bundle
SHA-256 767f79598697b2b12770de74f9d3da359378fe8b6a145e0edf463862b1940d92
```

Do not overwrite it. After the audit and any new commit, create a fresh, commit-named bundle and sidecar, verify it, and preserve previous bundles.

Historical SSH key path:

```text
C:\Users\nkomi\.ssh\ECE4150-LAB2.pem
```

No host should be contacted unless the user supplies or confirms current access. A Hugging Face token was pasted into chat earlier. **Never copy it into code, documentation, logs, manifests, or shell history. Ask the user to revoke/rotate it before future use and provide secrets through an environment variable or protected token store.**

## 5. Portfolio ledger

This is a summary, not a substitute for raw evidence. For each row, open the linked protocol/result documents and their artifact roots, rerun the committed verifier, and compare the result with this description.

| Direction or stage | Best current classification | What happened | PI disposition |
|---|---|---|---|
| Under Extinction, stage 1 | Valid negative for registered candidate | Acquisition occurred, but DEV comprehension/selectivity/reversal/robustness gates failed. Locked TEST stayed closed. | Kill original formulation. |
| DID v1.1.1 diagnostic | Invalid assay/interface | 76,800 scores; label-equivariance error `.1897`; label-swap semantic agreement `75.09%`. Interface contamination prevents controller inference. | Do not claim controller failure. |
| Provenance authority | Valid negative for tested effect size | Self-versus-external effect `.0277` percentage points versus a 10-point requirement. | Stop candidate. |
| Response-interface invariance | Valid negative | `0%` selection disagreement and `.1051` percentage-point spread. | Stop candidate. |
| Hybrid memory | Developmental/weak negative | Recurrent signal `.168` logits versus `.50` threshold; KV signal `14.449`. | Insufficient for paper. |
| Recency alignment-faking G0 | Valid negative | Registered checks failed; corrected effects `-.072` and `+.704` percentage points. | Kill. |
| Recipe-invariant J0 | Valid negative | Source agreement strong, held-out steering `-.095` and `-.224` percentage points. | Kill. |
| Semantic ancestry developmental run | Invalid/confounded | Role and style controls were confounded. | Developmental only. |
| Semantic ancestry G0b | Valid negative for exact remedy | Collapse signal large (`+21.7` to `+53.3` points), but specificity failed and remedy missed MMR in all four cells. | Kill exact candidate; mechanism clue only. |
| Effect-consistency uncertainty | Valid negative | Passed 3/10 registered gates. | Kill. |
| Visual patch-phase | Valid negative | No specificity or phase lock; ensemble worsened Qwen by `7.5` points and Gemma by `17.5` points. | Kill. |
| Feedback leakage / Critique Oracle | Valid negative | No fabricated-authorization evidence; safe recovery improved `35.4`/`45.8` points. Hypothesized gaming trade-off absent. | Kill candidate. |
| Reward-hack early warning | Developmental failure | Both AUROCs `1.0`, no advantage, false alarms `26.7%`. | Not a paper result. |
| Phantom rollback | Invalid assay/capability | Availability effect zero; Qwen comprehension `12/16`; Gemma parse `0/16`. | Do not call thesis false; do not rerun unchanged. |
| Reward Extinction Debt | Invalid model organism | Induction `+97.69` points, but organism matching failed; debt AUC difference `-.0174`, CI `[-.0967,.0621]`. | Formation failed; endpoint uninterpretable. |
| Hindsight initial smoke | Developmental only | 64 forwards; `6.724`-point anchor restoration on a conflict subset. | Replaced by better assays. |
| Hindsight broad prompt assay | Invalid/no signal in this assay | 512 forwards; anchor restoration only `.003142` points. No training or state transition. | Do not infer thesis false. |
| Hindsight exact finite-state identification | Valid positive but elementary | Immediate logs identical across expression/transition worlds; policy ranking reverses; delayed neutral probe separates them. | Useful theorem component, not a paper alone. |
| Hindsight minimax bound | Valid theoretical result | Observational worst-case regret lower bound `.2` deterministic and `2/15` randomized; minimax action-1 probability `1/3`. | Paper component if neural story survives. |
| Hindsight finite-state anchor correction | Valid positive, developmental scale | At 16 anchors, expression regret raw `.086281`, equal-anchor `.013770`, augmented `.009343`; reversal cells `.259371/.018028/.013609`; transition raw and augmented `.000067`. | Promising method screen, not paper-qualified. |
| Hindsight anchor/selection robustness | Valid positive, developmental scale | Clean reversal regret raw `.345673`, equal-anchor `.006765`, augmented IPW `.004177`; selection bias `.106438` naive to `.0056125` IPW. Fails when the anchor itself flips target. | Explicit identification boundary. |
| Hindsight bifurcation exact map | Developmental positive | 14/16 confirmation contexts had three fixed points; median separation `.9935`; fixed marginal control within `.082`. | Mechanism only. |
| Hindsight calibrated shared-LoRA bifurcation | Valid negative gate with suggestive contrast | Dynamic separation near 1, but a fixed control reached `.171` above frozen `.10` maximum. | Failed attribution gate. |
| Hindsight natural-initialization LoRA | Valid negative | Dynamic-over-fixed movement median `.0023008`; final arm gap `.00494`; only 9/14 contexts in predicted basin. | Park polarization/bifurcation line. |
| Hindsight symmetric longitudinal model | Valid negative for tested dynamics | Exact equations and 28 simulations show adaptation can reduce, not necessarily amplify, baseline-preference drift. | Automatic-harm claim is false in this formulation. |
| Hindsight semantic constrained-choice | Developmental positive | Published hindsight `32/32` DEV and `31/32` confirmation; direct preference control weaker. No learning or endogenous transition. | Teacher signal only. |
| Hindsight full-response teacher audit | Developmental warning | Greedy correct label `55/64`, exact whole-response A/B `25/64`; format and semantics diverge. | Separate semantic scoring from format. |
| Hindsight full-vocabulary learning control | Invalid teacher qualification | Held-out accuracy `8/32` to `24/32`, below gates; teacher `44/64`; downstream arms did not run. | Apparatus diagnosis. |
| Hindsight acquisition positive control | Valid apparatus positive | SFT learned `64/64` train and `32/32` held-out with choice mass `>.999996`. | Usable regime, not thesis evidence. |
| Hindsight matched learning | Valid weak/negative method result | Six LoRA arms; copying and fixed noise both `8/16`; anchors `9/16` versus anchor-only `8/16`; truthful supervision also transferred poorly. | Acquisition regime too weak. |
| Hindsight human PUPPET classical reader | Valid negative for weak reader; capability-limited | TF-IDF reader negative and weaker than source capable-LLM baselines. Initial root also had one-target binding bug, then repaired. | Does not falsify causal thesis. Capable reader remains unrun. |
| Hindsight Qwen3-0.6B gradient rehearsal | Invalid capability | 34/64 teacher cases; mean target probability `.5529`; no gradients admitted. | Do not use 0.6B endpoint. |
| EndoPAHF 0.6B interface rehearsal | Invalid capability | Immediate/new target `30/32`; old delayed expression `23/32`. | Stronger exact-interface preflight required. |
| PAHF source audit and EndoPAHF v3 construction | Developmental/apparatus | 630 matched learning bases; v3 has 2,520 rotated learning rows, 384 DEV variants, 1,024 locked confirmation variants. | External substrate, not evidence of assistant-caused change. |
| EndoPAHF learnability/power audits | Developmental qualification | Full 630-base old-target diagnostic improves to `47.92%`/NLL `1.0744`; new target `70.83%`/`.8976`. Cluster rule routes null `.0075`, planned signal `.9025`, noisy signal `.7450`. | Statistical/data checks only. |
| EndoPAHF neural G2 | Not run | Exact-interface preflight, DEV routing, checkpoints, and locked-confirmation verifier are implemented. No capable-model endpoint exists. | Conditional on Hindsight G0/G1. |
| UNDO inference robustness | Descriptive signal below gate | Edited histories `87/96`, canonical/padding/counterfactual `96/96`; 9.375-point gap missed 10-point gate by one case. | Signal is real descriptively, not a contribution. |
| UNDO local-rewrite training | Valid negative method comparison | At depth 60/100: baseline `23/32`, terminal SFT `24/32`, canonical distillation `28/32`, local rewrite `24/32`. Local loses four to canonical. | Park; no three-seed expansion. |
| LC0 latent communication smoke | Invalid interface | Text/latent `12.5%`, no-message `37.5%`, parse `68.75%`. | Replaced by reasoning DEV. |
| LC0 reasoning DEV | Invalid due validity gate | Text variants `100%`, latent `93.75%`, no-message `0%`, but parse validity `81.25%` due token caps. | Interesting apparatus, not evidence. |
| C2C released baseline | Valid weak utility result after parser repair | Upstream parser: receiver `51/128`, sender `113/128`, fused `72/128`, disabled `50/128`; conservative fused `65/128`. | Fusion beats receiver but loses to sender and costs more; park. |
| Native-video Efference EP0 | Invalid oracle | Native, joint, and oracle all `25%`. | Assay unusable; no expansion. |
| Visual hindsight | Valid negative | 240 calls; readable and prefix accuracy `100%`, but treatment effect zero. | Kill frozen hypothesis. |
| Cross-tokenizer/byte-clock coupling | Valid negative DEV | 1,024 outputs; hierarchical variance/cost did not beat simple token-clock baseline. | Park heuristic. |

Many other candidate memos, code paths, and unit tests exist. **Do not count an implemented runner, a passing unit test, a generated dataset, a power simulation, or a deployment bundle as an experiment supporting a paper.** The research journal and evidence index must be reconciled with the filesystem to enumerate every actual run.

## 6. The live Hindsight candidate

### 6.1 Defensible narrow thesis

Let `z_t` be a latent persistent user state, `a_t` the assistant response, and `o_(t+1)` the user’s next message. Interaction learning often treats `o_(t+1)` as passive hindsight supervision. But `a_t` can affect both transient expression and persistent state. Ordinary next-turn logs can therefore combine:

- information about a preference that already existed;
- expression/compliance changes with no persistent preference change;
- persistent preference or belief transitions caused by the assistant.

The intended contribution is not merely to observe this ambiguity. A viable paper needs:

1. a compact formal non-identification and decision-regret result;
2. explicit identification assumptions for delayed neutral probes or other sparse interventions;
3. a neural causal-hindsight method that beats raw feedback and fair equal-label controls;
4. transfer to a natural task surface;
5. at least one independent validation substrate and a second model family;
6. clear sensitivity analysis when anchors are selected or contaminated.

### 6.2 What is already positive

- Exact expression-versus-transition observational equivalence under a shared action-independent emission factorization.
- Exact policy-ranking reversal for a declared persistent-state objective.
- Exact delayed-probe separation and observational minimax regret bounds.
- A finite-state residual/augmented estimator that outperforms raw immediate feedback and an equal-anchor baseline under clean and known-propensity selection settings.
- A clear impossibility boundary: if the delayed anchor itself is sufficiently affected to reverse the target, reweighting faithfully estimates the wrong target.
- A PAHF-derived controlled surface with exact matched immediate logs and separated delayed targets.

### 6.3 What is negative or unresolved

- The original claim that hindsight learning generically causes self-amplifying polarization is unsupported. It fails under simple symmetric dynamics and under natural shared-LoRA initialization.
- The finite-state correction is based on standard difference estimation/IPW machinery. Novelty must come from the specific post-treatment next-turn LLM problem, neural consequence, and empirical correction—not from claiming new causal inference mathematics.
- There is no capable 9B neural gradient endpoint, no successful policy-learning endpoint, no external neural transfer result, no second-family replication, and no human causal validation.
- Talk2AI and PUPPET are observational or estimand-mismatched for the clean causal claim. Do not claim that they identify message-level preference influence.
- EndoPAHF is a controlled construction derived from exogenous persona evolution. It can test learning under matched logs, but it is not evidence that an assistant changes real users.
- The full current design is complicated. Reviewers may see a synthetic causal toy with an engineered correction unless the neural result is simple, strong, robust, and tied to an actual interaction-learning baseline.

### 6.4 Closest prior art and novelty boundary

Re-run a current primary-source literature search. At minimum audit:

- **SLIFT, _Different Feedback, Different Updates_ (arXiv:2608.09109):** decomposes feedback into Fix/Spec/Null and trains a Generalist/Specialist. This occupies generic selective self-learning from heterogeneous feedback.
- **PUMA, _Know You Before You Speak_ (arXiv:2605.24647):** explicitly separates action-conditioned user-state transitions and utterance observations, but uses supervised state/simulator structure rather than passive-log identification.
- **SDPO, _Aligning Language Models from User Interactions_:** the direct trigger/baseline for next-turn hindsight learning.
- **ThoughtTrace:** separates message content from private thought/feedback signals; do not claim all feedback provenance questions are new.
- **Causal Inference out of Control (ICML 2024), CAFL, CausalRM, Doubly Robust Alignment, NextQuill:** occupy generic performative identification, propensity correction, causal reward/personalization, or doubly robust adjustment.
- **Dynamic Reward MDPs and Constructive Alignment:** occupy changing-preference objectives and normative ambiguity.

The remaining defensible novelty is a conjunction: **immediate linguistic feedback is observationally identical under transient expression and persistent transition, this causes opposite next-turn self-distillation updates for a declared longitudinal objective, and sparse delayed neutral probes repair the neural update under explicit assumptions.** If the new session cannot defend that conjunction after current literature search, kill or reformulate before spending GPU time.

### 6.5 Unrun capable-model queue

Do not automatically launch everything. Run sequential gates and stop when a prerequisite fails.

1. **Qwen3.5-9B neural gradient G0 v2** — 48 backward passes, previously estimated at roughly 25–70 GH200 minutes. It compares raw immediate, equal-anchor, augmented residual, and full delayed-oracle update directions using outcome-blind panel aggregation.
2. **Independent capable PUPPET reader DEV** — 576 generations, roughly 30–60 minutes. This is a substrate/reader qualification, not the main causal endpoint.
3. **Neural policy G1**, only if G0 qualifies — 26 reset arms, roughly 4–8 GH200 hours. It tests actual optimizer updates against equal-label SDPO and supervised controls and a full delayed oracle.
4. **EndoPAHF exact-interface preflight and G2 v2 DEV**, only if G0 and G1 qualify — roughly 4–8 hours. Locked confirmation remains unopened until DEV qualifies.
5. **SLIFT role/target preflight**, then an official raw-SLIFT plus all-FIX/all-SPEC sensitivity comparison if earlier gates justify it — preliminary estimate 20–60 minutes plus 4–8 hours.
6. Only after all of the above: locked confirmation, second model family, and a genuinely independent external or human validation design.

These are historical estimates, not promises. Benchmark the current host before quoting cost or duration. The fastest honest kill path is about one hour; the full conditional path is roughly 9–18 cached GH200 hours before independent replication, and more if SLIFT is included.

## 7. SLIFT collision audit and unfinished baseline work

The August 2026 SLIFT paper is too close to ignore. The latest source inspection found:

- It labels later feedback as Fix, Spec, or Null.
- Its Generalist uses logged-prefix SDPO plus a behavioral-anchor KL term.
- Its Specialist learns KEEP/APPLY and residual guidance.
- It evaluates Qwen3-8B and Ministral3-14B on MemoryBench and WildFB.
- Its prompt explicitly avoids retaining a Spec supported only by feedback.
- No causal treatment of assistant-caused transient expression versus persistent state transition was found in the inspected paper text.

The anonymous code host was available through a ZIP API rather than `git`. The raw ZIP hash changes because the server regenerates container metadata, so the fetcher validates a canonical member-path/content tree instead. The stable tree SHA-256 observed twice was:

```text
dd6608719f7e46985e88c9b9e674bed076d1755b9090cff80bffca93ed87739e
```

The snapshot had no license file and therefore was **not vendored or redistributed**. The local baseline prepares official-schema raw, all-FIX, and all-SPEC views as sensitivity controls. Prepared v2 contained 630 unique learning bases and 384 DEV tasks; its manifest SHA-256 was:

```text
6927cb66609251223ae443210b0062f9a605ab0c93cadb62f9f1cfbef087dd10
```

Five targeted SLIFT/Hindsight tests passed and official schemas accepted all prepared rows. This is unrun baseline infrastructure, not an empirical result. At handover start these files were uncommitted; inspect and commit only after reviewing them:

```text
scripts/fetch_slift_snapshot.py
scripts/prepare_hindsight_slift_g2b.py
scripts/verify_hindsight_slift_g2b.py
src/interaction_sprint/hindsight_slift_baseline.py
tests/test_hindsight_slift_baseline.py
```

## 8. What we may have been doing wrong

The new session is explicitly asked to challenge these diagnoses and find additional ones.

1. **Breadth-first thrashing.** Too many hypotheses were implemented or screened before a candidate had a stable novelty claim, a validated assay, and a credible paper-shaped success criterion.
2. **Running before qualifying the measurement.** Several expensive or long experiments preceded comprehension, parser, target, oracle, or model-organism checks. Phantom, Reward Debt, DID, LC0, EP0, and early Hindsight runs illustrate this.
3. **Confusing invalidity with falsification.** Capability or parser failures were sometimes discussed as if the scientific hypothesis failed. Later notes corrected many of these statements, but the audit must find all remaining overclaims.
4. **Format and parser confounds.** A/B label inequivalence, answer-position bias, strict end-of-string parsing, token truncation, and whole-response versus semantic-label scoring changed conclusions.
5. **Weak or unmatched model organisms.** A large treatment effect is meaningless if the control organism is not matched or the model cannot understand the assay.
6. **Mixed-factor post-hoc diagnostics.** Wording, reasoning budget, templates, and model settings sometimes changed together. These can diagnose apparatus but cannot cleanly identify a mechanism.
7. **Overvaluing synthetic positives.** Exact toy theorems, power simulations, deterministic dataset constructions, unit tests, and acquisition controls validate components; they do not establish a natural phenomenon or an ICLR contribution.
8. **Arbitrary or underpowered early thresholds.** Some rules were repaired only after prospective power audits. Preserve superseded versions and ensure no endpoint-informed threshold changes occurred.
9. **Pseudoreplication.** Option rotations initially risked being counted as independent samples. The EndoPAHF confirmation analysis now clusters by base task; audit every other result for the same issue.
10. **Outcome-aware selection risk.** Neural policy panels and checkpoints could create researcher degrees of freedom. The current protocol uses outcome-blind panel aggregation; confirm the implementation truly does so.
11. **External-data estimand mismatch.** Public conversational datasets often measure post-treatment reports without an independent pre/post measure of the same latent preference. Association is not identification.
12. **Novelty checks came too late.** SLIFT, PUMA, ThoughtTrace, CausalRM, CCOPD, ICF-Bench, CacheBridge, CREDIT/CamChoreo, and cross-family code verification narrowed ideas after implementation work had begun.
13. **GPU utilization versus scientific discipline.** Idle GPU time felt wasteful, but launching unqualified runs is also waste. Use a prebuilt conditional queue and parallel CPU literature/code work, while avoiding concurrent GPU jobs that change memory or dependency conditions.
14. **No complete paper-shaped artifact.** There is no current nine-page narrative, figure plan, or evidence table whose every claim is supported. The work has optimized gates more than a paper.
15. **Protocol complexity.** Hindsight’s full queue may be too elaborate for the remaining deadline. The audit should seek the smallest decisive theorem/method/experiment package, not execute every prepared branch.
16. **Stale documentation.** `docs/ICLR_2027_SUBMISSION_READINESS.md` still centers semantic ancestry, which is no longer live. `docs/RESEARCH_JOURNAL.md` has stale consolidation metadata. The README ledger predates many 4 September experiments.
17. **Remote Git drift.** A 195-commit local lead means collaborators or a fresh machine may see an obsolete repository. This is a reproducibility and continuity risk.
18. **Credential exposure.** A Hugging Face token appeared in chat. It should be rotated before use and never preserved in the repo.

## 9. Mandatory independent audit before more GPU spend

The next session’s first substantive task is an audit, not another brainstorm. It should produce a claim-level table with columns: `claim`, `evidence root`, `protocol commit`, `unit of analysis`, `verifier result`, `classification`, `error or confound`, `corrected conclusion`, and `next action`.

Perform at least the following:

1. Inventory every protocol, runbook, result file, artifact directory, retrieval directory, checkpoint, manifest, and deployment bundle. Reconcile this inventory with README and `docs/RESEARCH_JOURNAL.md`.
2. For every headline experiment, locate the raw evidence and rerun the committed read-only verifier. Check that the verifier uses the intended commit, configuration, private key, split, and model revision.
3. Recompute headline metrics directly from raw rows, not only from `summary.json`. Compare remote/local SHA-256 manifests where remote artifacts exist.
4. Confirm that TEST/confirmation splits stayed locked until their registered prerequisite passed. Record every exception.
5. Audit target construction and leakage: old versus new preference, delayed probe visibility, user/task overlap, rotation logic, prompt labels, and answer-key access.
6. Audit the experimental unit and uncertainty calculation. Cluster repeated rotations, prompt variants, checkpoints, and seeds correctly. Identify pseudoreplication and multiplicity.
7. Audit statistical power and thresholds. Distinguish frozen gates from post-hoc descriptive analyses. Re-run simulations when the decision rule was changed.
8. Audit comprehension, parser, response format, token budget, oracle, and model-organism qualification before interpreting an endpoint.
9. Inspect training code for reset failures, optimizer-state reuse, checkpoint selection, gradient sign, KL direction, detached tensors, prompt mismatch, and accidental cross-arm contamination.
10. Check whether each control isolates one factor. Mark mixed-factor analyses as developmental.
11. Reconstruct actual GPU use and queue history from logs rather than chat recollection. Identify idle periods, duplicate runs, failed launches, and runs whose output was never retrieved.
12. Repeat a current primary-source novelty search for each surviving claim. Compare exact task, estimand, intervention, training signal, baseline, and evaluation—not titles alone.
13. Red-team the best candidate as a skeptical ICLR reviewer: What is new? What is proved? What is measured? What natural setting supports it? What simpler baseline wins? What assumption is doing the work?
14. Correct README, the journal, submission-readiness page, and evidence index after the audit. Keep an append-only record of corrections rather than silently rewriting history.
15. Only then recommend one of: continue Hindsight, pivot Hindsight to a narrower measurement paper, revive a different candidate with a specified missing contribution, or target a later venue.

Do not trust this handover’s metrics until verified. It was assembled from current summaries and prior verified reports, but the entire point of the new session is to detect mistaken summaries, broken verifiers, selection bias, useless experiments, and wrong PI conclusions.

## 10. Recommended 48-hour decision plan

### Hours 0–8: audit only

- Freeze the working tree state and inventory all evidence.
- Audit the Hindsight theorem, finite-state method, unrun GPU code, PUPPET/PAHF estimands, SLIFT collision, and UNDO negative first.
- Update the portfolio classification and identify any conclusion that changes after raw-data review.
- Draft a one-page Hindsight paper pitch and a one-page rejection memo against it.

### Hours 8–12: choose exactly one candidate

Default selection is Hindsight only if all of the following survive:

- the narrow novelty claim is distinct from SLIFT/PUMA/SDPO/causal-personalization work;
- the theoretical estimand is meaningful and stated without privileging an undefended welfare definition;
- the Qwen3.5-9B gradient gate truly tests the causal correction rather than prompt formatting;
- the controls are fair and the success rule is prospective;
- a plausible external validation can be completed before the paper deadline.

If any of these fail, state the exact failure and make a deliberate pivot. Do not return to broad idea generation until the best one or two alternatives have equally concrete contribution and kill criteria.

### Hours 12–24: fastest decisive run, if GPU is supplied

- Run host/hardware/integrity and model-revision preflight.
- Launch only gradient G0 v2 plus the independent PUPPET reader if the audit approves both.
- Retrieve and verify evidence immediately after completion.
- Stop the candidate if the qualified endpoint fails. Do not rescue it by changing thresholds.

### Hours 24–48: paper or kill decision

- If G0 passes, launch G1; begin the manuscript, figures, theorem appendix, and baseline table in parallel.
- If G0 fails validly, write a concise negative decision memo and select the best already-audited fallback.
- If G0 is invalid, fix only the isolated apparatus issue and repeat once under a prospectively superseded protocol.
- By the end of 48 hours, produce a concrete submission recommendation and a day-by-day manuscript/evidence schedule.

## 11. Paper green-light criteria

A candidate is not green-lit until it has all of the following:

- A one-sentence contribution that survives direct current-prior comparison.
- A valid, prospectively gated neural result on a capable model.
- A method or analysis that beats the strongest simple and published baselines under equal budget.
- At least one natural or externally sourced validation whose estimand is accurately described.
- Replication across seeds and preferably a second model family.
- Correct experimental units, uncertainty, and multiplicity treatment.
- Reproducible code, immutable raw evidence, checksums, and read-only verification.
- A complete paper narrative with limitations and negative results, not merely a positive table.
- Enough time before 18 September to lock the real title, abstract, authors, and claims.

For Hindsight specifically, a theorem plus finite-state correction is insufficient. The minimum paper-shaped evidence is a capable neural gradient result, a policy-learning win over equal-label and raw-feedback controls, natural-surface transfer, and an honest independent validation or sharply stated limitation.

## 12. Things the next session must not do

- Do not say a paper is dead because a model failed to parse or comprehend the assay.
- Do not say a hypothesis survives merely because a toy construction or power audit passed.
- Do not open locked confirmation data to debug DEV.
- Do not tune gates, choose checkpoints, or select prompt variants after viewing endpoint outcomes.
- Do not count rotations or repeated views of one base task as independent samples.
- Do not call standard IPW/difference estimation a novel causal method.
- Do not claim PAHF or an LLM simulator proves real human preference manipulation.
- Do not revive automatic polarization as established.
- Do not auto-run the full GPU queue after an upstream failure.
- Do not launch new idea screens simply to keep a GPU busy.
- Do not delete failed attempts, checkpoints, caches, logs, or superseded protocols.
- Do not stage `analysis/` or expose secrets.
- Do not push, publish, or submit without the user’s explicit authorization.

## 13. Files to read first

Start with these, then follow their artifact and source links:

- `README.md`
- `docs/RESEARCH_JOURNAL.md`
- `docs/ICLR_2027_SUBMISSION_READINESS.md` — stale; audit and replace its semantic-ancestry framing.
- `docs/HINDSIGHT_ENDOGENEITY_PI_MEMO_20260904.md`
- `docs/HINDSIGHT_ESTIMAND_REAUDIT_20260904.md`
- `docs/HINDSIGHT_EXPRESSION_TRANSITION_THEOREM_20260904.md`
- `docs/HINDSIGHT_EXPRESSION_TRANSITION_NOVELTY_AUDIT_20260904.md`
- `docs/HINDSIGHT_PERFORMATIVE_CAUSAL_COLLISION_AUDIT_20260904.md`
- `docs/HINDSIGHT_PUMA_COLLISION_AUDIT_20260904.md`
- `docs/HINDSIGHT_THOUGHTTRACE_NOVELTY_AUDIT_20260904.md`
- `docs/HINDSIGHT_DELAYED_ANCHOR_DEV_RESULT_20260904.md`
- `docs/HINDSIGHT_ANCHOR_ROBUSTNESS_DEV_RESULT_20260904.md`
- `docs/HINDSIGHT_NEURAL_GRADIENT_G0_V2_PROTOCOL_20260904.md`
- `docs/HINDSIGHT_NEURAL_POLICY_G1_PROTOCOL_20260904.md`
- `docs/HINDSIGHT_GPU_QUEUE_RUNBOOK_20260904.md`
- `docs/HINDSIGHT_PAHF_SOURCE_AUDIT_RESULT_20260904.md`
- `docs/HINDSIGHT_ENDO_PAHF_V3_RESULT_20260904.md`
- `docs/HINDSIGHT_ENDO_PAHF_G2_V2_PROTOCOL_20260904.md`
- `docs/HINDSIGHT_HUMAN_FEEDBACK_DEV_RESULT_20260904.md`
- `docs/UNDO_PI_REAUDIT_20260904.md`
- `docs/UNDO_TRAINING_RESULT_20260904.md`
- `docs/DATA_AND_EXPERIMENT_QUEUE_20260902.md` — historical; reconcile with current queue.
- `docs/RESEARCH_EVIDENCE_INDEX_20260902.json` — historical; rebuild after inventory.

Then inspect all candidate and result documents rather than assuming the above shortlist is complete.

## 14. Required first report from the new session

Before asking for GPU access, return to the user with:

1. a concise correction ledger listing every conclusion you accept, reject, or downgrade;
2. a count of actual valid-positive, valid-negative, invalid, developmental, and never-run experiments;
3. the single best paper thesis after current literature review;
4. the strongest reviewer objection and how the proposed experiment resolves it;
5. the exact first gate, what it tests, what each possible result means, and the runtime estimate;
6. a conditional queue that stops automatically on failure;
7. a calibrated paper recommendation, without claiming guaranteed or “high-confidence” acceptance.

The user has explicitly asked for an independent audit of whether we made mistakes, ran pointless experiments, or drew wrong conclusions. Treat that skepticism as part of the research method. The desired outcome is not vindication of the existing program; it is the best honest ICLR paper that the remaining evidence, time, and compute can support.
