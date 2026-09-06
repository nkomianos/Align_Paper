# Alignment research selection under 50 H200 hours

Decision date: 6 September 2026. These are prospective research candidates, not positive results or claims of acceptance probability. No new GPU run has occurred. Prefer one sharply tested thesis to a portfolio of partial demonstrations. Original OSH/CLARA findings are in `OSH_CLARA_INDEPENDENT_AUDIT_20260905.md`.

## What the literature rules out

| Primary source inspected | What is already covered | Consequence for us |
|---|---|---|
| [Does Fine-Tuning Undo Activation Steering?](https://arxiv.org/html/2608.24988), August 2026, results and limitations | Behavioral recovery can occur while the embedded steering edit remains. | Merely showing that weights remain changed is not new; test a causal dependence and an actionable repair. |
| [Reference-Grafting Matches Fine-Tuning](https://arxiv.org/html/2608.29458), August 2026, methods and discussion | Cheap reference-based elicitation, circuit localization, failure across induction mechanisms and under rerouting; honest-reference acquisition explicitly remains unresolved. | Generic activation patching, unknown-reference difficulty, and MCQA/coherence mismatch are occupied. A new result must improve reference acquisition/qualification or isolate a different boundary. |
| [Mechanistically Eliciting Latent Behaviors](https://arxiv.org/html/2606.29604), June 2026, methods and organism results | CPE discovers weight perturbations and selects them with validation metrics; rewards and alignment-faking organisms studied. | “Use LoRA to elicit aligned behavior” is not a contribution. Its helpful-only organism objective is explicit; do not misreport this as a hidden error or harmlessness guarantee. |
| [The Rogue Scalpel](https://arxiv.org/abs/2509.22067), September 2025 | Steering can compromise safety. | Generic collateral damage from steering is not new. |
| [Sycophancy Towards Researchers](https://arxiv.org/html/2606.08629), June 2026 | Researcher expectations confound alignment-faking interpretation; probing and steering do not automatically disambiguate intent. | Do not resurrect our old evaluation-cue assay as a scheming detector. |
| [TamperBench](https://arxiv.org/abs/2602.06911), February 2026; [spectral model locking](https://openreview.net/pdf?id=cjrm7bo6Eg), 2025 | Weight-tampering evaluation and model locking already exist. | OSH must have a specific advantage and correct threat model, not just loss of coherence plus a key. |
| [Compositional conformal prediction](https://arxiv.org/html/2405.15912); [active learning for neurosymbolic synthesis](https://arxiv.org/abs/2508.15750) | Neural uncertainty can be propagated through symbolic programs; uncertainty-aware querying is established. | A CLARA pivot must beat these, not rename them tropical inference. |
| [METR MALT](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/) | Real and prompted agent traces, human-reviewed labels and monitoring experiments. | Gives a cheaper external substrate than training a new scheming organism, but monitoring itself is established. |

Literature checks narrow novelty; they cannot establish that a gap is unoccupied. Full bibliography and code comparison remain mandatory before committing the main compute budget. Search leads from secondary summaries were not treated as verified findings. In particular arXiv:2606.08381 has a different title from a secondary search summary; it is not evidence for the claimed symmetric intervention experiment.

## 1. Does behavioral recovery create dependence on a compensating update?

**Priority: best new mechanistic candidate for a small pilot.** Biological inspiration: compensation after a persistent perturbation can conceal dependence on that perturbation. The test concerns parameter interactions, not biological feelings or a mesa-objective.

Let W be the base, E a fixed known behavior edit, and D_E the update learned while E is present. Evaluate W, W+E, W+E+D_E, and W+D_E. Separately train D_0 without E and evaluate W+D_0 and W+E+D_0. This separates ordinary training, edit removal and conditional adaptation. Preserve the exact additive E during training so subtraction is well-defined; matched forward checkpoints must verify numerical implementation.

**Question:** after an apparently repaired model returns to baseline behavior, does removal of E expose a systematic opposite shift, and can we localize and remove the responsible part of D_E while retaining learned utility? The interaction statistic is [f(W+E+D_E)-f(W+D_E)]-[f(W+E)-f(W)], with f a preregistered behavioral measurement. Raw arithmetic interaction alone is insufficient: nonlinear response curves can create it without a learned compensator.

**Pilot:** one 3B-4B instruction model, short benign style edit (brevity) to qualify apparatus, then a safety-relevant behavior on harmless proxy tasks with objectively scored authorized/unauthorized actions. No real external tool execution. At most 128 learning cases and 128 disjoint development families, with semantic polarity balanced. Freeze a fresh confirmation split before checkpoint selection. Require the original edit to have a >=15 pp effect and recovery to remove >=75% of that effect without >3 pp utility loss before interpreting the withdrawal endpoint.

**Controls:** D_0; norm-matched random edits; edits inducing comparable initial behavior change; edit-dose curve before/after adaptation; response-length and token-bias controls; heldout benign usefulness; module swaps of D_E against random matched swaps. Independent training seeds are replication units; paraphrases are clustered within scenarios. Match tokens, steps, optimizer and data. Do not compare LoRA to full fine-tuning without declaring the difference.

**Go gate:** withdrawal produces >=10 pp directional shift beyond the matched controls on development data, without general incoherence, and causal module restoration attenuates it. For confirmation require a paired scenario-cluster interval excluding zero, at least two model families, three training seeds, and a practical correction that beats simple full revalidation/retraining at matched budget. These gates are research decisions, not a powered guarantee with n=128.

**Kill:** no acquisition/recovery means invalid apparatus, one permitted repair within cap; no controlled withdrawal effect kills this hypothesis; effect explained by the static dose curve or generic utility damage kills the mechanistic story. If removing the edit alone already solves the problem without collateral damage, there is no need for a new method. If the closest paper/code already performs this full factorial and repair, stop before GPU.

**Budget:** target 1-2 H200 hours for the pilot, hard cap 3 including loading and idle time. These are planning estimates, not measured runtimes. Conditional expansion 12-18 hours. Need source-code comparison and prepared runner before deployment; not currently deployment-ready.

## 2. CLARA pivot: which safety guarantees survive operator rule edits?

**Priority: strongest CPU-first candidate.** Retain frozen perception, editable declarative rules and proof certificates. Remove the false tropical-attention and nearest-neighbor/treewidth claims.

**Question:** a perception system is calibrated for one set of policy queries. What can still be certified after an operator edits the rules or chooses queries based on observed errors, without retraining perception? The target is selective decision reliability and usefulness under composition changes, not a new generic conformal theorem.

**Critical existing solution:** if a joint scene prediction set contains the true complete scene and symbolic propagation is sound, all deterministic rule queries over that set are simultaneously sound on that coverage event. Adaptive rule choice does not automatically break this guarantee. Our paper must compare against this strong baseline and find an efficiency or expressiveness advantage, not manufacture a failure by calibrating the wrong object.

**CPU pilot:** an exact small-world generator with enumerable latent scenes, shared-fact correlations, rule edits, and an independently implemented truth evaluator. Compare point predictions, marginal fact sets with union bounds, joint scene sets propagated through rules, query-specific calibration, and exact oracle posteriors. Separate arbitrary chosen edits from deployment distribution changes. Include duplicates, equivalent rewrites, longer conjunctions and disjunctions, missing facts and negatively correlated facts. Cluster by underlying scene, not rule/query count. All syntax-only equivalents must preserve decisions under the declared semantics.

**Possible method:** certify only the facts relevant to the proposed action using reusable provenance and a simultaneous scene-level uncertainty budget; defer when competing supported worlds disagree on the action. This is an implementation hypothesis, close to existing abstract interpretation and selective prediction. Derive a valid guarantee first and compare label-efficiency at the same accepted-action error rate. Ordinary Bonferroni/joint sets may already solve it well.

**Kill:** no advantage over the correct joint-set baseline; only toy independent facts show benefit; violation relies on an unfair semantic substitution; or no viable naturally annotated scene/task dataset. If useful, use cached predictions from a frozen image model on ordinary object/relation tasks, split by original image/video rather than frame. Avoid the full surveillance platform and end-to-end VLM retraining.

**Budget:** CPU falsification first (0 GPU); 1-3 hours frozen prediction extraction only if CPU result and novelty survive; conditional full study 8-12 H200 hours. Requires an empirical advantage beyond a known mathematical implication to justify ICLR.

## 3. Elicitation with unreliable references

**Priority: reserve.** Reference-grafting explicitly needs a trustworthy same-mechanism reference. Quantify whether a small, partly erroneous reference can be qualified cheaply, and whether failure can be predicted before trusting the elicited model.

**Pilot:** benign math/coding model organisms with known reference truth; hide the unlock reference from the auditor. Give equal budgets of 1, 4, 16 and 64 independently checked examples. Vary reference failure and induction mechanism separately. Compare mean/median/trimmed reference, task-solution-verified reference, CPE, small SFT and no intervention. Use executable correctness, not an LLM claiming it is honest. Every selected reference example is charged to the audit budget. Split tasks before obtaining references.

**Kill:** performance requires oracle unlock captures; robust averaging already matches the proposed method; only content transfer within one lock is demonstrated (already known); or qualification predicts only corruption injected by our generator. Stop a pilot at 2 H200 hours; reserve 8-12 hours for replication only if it survives. This is not permission to launch a broad defense-bypass search.

## 4. Low-compute external-data alternative: specification-grounded monitoring

MALT permits auditing whether a monitor distinguishes a forbidden shortcut from a legitimate operation under the actual task rules. The same surface operation can be legitimate maintenance or a reward hack; keywords alone do not establish the distinction. A candidate method retrieves the relevant task constraint and requires an evidence span linking action to violation.

Before implementation, inspect MALT's label provenance and existing monitor prompts: if they already implement this comparison and perform well, kill the idea. Use only reviewed, relevant labels; unreviewed examples are not clean negatives. Split by task family and agent lineage, distinguish prompted from natural cases, and include task-only, trace-only and full-context baselines. Do not claim a population prevalence from a curated dataset. Measure recall at a validation-fixed false-positive rate and include an unmodified strong full-context monitor. Any edited-policy counterfactual requires independent relabeling; it is developmental until validated.

A CPU metadata and lexical baseline can reject leakage explanations before any long-context inference. Target 1-2 H200 hours for a bounded development screen, then 6-10 only if evidence quality and baseline advantage qualify. Generic MALT classification is not novel; constraint-sensitive transfer to unseen tasks is the proposed question, not an established gap.

## Ideas to decline now

- A generic mesa-optimizer detector: behavioral agreement with an optimizer does not identify an internal optimization algorithm or its objective. A credible study needs independently known algorithms, causal interventions, and representation-invariant predictions. This is not a cheap LLM experiment under this deadline.
- Repeating OSH poison/inverse recovery at another rank/model: standard linear algebra plus inference cost, without a new security guarantee.
- “Prove safety with tropical logic”: proofs remain conditional on perceptions and rules; proposed algebra contains counterexamples.
- Demonstrate safety gains by increased refusal alone, or prove intent from a linear probe: neither has adequate construct validity.
- A new self-distillation-collapse story without defeating the close July/August baselines already identified in the previous shortlist.

## Compute allocation and paper decision

50 H200 hours is the total ceiling, not 50 per idea. Reserve 5 hours for environment/loading/failed jobs, up to 6 for **at most two** new pilots plus the already-prepared <=1-hour Hindsight calibration, 24 for the single winning line, 10 for independent-family/seed confirmation and external validation, and 5 for final reproduction. Total 50. Stop when a cap is reached; no automatic multi-idea queue. Measure throughput with a tiny batch before scaling, count allocated idle time, cache dependencies/data locally, and keep all development and confirmation artifacts separate.

Hindsight remains conditional and its existing calibration package remains intact. Do not silently replace its protocol or launch it alongside a new full study. The new research recommendation is: complete the compensation code/novelty check and CLARA CPU pilot, compare them with the Hindsight calibration outcome once hardware is available, then commit to one line.

For an actual paper: (1) establish a valid effect and a baseline advantage; (2) freeze one thesis and its scope; (3) run independent confirmation, model-family replication and an external task; (4) reproduce every table from immutable outputs; (5) write the manuscript around only those verified claims. A negative or failed pilot produces a stop decision, not another formatted paper. Recheck official submission dates before scheduling. No candidate here is yet paper-qualified, and acceptance cannot be inferred from a promising pilot.
