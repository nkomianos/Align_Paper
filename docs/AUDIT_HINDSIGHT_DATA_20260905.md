# Independent Hindsight empirical, data, and SDPO audit

Audit date: 5 September 2026. Scope: historical Hindsight empirical runs; PAHF/PUPPET target and split construction; G2 training/statistics/verifiers; SLIFT input controls; omitted historical SDPO apparatus. The complete handover was read first. Existing evidence, protocols, and user-owned `analysis/` were preserved. No model inference, gradient experiment, host contact, GPU request, or paid service was launched. Read-only verifier replay and direct arithmetic were performed locally.

Machine-readable ledger: `artifacts/independent_audit_20260905/data/claim_ledger.json`. Reproducible audit code and new receipts are in the same directory. This report should be combined with the independent theory, novelty, G0/G1, and other-candidate audits; it is not the complete repository audit.

## Main corrections

| Claim or inherited description | Independent evidence | Correction and disposition |
|---|---|---|
| PUPPET first attempt had an all-zero target | Saved v1 predictions contain the same **+7** target on all72 rows; every prediction also equals7 | The bookkeeping failure was constant target binding, not specifically zero. Keep v1 invalid. |
| A passing PUPPET verifier demonstrates valid source targets | Current `verify_hindsight_human_feedback_dev.py` passes **both invalid v1 and repaired v2** | The verifier reuses saved targets and cannot detect original target-binding failure. Bind DEV source row, pre/post arithmetic, and frozen code before calling it source-verified. |
| Broad512-forward Hindsight prompt assay was capability-invalid | Its frozen language validity prerequisites all pass: choice mass≈1; independent anchor .994064; truthful correction .938036. Restoration is .0000314161 probability | It is a **valid negative for the narrow registered prompt-restoration endpoint**, not a capability failure or refutation of causal-state learning. No training/state transition occurred. |
| Matched learning is a qualified method negative | Prospective protocol calls itself a small developmental test and states no quantitative success/failure effect gate. Direct truthful supervision transfers only7/16 | Classify **developmental**, with a negative descriptive result. Copying and fixed-noise8/16; correction9/16 versus anchor-only8/16 do not establish an effective method. |
| Acquisition is a scientific valid-positive experiment | Raw final token decisions reproduce64/64 train and32/32 held-out; eight known domains and new phrasings | Classify **developmental/apparatus only** under the mandate's definition. Capability calibration cannot inflate causal-neural positive count. |
| Generation audit has55/64 correct greedy labels | Raw rows give44/64 correct first tokens,55/64 unique correct-label mentions,25/64 strict full responses | Name each endpoint separately. Semantic mention is not first-token accuracy or exact-format success. |
| Calibrated shared-LoRA bifurcation qualified | Raw final fixed-arm maximum separation .170681 exceeds frozen .10 ceiling | Retain the valid negative for its combined attribution screen. Dynamic median near1 does not rescue its failed control. Manifest covers25 files, not20 stated in one historical note. |
| Natural shared-LoRA supports robust polarization | Independent arithmetic reproduces signed upper median .002300806, basin9/14, median absolute arm gap .00494133 | Retain valid negative and park this particular mechanism. No universal rejection of post-treatment feedback follows. |
| PAHF supplies external causal or unseen-user transfer | Learning contains630 bases from20 users; DEV96 bases from19 users; **all19 DEV users are in learning**; no exact prompt or surface overlap | At best, new-task transfer for previously observed synthetic users. Exogenous source persona changes plus invented immediate/delayed feedback do not establish assistant-caused change. |
| PAHF rotations solve all pseudoreplication | Four rotations per base are correctly averaged in G2, but same users recur across many bases | Base clustering supports a conditional task estimand. It does not support population-of-users uncertainty; add user-aware sensitivity or explicitly restrict estimand. |
| G2 power .9025 etc validates full experiment | Replayed cluster audit draws IID normal NLL gains for **one comparison**, omitting accuracy guard, seven-comparison conjunction, shared training, and user dependence | Report marginal simulated operating characteristics only. It is not full-experiment power. DEV .2063 null routing is an engineering routing rate, not a confirmatory significance level. |
| Confirmation was literally never read | Source/preparation verifiers reconstruct and deserialize confirmation labels before any neural gate; learnability helper constructs all partitions while emitting `confirmation_read=False` | Distinguish data-construction access from model-evaluation/selection access. No capable confirmation endpoint was found; literal unreadability is false. This audit never deserialized reserved confirmation content. |
| Raw SLIFT plus FIX/SPEC establishes fair published-baseline superiority | All prepared SLIFT views receive only immediate feedback; correction gets delayed labels | This is appropriate for demonstrating passive-log limitation, but an algorithmic comparison needs the same delayed labels and declared compute budget for a strong published/simple baseline. No SLIFT model comparison has run. |
| SDPO v3 collapse is clean evidence of SDPO failure | Qualified recovery24/24 but omitted preservation; teacher preserves **0/8** initially correct responses; final strict0/64 while content64/64 | Invalid beneficial-learning positive control, with a useful developmental collapse trace. Do not treat it as a causal-feedback or general SDPO rejection. |
| Single-profile receipt proves intended greedy decoding | Receipt verifier passes supplied config; pinned effective-config forensic evidence says sampling executed; recovery stratum empty | Invalid intended assay, no training. Preserve sampled observations and distinguish config receipt from effective execution. |
| Frozen-forward directions fully verified | Rerunning committed v2 verifier fails `numeric sampled_vs_expected_cosine` | Preserve scalar observations and unresolved numerical directions. Do not use this diagnostic for directional/parameter-gradient claims. |

## Count convention and coverage

The ledger separates physical evidence from statistical claims. There are **29 executed empirical assay versions or launch attempts** in this subaudit: **0 valid positive,3 valid negative,13 invalid assay/capability,13 developmental/apparatus only**. Five of the29 are failures before scientific inference: parameter-probe v1, revelation v1, first early Hindsight smoke, initial SDPO-format launch, and initial single-profile launch. Counting only runs with scientific measurements gives24. These counts include Hindsight and its historical SDPO apparatus, not other portfolio directions.

The three valid negatives are the narrow512-forward prompt-restoration assay, the calibrated shared-LoRA combined screen, and the natural-initialization shared-LoRA screen. This count does not mean three independent rejections of the causal-identifiability thesis. None is a capable causal-correction experiment.

The ledger additionally records **33 preparation/model-free-computation roots** and **six grouped never-run neural stages**. Mathematical theorem artifacts, finite-state estimator calculations, power simulations, generated inputs, duplicate source archives, and deployment packages are excluded from empirical counts. Theory claims are evaluated separately by the theory audit and may be valid positives without being empirical experiments. The six never-run groups are the superseded neural-anchor G0, capable gradient G0 versions, neural-policy G1 versions, capable PUPPET reader, capable EndoPAHF preflight/G2/confirmation, and SLIFT preflight/training/comparison. Do not sum stage-group counts with run counts to imply independent experiments.

**28 committed read-only verifier calls were executed:27 passed and1 failed.** The failed call is the frozen-forward v2 numerical replay. Passing receipts include the invalid PUPPET v1 and single-profile passed-config record, illustrating why verifier success is not scientific qualification. See `verifier_batch.json`, `verifier_batch2.json`, and the two early-assay receipts. Thirteen historical archived runner files were compared directly to their named frozen Git blobs: all13 match exactly; see `protocol_source_binding.json`. This verifies code identity, not an externally timestamped preregistration or complete execution replay.

The44 top-level `hindsight_*` roots were inventoried. All manifest member hashes checked after accounting for the delayed-anchor manifest's nested `files` schema. Five roots lack a final manifest: anchor-phase calculation, minimax calculation, aborted parameter v1, aborted revelation v1, and recovered revelation v2. Revelation v2 retains a deliberately truncated invalid `RESULT.json`; complete scores and partial scores agree, and the explicit recovery verifier passes. Do not quietly rewrite it into a normally sealed run.

Five verifier scripts covering seven PAHF source/input/learnability roots were deliberately **not run** because they read source evaluation pools or `confirmation.json`: source audit(v1/v2), external v1/v2/v3, and learnability(v1/v2). Their file hashes were checked; relevant code was inspected; unlocked learning and DEV were independently analyzed. Reserved confirmation bytes were hashed without parsing. A statement that every verifier passed would be false. Mathematical calculations/power roots outside the two G2 power checks were delegated to the root/theory audit, and their substantive classification must use that report.

The actual local PAHF checkout independently resolves to pinned commit `7a11213360a82d5f437a035e3a31c92d6307f8cf`; all pinned source/license/data-file SHA-256 checks pass in `pahf_source_hash_only.json`. The source data were not deserialized. Direct inspection of its actual shopping agent confirms that post-action feedback compares against fixed `test_data['gt']`, and the launcher loads updated personas before phases3/4. This supports the exogenous-evolution interpretation independently of the local summary report.

## Independent numerical results

Direct recomputations avoid importing the experiment's summary functions. Scripts and results are `audit_data.py`, `independent_prior_arithmetic.py`, `pahf_unlocked_independent.json`, `puppet_independent_arithmetic.json`, `pahf_interface_independent_arithmetic.json`, and `independent_prior_arithmetic.json`.

| Endpoint | Raw-row recomputation |
|---|---:|
| Early smoke restoration | 6.72445035 percentage points; only4 selected conflict blocks |
| Broad prompt restoration | .00314161 percentage points;48 conflict blocks |
| Acquisition final train/held-out | 64/64;32/32; held-out minimum A/B mass .99999608 |
| Matched copying/fixed/anchor-only/correction | 8/16;8/16;8/16;9/16 |
| Matched direct truthful / truthful-KL transfer | 7/16;8/16 |
| Semantic hindsight DEV/confirmation | 32/32;31/32 |
| Generation first-token / unique-mention / strict | 44/64;55/64;25/64 |
| Calibrated LoRA dynamic upper median / fixed maximum | .99999819;.17068104 |
| Natural LoRA signed upper median / basin agreement | .002300806;9/14 |
| PUPPET v2 query / user / assistant / full MSE | 418.852958;426.436658;426.881992;427.121049 |
| PAHF0.6B old-expression / new-immediate accuracy | 23/32;30/32 |
| SDPOv2 explicit / teacher / preservation | 16/32;25/32;1/8 |
| SDPOv3 explicit / teacher / preservation | 32/32;24/32;0/8 |
| SDPOv3 final strict / content | 0/64;64/64 |

Repeated option orders, wording variants, checkpoints, and teacher targets do not create independent users. For example, the16 matched-learning evaluation rows represent eight domains; four anchor rows represent two domains. The current semantic and bifurcation "confirmation" cases were already measured in their old assays and are now historical evidence, not reserved new confirmation.

## PAHF construction and G2 review

PAHF changed-pair construction requires exact pre/post task surfaces and different old/new targets; it assigns new-target recommendations and immediate feedback, old-target delayed-expression feedback, and identical immediate/delayed-transition feedback. Thus expression/transition equality is constructed, not learned or observed in real interaction. Old versus new defines an estimand only after declaring which persistent state the task is supposed to serve.

Unlocked v3 checks reproduce2520 learning rows/630 bases,384 DEV rows/96 bases, exactly four rotations per base, equal old/new labels over the complete rotated files, and no same-old/new records. Four disjoint anchor panels contain16 bases each(64 total), with280 repeated anchor-row presentations per panel. The global35-step schedule covers630 bases once. Selected old-target labels are A156/B142/C167/D165 and new-target labels A158/B154/C158/D160: rotation-index balancing is not exact target-label balancing. This mild realized imbalance should be reported; it is not cross-arm leakage because the arms share the schedule.

Training resets look correct in inspected code: `adapter_state` clones tensors, `load_adapter` restores the initial adapter at every arm, a fresh AdamW is constructed, and gradients are cleared. Student loss retains gradients; teachers are calculated under no-grad; residual sign is delayed minus immediate. No concrete reset/sign contamination was found. These are source-code checks, not proof of future execution.

The proposed correction and anchor-only methods use the same sparse delayed labels, but they do not use equal total compute: augmented trains on630 population examples plus280 anchor presentations per panel, whereas anchor-only processes280 anchors. Four-model panel ensembles must be compared to correspondingly aggregated baselines or have inference cost disclosed. Single-seed panel variation is not independent training replication.

The transition-sanity residual is explicitly `anchor_immediate_each-anchor_immediate_each`; it is algebraically zero. Its35 duplicate training steps verify implementation identity, not successful learning in a distinct transition world. A deterministic unit check is sufficient for that identity. G2 also currently completes all15 training arms before its control summary; it should check raw/oracle acquisition controls before training12 downstream panel arms, consistent with a stop-on-failed-prerequisite queue.

The G2 verifier validates step numbers and checkpoint-file presence but does not replay logged batch IDs against each step's schedule, verify loss-component arithmetic, deserialize every method adapter/optimizer, or establish independent optimizer resets. It loads raw and transition adapters for their identity control. Improve these checks prospectively before spend, without claiming they already audit all training states.

G2 accuracy "noninferiority" is a guard on the **point estimate**, not a formal confidence-bound noninferiority test. Confirmation's positive NLL bootstrap interval is the inferential component. All primary comparisons must pass, so separate unadjusted intervals do not automatically inflate the joint intersection-union decision's type-I error; selecting and reporting only successful individual comparisons would create multiplicity problems. Power of the conjunction is not established by simulating one marginal comparison.

## PUPPET and SLIFT limitations

PUPPET's seven DEV queries are the clustering unit in its saved bootstrap, which is preferable to72 independent rows but remains a very small cluster count. Leave-one-query-out predictions share most fitted training data; bootstrapping the saved predictions does not refit the reader, and does not include all model-fitting variation. The six-user-turn inclusion rule conditions on conversation length. A capable reader's ability to predict a final belief survey remains an observational predictive result, not identification of assistant-caused persistent change.

The repaired code correctly binds each `AuditRecord` to its own pre/delta tuple. However, `records_from_rows` parses targets and texts for all eligible queries before callers select DEV, so the statement that its code literally never reads confirmation outcomes is incorrect. It may still be true that no confirmation predictions or model selections occurred; no such capable endpoint was found. Original mixed-file reserved source rows were not reopened here; repaired raw-source target correspondence remains a coverage limit beyond saved numerical arithmetic.

SLIFT preparation safely emits only task/logged-response/immediate-feedback and ID; old/delayed/world metadata are excluded. Official/raw, all-FIX, and all-SPEC role sensitivity do not supply the same delayed information as the correction. A fair method claim needs a delayed-information-matched simple/published baseline. The role/target preflight, effective training recipe, official result schema, cost accounting, and actual model evaluation remain unrun. Input-schema acceptance is not baseline performance.

## Scientific disposition

The historical empirical record supplies apparatus lessons and narrow negatives; it does not supply a valid capable-model causal-correction positive. Keep automatic-polarization/shared-LoRA headline parked. The remaining defensible empirical target is whether sparse delayed observations improve a declared persistent-target learning objective beyond a strong same-label baseline, under qualified semantics and a fixed simple update rule.

Continue the identifiability project only conditionally on the independent theory/novelty/G0-G1 audit. Treat G2 as constructed known-user task transfer, PUPPET as independent predictive measurement qualification, and SLIFT as a necessary baseline family with an information-budget caveat. None can be represented as human causal validation. Existing source/verifier lock-language, control, and power defects should be corrected in the next prospective protocol before any expensive queue, and final paper claims must use the narrowest supported estimands.
