# Feedback-context calibration: novelty and falsifier

Research only. No code, model call, or new experiment launched.

## PI decision

**No-go as a new paper based only on replacing the short-prompt denominator by a matched neutral-follow-up teacher.** A narrowly scoped diagnosis could be useful for our implementation, but the proposed subtraction is already in a crowded contrastive-distillation family. I did not find the exact neutral-user-follow-up recipe in the papers inspected; that limited negative search is not evidence of sufficient novelty.

## Direct primary overlap

- [Privileged Likelihood Is Not Automatically Value, arXiv:2608.09263](https://arxiv.org/html/2608.09263v1), Eq. (8), explicitly subtracts two teacher-context log probabilities after cancelling the student denominator. Sections 4.3 and A.10 discuss matched templates, style/format confounds, template-only controls and identical-feedback nulls. Its contrast is helpful versus incorrect feedback, not our proposed neutral follow-up. Importantly, A.10.2 explains why negative likelihood on observed correct paths does not alone show reduced total probability of correctness. Our reported colon score is a diagnostic symptom, not that global conclusion.
- [CEPO, arXiv:2605.19436](https://arxiv.org/html/2605.19436v1), Eq. (5), replaces teacher/student evidence with a positive-teacher/negative-teacher contrast. Unlike a free signed update, its weight modulates a verifier-anchored advantage. [Official code](https://github.com/ahmedheakl/CEPO) is linked by the paper. Cancelling a student prior to remove shared nonspecific evidence is therefore not novel.
- [CROP, arXiv:2608.13387v2](https://arxiv.org/html/2608.13387v2), calibrates task-changing context sensitivity by meaning-preserving paraphrase sensitivity on fixed rollout prefixes. It selects tokens rather than replacing the teacher advantage. This is a close precedent for subtracting nuisance context effects, though not an exact duplicate objective.
- [GC-OPD, arXiv:2608.19181](https://arxiv.org/html/2608.19181v1), calibrates group-relative teacher scores against verifier rewards; [official code](https://github.com/SolereZhang/GC-OPD). That is a different calibration axis and not the exact neutral-context baseline.
- [Contextual calibration, ICML 2021](https://proceedings.mlr.press/v139/zhao21c.html), estimates prompt biases using content-free inputs. It is inference calibration, not feedback learning, but makes the broad neutral-context-normalization idea old.

These citations establish proximity, not that all cited empirical or theoretical claims have been independently reproduced.

## What the subtraction actually asserts

For a fixed scored prefix s and candidate token v, proposed credit is

    d(v) = log q(v | s, actual feedback) - log q(v | s, neutral feedback).

This is token-dependent. It is not an action-independent REINFORCE baseline and generally changes the objective. Removing the student denominator algebraically does not prove alignment with correctness, calibrated reward, or a causal preference effect.

Shared nuisance cancellation requires an additional model: for example, both teacher log distributions contain the same token-dependent nuisance term, while only the actual-feedback teacher contains the desired semantic term. Under that assumption their difference removes the nuisance up to normalization constants. Matching length, roles and wrapper text does not establish that assumption in a nonlinear transformer. A neutral message can also alter task interpretation or continuation style.

## Small diagnostic that would falsify the mechanism

Before training, freeze a held-out set of already gradeable correct and incorrect responses. Include multiple substantive error types, not only punctuation. Rescore the *same* response tokens under:

1. Original short prompt.
2. Actual feedback in the current wrapper.
3. At least three prespecified neutral follow-ups with the same roles and wrapper.
4. A response-retaining hindsight teacher.
5. Actual-feedback paraphrases and an identical-feedback null.

Check token IDs, positions, response boundaries, truncation and exact scoring spans first. The unchanged-feedback null must produce zero contrast within numerical tolerance. Teacher forced-choice correction and preservation checks must succeed before any learning interpretation.

Primary mechanistic requirement: the contrast must retain corrective direction on wrong semantic decisions while removing harmful generic-context shifts on already-correct behavior, consistently across neutral templates. Inspect task-verifier outcomes or the entire finite correct-answer set where enumerable; do not infer correctness mass from a single colon log probability.

Falsify the mechanism if neutral choices flip the semantic direction, if correction vanishes with the nuisance, or if response-retaining prompting alone fixes the effect. A near-zero update on every example is also a failure, not a successful correction.

## Mandatory learning comparators if separately approved

Only after that preflight: no adaptation; ordinary feedback SDPO; skip all approval updates; matched smaller learning rate; full-distribution KL regularization with matched update budget; response-retaining teacher; and the neutral contrast. Measure held-out correction gains *and* preservation, across feedback paraphrases and another neutral template. Match gradient/update scale and account for the extra teacher pass.

Kill the method claim if it merely ties skip-on-approval, loses useful correction, or its benefit disappears under lower learning rate/full KL. A successful two-hour pilot would establish an implementation-level intervention worth auditing, not distinguish this project sufficiently from the papers above. I would not reserve that GPU time as a new paper gate without a sharper claim or a previously unreported failure regime surviving these simple controls.
