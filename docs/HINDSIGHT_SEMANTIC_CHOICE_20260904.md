# Fresh constrained semantic-choice qualification

Prospective CPU-only protocol. No training or paper go decision. This changes
the action interface explicitly; it does not repair or rerun the earlier gate.

The deployed policy is a two-candidate likelihood classifier: choose between
the option texts, not A/B labels. For each candidate, sum the autoregressive
log probabilities of its complete token string after the assistant prefix;
normalize the two sequence likelihoods. No length averaging, added EOS, prefix
forcing, vocabulary-top-k, or generated-response parser. Require prefix-free
candidate token sequences. Save token lengths and unconstrained prefix mass;
this mass need not be high because the classifier is explicitly constrained.
Do not call its accuracy free-form generation accuracy or full SDPO.

Use8 new domains,4 development and4 confirmation. Each crosses2 option orders,
2 wordings and2 possible true preferences:32 cases per split. Confirmation
domains differ from development; wordings are shared. No choices are made
between evaluating the two splits. All cases are retained even if development
fails. The model is frozen Qwen3-0.6B at the same pinned revision.

Score32 unique base prompts without feedback,64 direct truthful-feedback
prompts,64 truthful-feedback prompts using the pinned published hindsight
block:160 prompts,320 candidate-sequence forwards. Direct feedback controls
semantic interpretation without the hindsight wrapper. Baseline has no hidden
preference and is not expected to identify one. Both target preferences and
orders prevent an always-first or favorite-option rule from qualifying.

Require >=90% accuracy in each split for **both** direct and hindsight
conditions, and >=75% in every domain. Also report changes in true-candidate
probability versus base; this is a diagnostic, not an adjustable threshold.
No intermediate selection or automatic training. A pass qualifies only this
constrained teacher interface, not endogenous-feedback harm, an anchor-based
correction, persistent preference changes or publication novelty.

## Verified result

Frozen code `93b47c3`. Root `artifacts/hindsight_semantic_choice_cpu_20260904_v1`.
160 prompts/320 candidate-sequence forwards completed in143.95seconds CPU,
with zero parameter updates. The verifier checked all6 manifest files, every
prompt/candidate tokenization, per-token-to-sequence likelihood sums,
normalizations, full crossed coverage, summary arithmetic and the stopping
decision. Receipt: adjacent `_verified.json`. Not an independent neural replay.

| Condition | Development (32) | New-domain confirmation (32) |
|---|---:|---:|
| Direct preference feedback |26|30|
| Published hindsight wrapper |32|31|

Hindsight mean correct-option probabilities are.9763 and.9629 respectively.
The correct-option probability increases over no-feedback baseline on29/32
development and31/32 confirmation cases. Baseline mean correct-option probability
is exactly.5 by the paired-target construction, not empirical calibration.

**The hindsight arm passes its own accuracy and per-domain thresholds.** Every
development domain scores8/8; confirmation domains score8/8 except writing
tool7/8. This is a genuine positive teacher-interpretation result for the declared
constrained interface. Cases are correlated crossings of8 domains, not64
independent populations or a multi-family replication.

The frozen joint status is `CONSTRAINED_TEACHER_UNQUALIFIED` because direct
feedback development26/32 is below29/32 required. Keep that status unchanged.
Its name refers to the joint requirement, not a claim that the hindsight arm
failed. Do not erase the positive arm or lower the control threshold post hoc.
Nor does the direct arm establish that a directly supervised learner would fail:
no parameter learning occurred in either condition.

Candidate token lengths range1–3. Minimum unconstrained candidate-prefix mass
across conditions is1.19e-10. Therefore these probabilities are **conditional
classifier probabilities**, not high likelihood of emitting the option text in
free-form generation. This limitation was part of the prospective action-space
definition; the result must not be promoted to free-form instruction compliance.

PI action: retain this teacher interface as a candidate for a newly declared
learning assay; do not resume the old report-copying sweep automatically. The
[longitudinal intervention design](HINDSIGHT_LONGITUDINAL_NEXT_20260904.md)
describes what is needed to test the actual causal thesis. Still missing:
truthful neural acquisition with this interface, a meaningful persistent-state
contrast, and an identifiable correction that beats same-anchor baselines.

Reproduce the audit with a fresh output path outside the evidence root:

```powershell
$env:PYTHONPATH='src;.'
python scripts/verify_hindsight_semantic_choice.py --root artifacts/hindsight_semantic_choice_cpu_20260904_v1 --out artifacts/hindsight_semantic_choice_new_audit.json
python -m pytest tests/test_hindsight_semantic_choice.py -q
```
