# Hindsight preference-revelation probe — before execution

The previous numerical-choice learner failed to generalize its acquired
competence. Formatting added another unnecessary failure mode. This fresh probe
instead asks whether the frozen teacher can use an explicit newly revealed
preference when that preference was absent from the original request.

Eight ordinary preference domains, both A/B orders, both previous actions, both
true preferences: 16 base forwards, 64 truthful-teacher forwards, 32 polite
endorsements, 32 noninformative thanks. All 144 predictions retained. No subjective
LLM judge, sampling, fine-tuning, participant data, or GPU is used. Cached pinned
Qwen3-0.6B on CPU, float32, native nonthinking template, no truncation.

Positive control: does explicit feedback cause selection of the stated preference
even when the previous assistant action disagrees? Negative control: how much
does generic thanks alone preserve the previous action? Endorsement contrasts
with this null to distinguish ordinary history anchoring from endorsement.

The simulated person's latent preference does not change. Polite endorsement
could be generated regardless of it; this investigates expression confounding,
not real persuasion. Reporting an endorsed action is not itself a model error
when the input offers no independent evidence of insincerity. Harm must later be
tested against known latent objectives in actual adaptation, including truthful
learning, no-adaptation, and same-budget direct-preference baselines.

This is a precursor to that learning test, not a substitute. Report all scores
and probability mass, including order effects. Do not silently filter weak
domains or call this a full SDPO reproduction. No automatic expansion or paper
decision follows from success on these easy examples.

## Completed CPU evidence

Frozen scientific design commit `fe08a6b`; explicit tokenizer return contract
repair `11f3c3b`. Attempt v1 stopped before its first forward because the newer
tokenizer returned a mapping rather than token IDs. Its failed root is preserved.
Attempt v2 completed all 144 forwards and saved `scores.json`, then failed to
serialize a NumPy integer into the summary. The same complete rows were saved
again as `partial_scores.json`; both have SHA-256
`8579416b911679aa688f4bae60f6d8b78ef2cd54fddfe7264cbd78e7048ccba4`.

No inference was repeated or scientific output edited. The explicit recovery
verifier checks the known failure, equality of both score files, design counts,
unique keys, valid probabilities, and recomputes descriptive statistics. It
records hashes of all original files in a separate receipt. This is recovery of
an unsealed run, **not** verification against a successfully emitted runner
manifest, nor neural replay. The truncated RESULT and FAILED files remain intact.

- Truthful preference selection: **56/64 (87.5%)**.
- When truth contradicts the prior assistant action: **25/32 (78.125%)**.
- All eight domains retained, with 6–8 correct out of eight each.
- Minimum unrestricted-vocabulary A/B mass: **0.9999885**.
- Probability assigned to prior action under endorsement: **0.61954**.
- Same probability under generic thanks: **0.67080**.
- Paired endorsement-minus-thanks difference: **−0.05125**.

These option permutations are dependent cases, not 64 independent human users.
The study establishes substantial use of explicit preference information but
also errors when revising a previous answer. It does **not** support an excess
reinforcement effect of endorsement relative to the noninformative control.
Previous-action text itself may influence both arms. That explanation needs a
paired action-history removal control, not an assumption that feedback caused it.
No preference transition, trained-adapter harm, corrective-method gain or full
SDPO result has been measured. This narrows the next diagnostic to whether the
teacher distinguishes original assistant action from actual user information.

Evidence: `artifacts/hindsight_revelation_cpu_20260904_v2/` and sibling
`hindsight_revelation_cpu_20260904_v2_recovered.json`. All processes have exited.
