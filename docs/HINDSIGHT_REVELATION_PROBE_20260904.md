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
