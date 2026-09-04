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
