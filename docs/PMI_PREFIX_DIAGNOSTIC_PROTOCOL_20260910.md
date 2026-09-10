# Frozen PMI target diagnostic on exposed math prefixes

This is a forward-only apparatus and attribution diagnostic, not a replication
of thinking-model collapse or a new training result. The preceding exact null
model established a question-only sharpening effect. This run asks whether the
full-reference target is close to that control on actual model distributions.
Target difference alone does not measure usefulness or transferable reasoning.

Data: all 48 saved prefixes from 24 MATH development questions, each restricted
to its first 32 generated token IDs. No rows were excluded. The original source
JSON provides full reference solutions and matches the bank's question strings.
The bank manifest and source hashes were checked before preparation. These
questions and outcomes are already exposed development data. Two prefixes per
question are not independent task replications.

Model: the original Qwen3-8B pinned revision
`b968826d9c46dd6066d109eabc6255188de91218`, using native nonthinking chat.
Authenticate downloaded weight files against the generating model's hashes and
require the tokenizer to reconstruct every saved full prefix text. Then append
exactly the same 32 token IDs under four contexts: question+reference,
reference-only, question-only and neither. No rollout sampling or optimizer.

Compute full-vocabulary purified and question-only target distributions with
beta=1 and c=10. Save raw float32 logits, rendered contexts, input IDs, per-prefix
KL/TV and entropy. An independent verifier checks the manifest, equal prefix
tails and metric reconstruction; aggregate TV by question. The control uses
question-only versus unconditional logits, not a fitted temperature chosen after
seeing these results. Prompt framing differs between contexts by design and can
affect outputs independently of semantic information; this is a limitation.

Estimate: 192 forward passes, 5-15 GPU minutes including load, plus download;
not a measured runtime. Admission requires an idle AWS GPU. Stop on failed
weight/tokenizer identity or nonfinite logits; do not silently substitute models
or repair inputs. The GH200 is excluded during its OSH reservation. Runtime is
an estimate, not an arbitrary midrun cutoff.

Interpretation: small differences would motivate testing whether the control
explains downstream gains; large differences would reject near-equivalence on
these prefixes but would not demonstrate useful reference information. Neither
outcome automatically admits training. A future causal study needs prospective
thinking-model traces, reference-relevance controls, development-only tuning,
held-out outcome evaluation and comparison with existing distillation remedies.

Local preparation: `artifacts/pmi_prefix_diagnostic_20260910/`.
AWS deployment: `/home/ubuntu/align_research_20260910/pmi_diagnostic/`.
Source: `run_pmi_prefix_diagnostic.py`, `pmi_target_controls.py`,
`prepare_pmi_prefix_diagnostic.py`, `verify_pmi_prefix_diagnostic.py` in `scripts/`.
Three target-transform tests passed. Deployment file hashes were verified.
