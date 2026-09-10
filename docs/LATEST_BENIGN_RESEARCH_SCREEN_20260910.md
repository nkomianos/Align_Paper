# Latest benign research screen

Revisited the user's requested DAIR AI-Papers-of-the-Week repository. Its newest
listed issue is August31–September6. A pinned snapshot and hash receipt are saved
in artifacts/dvpo_source_audit_20260910/dair_2026.md and DAIR_RECEIPT.json.
The newsletter is used to locate papers, not to validate their scientific claims.

Two relevant primary abstracts were read independently:

- [Trace as State](https://arxiv.org/abs/2609.02702), September2: puts collected
  reasoning before context on a fresh pass and compares the same trace appended
  after context. Generic trace-prepending is already the method being proposed.
- [Language Models Can Control Their Own Attention](https://arxiv.org/abs/2609.02737),
  September2: declarative global/focus/local attention modes reduce attended
  tokens, with an accuracy tradeoff, on released models. Training-based extension
  is explicitly identified as future work. This is not automatic novelty clearance
  for training a router or adding selective context retrieval.

Possible unresolved question, not a selected paper: when an intermediate state
is incomplete or wrong, does exposing it before rereading improve correction or
entrench the error? A useful study would need naturally generated erroneous states,
matched rereading and question-first controls, full source-code/protocol review,
and improvement beyond verification or simply removing incorrect traces. Artificial
false hints alone would repeat generic feedback susceptibility experiments.

No new experiment is admitted by abstract inspection. The next bounded action is
to inspect the full Trace as State method and error analysis to determine whether
this question is already addressed and whether a specific corrective intervention
can be distinguished from existing self-correction baselines. Do not describe this
question as a newly discovered phenomenon or claim open-ended novelty.

## Trace as State method/control review completed

Read the primary paper's method, setup, placement/count ablations, limitations,
scoring appendix and serializers. The authors already warn that traces can be
incorrect and explicitly instruct source verification. Question-first, repeated
context, answer feedback, random traces and trace-only controls are present.
No item-level error-transition analysis was identified in these sections. This
does not establish its absence from all author-held data or later versions.
The paper uses five source traces and five second-pass repeats per problem.

For Parents, reported Oracle@5 EM is50% and Trace as State EM81.8%. Under the
stated shared-cohort weighting, these marginals imply at least31.8 percentage
points of second-pass successes on problems with no correct first-pass answer.
Thus merely demonstrating some recovery is not a differentiated contribution.
Wrong source answers need not mean all their intermediate reasoning is wrong.

Independent arithmetic enumerated92 feasible joint tables using those reported
marginals: conditional recovery among source-oracle failures can range63.6–100%,
while losing correctness among source-oracle successes can range0–36.4%. These
are elementary sharp bounds, not confidence intervals or a raw-data replication.
The500 trial slots represent100 questions with repeats, not500 independent tasks.
See scripts/audit_trace_recovery_bounds.py and
artifacts/trace_state_audit_20260910/RECOVERY_BOUNDS.json. All consistency checks
passed; versioned paper HTML and hash receipt are saved beside the output.

Decision: no launch for generic verification warnings, trace-prepending or a
demonstration that wrong answers sometimes recover. A successor requires a
specific corrective intervention and a within-problem causal design separating
useful partial state from erroneous commitments. Neither is presently specified.
An outcome-selected subset of naturally wrong traces can support a conditional
diagnostic, but cannot estimate overall utility or causal effects of correctness
without additional design. Preserve this boundary rather than filling the GPU
with a new version of the earlier feedback assay.
