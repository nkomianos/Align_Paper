# Candidate screen after the masked-value null

Decision: no new GPU launch admitted from this screen. The AWS instance was
reachable and reported zero compute processes, 0 MiB GPU allocation and 0%
utilization during this turn. GH200 was not contacted. This is a point-in-time
observation, not a promise of continued monitoring or a provider billing total.

## Released self-distillation checkpoint

The existing u-OPSD 8B adapter remains technically available and structurally
verified. Its behavioral evaluation has never run here. The absence of that run
is not a negative result. A generic base-versus-adapter accuracy comparison would
be useful replication but would not itself answer a differentiated paper question.
The author [project page](https://williamium3000.github.io/u-opsd/) already reports
math evaluations across thinking and non-thinking settings. Broader thinking
degradation is also explicitly studied in [Thinking Collapse](https://arxiv.org/abs/2607.10805)
and [Purified OPSD](https://arxiv.org/abs/2607.02234). These sources were rechecked;
their claims were not independently reproduced this turn.

Do not relabel a small accuracy pilot as a new research contribution, or infer
that an available adapter authenticates its historical base revision. The next
admission still requires an outcome-level question beyond generic degradation,
reference bias, or uncertainty suppression. No such question was established
by this screen, so no adapter transfer or generation campaign was launched.

## Mechanistic extension of the memory ID result

New close primary source: [Temporal Token Matters, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.123.pdf).
Sections 2–4 already study before/after answer consistency, path-patch attention
heads, and intervene on temporal and structural attention patterns. Its explicit
limitations distinguish simple binary relations from implicit or multi-hop
dependencies. Consequently, identifying temporal heads and improving a simple
ordering assay would substantially overlap this contribution. Reading scope:
task definition, head-identification method, intervention overview and limitations;
not a complete empirical or source-code audit.

The [University of Vienna primary record for Do Language Models Track Entities
Across State Changes?](https://eprints.cs.univie.ac.at/8738/) also describes
mechanistic analysis of state updates and a suppression-tag intervention.
Only the institutional abstract and publication metadata were inspected; no
full-paper or code-verification claim is made.

Neither source establishes the exact cause of our Qwen3 ID-only degradation.
Event labels, presentation positions, textual temporal relations and latent
binding are distinct variables. Our 24/24 to 7/24 exact-graph result remains
a behavioral relabeling effect, not an identified arithmetic or attention
mechanism. A mechanistic extension would need to separate those variables,
predict new failures, and demonstrate a useful intervention on fresh tasks.
Simply running activation patching on the already exposed 24 cases does not
meet that standard.

## Allocation consequence

This screen adds a concrete novelty constraint, not a new positive result.
It does not establish that all possible research ideas are exhausted. It rules
out two generic experiments as adequate next steps toward this paper. No model
failure, checkpoint failure, or publication readiness is inferred from a decision
not to launch. The submission remains NO-GO, and the actual paper goal remains
incomplete. Further exploration must supply a specific testable contribution,
not just a different model, benchmark, or visualization of the same effect.

## Additional screen: single-answer gains versus repeated-attempt coverage

The proposed outcome-level question was whether the available adapter improves
single-answer accuracy while reducing correct-solution coverage across repeated
attempts. Primary-source checking found direct overlap, not merely related
terminology:

- [Self-Distillation with Sampled Demonstrations Reduces Output Diversity](https://arxiv.org/html/2606.26091v1)
  explicitly studies stronger average performance alongside flatter pass@k and
  reduced semantic/functional diversity. Its abstract and introduction describe
  both controlled graph tasks and science QA. We did not reproduce its runs.
- [Understanding OPD through Test-Time Scaling](https://arxiv.org/html/2608.11829v1)
  studies crossing pass@k curves across sampling budgets and problem-level
  transitions at 1,024 samples. We inspected its setup, metric definitions and
  Appendix E. This is a direct precedent for the proposed accuracy/coverage
  comparison, although it does not authenticate behavior of our u-OPSD adapter.
- [Influence-Directed Distillation](https://arxiv.org/html/2608.29846v1)
  proposes a sampled-token intervention for the diversity tradeoff. We inspected
  its abstract and introduction; neither the method nor its claimed gains were
  independently validated here. This rules out treating a generic diversity
  preservation proposal as an unexplored remedy.

A possible statistical pivot also has close prior art:
[Beyond Pass@k](https://arxiv.org/html/2510.08325v1), particularly Sections 4–5,
defines task coverage at a success-probability threshold and relates it to
pass@k by an integral identity. Its large-budget limit discussion already
distinguishes nonzero success probability from reliable performance. We must
not claim that distinction or the coverage-threshold idea as a new theorem.

Finite sampling still requires care: observing zero successes is not proof of
zero success probability. Conversely, the existence of this qualification does
not invalidate measured finite-budget pass@k comparisons. A critique must
distinguish the operational finite-budget result from any stronger interpretation
about capability. No author result was numerically refuted by this source screen.

Decision: do not launch a base/adapter repeated-sampling campaign on this generic
thesis. It would currently be replication, with no differentiated intervention
or independently supported new scientific claim. The adapter remains untested,
not failed. No new model generations, GPU costs, or paper-readiness claim result
from this additional screen.
