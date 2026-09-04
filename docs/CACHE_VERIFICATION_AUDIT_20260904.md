# Cached verification: local correction versus global independence

## Status and motivation

**Operator audit completed; pretrained-model effect and paper viability unproven.**

[COVER v1](https://arxiv.org/html/2602.06161v1), Section 5.1 and Appendix B,
describes masking verification seeds, retaining their cached KV for other
queries, and replacing the seed's own diagonal KV. Its single-layer softmax
correction is exact. We independently implement that local operator, both by
direct row recomputation and by its closed-form correction. No official COVER
implementation was located in this search; this is not a reproduction of its
benchmark results or a verified bug report about its code.

Our question is stronger than the layer-local identity: after composing layers,
does changing only the hidden candidate alter its verification distribution?
The local identity alone cannot guarantee the answer is no.

## Our counterexample

Use two scalar residual-attention layers on n positions. Set queries and keys
to zero (uniform attention), and value/output projections to one. Position 0
contains candidate a; all other inputs are zero. Compute its ordinary cache.
Then mask position 0 to zero and apply seed-column override with its own
diagonal corrected at each layer.

After layer one, the seed is zero, but every non-seed hidden state equals a/n.
At layer two the seed reads those states. Its final value is

`a * (n - 1) / n**2`.

A fresh masked forward gives zero at every position. With n=2, changing the
hidden candidate from -2 to +2 changes the corrected verification output from
-0.5 to +0.5, although the visible verification input is identical. The path is
candidate cache -> another position in layer one -> seed in layer two.

This witness contains no cache age across changed surrounding tokens, no stale
non-seed cache, and no incorrect softmax normalization. It is a deliberately
simple existence proof, not a realistic generative model.

## Numerical checks completed

Implementation: `src/interaction_sprint/cache_verification_audit.py`.
13 tests pass. They cover direct-versus-corrected rows, preserved non-seed
attention outputs, the analytic witness, no-change cache controls, depth-one
negative controls, deterministic replay, non-overwrite, and evidence tampering.

An independently coded direct-row normalization agrees with the closed-form
attention implementation within 1e-10 over the sweep. Fresh masked outputs are
candidate-invariant. These controls distinguish global feedback from an error
in our local correction arithmetic.

The predetermined random-network sweep uses eight weight seeds, width 16,
six positions, residual pre-normalized single-head layers, and a random
11-output readout. Only candidate position 0 changes between the two cache
constructions; every visible verification token remains identical.

| Depth | Sensitive networks / 8 | Median output total-variation change |
| --- | ---: | ---: |
| 1 | 0 | 2.60e-17 |
| 2 | 8 | .04965 |
| 4 | 8 | .14001 |
| 8 | 8 | .31059 |

Both tested seed sets, `{0}` and `{0,2}`, produce these same summaries in this
construction. They are not independent replications and must not double the
sample count. Random-readout TV is an intervention-sensitivity measurement,
**not accuracy, calibrated confidence, or evidence of downstream harm**.

## Interpretation and novelty limits

We have demonstrated that the implemented local rule does not generally imply
candidate-independent multilayer verification. We have **not** shown that a
released decoder implements exactly this composed computation, that its gains
are illusory, or that removing this dependence improves accuracy. A useful
heuristic may deliberately retain candidate information without being an exact
leave-one-out evaluator. Those are different claims.

Information-flow separation is not a new concept; XLNet and recent temporal
verification methods are mandatory priors (see the linked scout). No theorem
novelty or ICLR acceptance claim follows from this small counterexample.

## Next local diagnostic — planned, not an executing queue

1. Locate and inspect the actual relevant decoder implementation if available;
   identify exactly which hidden states are shared between layers. Otherwise
   continue to label the experiment as an independent operator implementation.
2. Use a pinned, public pretrained masked LM on explicit single-token cloze
   cases. Freeze cases before examining cache effects. Require clean-mask
   competence; retain all cases and separately report the competent subset.
3. Construct valid caches with the correct versus an incorrect candidate while
   keeping context identical. Compare fresh masking, cache override without
   correction, diagonal correction, and a separate clean verification pass.
   Verify unmodified-forward equivalence before trusting the instrumentation.
4. Measure candidate-dependent logit change, wrong-candidate retention, and
   accuracy of corrections. A meaningful practical lead needs decision changes,
   not just tiny nonzero floating-point discrepancies. CPU masked-LM evidence
   would still not establish a modern diffusion-decoder effect.
5. Only after that diagnostic, assess a modern diffusion model and a cost-matched
   clean-verification baseline. Do not claim a new cheap fix if it merely adds
   a standard full clean pass. No large sweep is currently authorized to start
   automatically, and no new GPU request is justified by this audit alone.

## Evidence and reproduction

Root: `artifacts/cache_verification_audit_v1`.
Manifest SHA-256:
`5090b532b24fbdf341138ef75891af92071d179191ca321579b0fe59e52b1621`.
Manifest binds result bytes, implementation source, and NumPy version.
Verification replays the same implementation; it is not external replication.

```powershell
$env:PYTHONPATH='src;.'
python -m pytest tests/test_cache_verification_audit.py -q
python -m interaction_sprint.cache_verification_audit verify artifacts/cache_verification_audit_v1
# For a new run, choose an absent directory; existing roots are refused.
python -m interaction_sprint.cache_verification_audit run artifacts/cache_verification_audit_new
```

No model download, parameter training, or GPU execution occurred. Every earlier
experiment and artifact remains untouched. Decision: investigate this specific
mechanism locally; **do not green-light a paper yet**.
