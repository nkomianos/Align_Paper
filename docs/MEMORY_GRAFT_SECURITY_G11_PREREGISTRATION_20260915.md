# G11 preregistration: frequency-aware conditional-memory writes

**Pre-execution state.** No G11 weight has been loaded and no G11 forward or
optimizer step has run. The implementation, configuration, source checkpoints,
and this protocol are hash-bound before execution.

## Question

G6 showed that raising the memory-table learning rate from `5e-5` to `1e-1`
moves a learned mapping from the backbone into the table, but increases clean
perplexity by about 7--8%. G11 asks whether the relevant mechanism is update
size alone or allocation of a fixed update budget across rows with different
access frequencies. A positive result would supply an ordinary-gradient write
policy rather than only a warning or a direct-write API.

The experiment reuses the validated retrofitted Pythia apparatus, five sealed
clean checkpoints per size, the same 64 poison exposures, 512 optimizer steps,
and the same causal transplants and deletions. It does not depend on jointly
pretrained Mini-Engram checkpoints.

## Six fixed policies

At 410M and 1.4B, every seed runs the same paired blocks and dropout RNG under:

1. ordinary AdamW at table rate `5e-5`;
2. the known routing reference, ordinary AdamW at table rate `1e-1`;
3. active-row AdamW-direction updates at `5e-5`, without frequency weighting;
4. the same active-row update with inverse cumulative-hit weighting;
5. the unweighted active-row update rescaled each step to the paired `1e-1`
   reference's complete table-update L2 norm;
6. the inverse-frequency update under the identical stepwise norm budget.

For a packed table row with cumulative access count `c_r`, the frequency arm
uses weight `1/c_r`. We divide these weights by their RMS among rows active in
that optimizer step. This keeps the scalar nominal rate fixed while reallocating
it. The norm-matched arm then rescales the complete active-row delta to the
paired high-rate reference norm at the same seed and step. Thus the two
mechanistic contrasts are frequency allocation at equal nominal rate and at
equal cumulative update path length. No clipping, exponent search, LR search,
or post-hoc profile selection is allowed.

## Derived decisions

The downstream localization effect is 0.15 ASR. With 1,024 paired binary
evaluation contexts, the registered row-level uncertainty allowance is
`sqrt(2 ln(2/0.05)/1024) = 0.084881`; therefore a cell must install at least
`0.15 + 0.084881 = 0.234881` before its localization is interpretable. This is
the sole blocking sanity bar. Every cell remains reported if it fails.

A seed selects the table route only when installation reaches 0.234881 and
both whole-table necessity and whole-table sufficiency exceed 0.15. A policy is
"typical" only if the two-sided 95% Wilson lower bound on five-seed prevalence
exceeds one half. With five seeds this requires 5/5; this bar derives from the
word *typical*, not from apparatus convenience.

The mechanism estimand is the paired frequency-policy minus active-row-control
contrast in whole-table necessity and sufficiency. A lower five-seed 95%
Student-t endpoint above +0.15 supports improved table routing at the fixed
budget; an upper endpoint below -0.15 supports impairment; intermediate values
are unresolved at this scale. Each conjunct is reported separately. Neither
blocks the other.

The constructive quality estimand is paired clean-NLL cost reduction versus
the `1e-1` reference. A lower 95% endpoint above zero supports a real quality
improvement. We also report perplexity ratios and comparison with ordinary
AdamW. Failure to reject a quality difference does not license a "free" or
quality-neutral claim.

Target-row necessity/sufficiency remain outcomes. They determine whether a
policy creates item-level deletion locality; they are not a gate on the
component-routing claim.

## Scale and verification

Five independent seeds are the minimum for a unanimous route to have a Wilson
lower bound above 0.5. Six policies separate the two factors without a tuning
stage, across both sizes where G6 established the scale-dependent transition.
The projected source plus exact replay cost is 14 GPU-hours. Full predictions,
per-step losses, active-row counts, hit ranges, update norms, endpoint table
displacements, causal readings, and clean quality are retained and replayed.
