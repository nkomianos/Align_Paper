# Recency G0: verified final result

## Decision

**Kill the recency-gated policy-switching candidate.**  This is not a
threshold-tuning decision.  The completed frozen run failed every preregistered
check for both seeds, and the independent read-only matched-cue audit confirms
that the missing mechanism is not rescued by correcting G0's original
intervention comparison.

## Immutable evidence

- Frozen remote run: `recency_g0_20260828T0432Z`
- Frozen source commit: `44d2aae78ee9d70d147cff2668303a74d8ce7808`
- Retrieved run-manifest SHA-256:
  `d165d6dd4e1f7f10393e99adbfb5cf07024d2e89bb72a13f200ae855cd939ed9`
- Retrieved metrics SHA-256:
  `fa657445ddd4ad614c7d13fe4ea5843f30e78e5e4e1e264ac01ccdac347baeee`
- Retrieved gate-report SHA-256:
  `ac60bf971591cca07b411c8f4cf41ecf7ee8199577b1b841ab5fe0ffa3e7fee8`
- Every saved adapter/checkpoint artifact was checksum-verified (12 artifacts
  for each seed).

The frozen gate returned `KILL_CANDIDATE`; both seeds failed readout,
cue-subtracted switching, steering mediation, matched-control specificity,
erasure necessity, and temporal homogenization.

## Corrected read-only audit

G0's original steering and erasure comparisons did not apply the intervention
to the cue-only control.  A later, offline and non-destructive audit applied
the same intervention to baseline and cue-only adapters.  Its evidence is
bound to the manifest above and has SHA-256
`68993beb5b14559f8bd2938c7fda268fc86d8d0da3a20d7519341387b13cefd8`.

| Seed | Cue-matched switch difference | Corrected steering contrast | Corrected erasure reduction | Consequence |
| --- | ---: | ---: | ---: | --- |
| 9101 | -0.072 pp | 0.147 pp | fail-closed (-1.0) | no behavioral effect |
| 9102 | 0.704 pp | 0.096 pp | 47.9%, but lower bound -100.7% and controls 73–93% | no specific causal effect |

Both seeds fail the registered descriptive mediation and necessity conditions.
The audit is explicitly non-retroactive; even a positive result could only
have informed a new preregistration.  Here it is also decisively negative.

## PI consequence

Do not run G1 or publish a recency-mechanism claim.  Preserve the full remote
and local evidence for reproducibility, but spend no additional GPU time on
this hypothesis.  The next experiment is the independent recipe-invariance
J0 gate, not a revision of this closed candidate.
