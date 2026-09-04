# EndoPAHF Cluster-Rule Power Audit Result

Status: **qualified decision rule; not a model result or paper green light.**

The prospective EndoPAHF confirmation rule was frozen at commit `11e2970`
before this simulation was executed. The four cyclic option rotations are
averaged within each PAHF base task, so confirmation inference uses 256 paired
base clusters rather than pretending that 1,024 rotations are independent.

## Verified result

The read-only verifier returned
`ENDO_PAHF_CLUSTER_RULE_POWER_QUALIFIED`. The evidence root is
`artifacts/hindsight_endo_pahf_cluster_power_20260904_v1`; its
`MANIFEST.json` SHA-256 is
`7a13b16c1a1acdf02d43efe30ba7830a519fcd7b1ce271c17a741891da9fe48e`.

Across 400 simulated confirmation studies per condition:

- null qualification rate: `.0075` (required at most `.06`);
- signal qualification rate: `.9025` for mean paired NLL gain `.08` and SD
  `.35` (required at least `.85`);
- noisy-signal qualification rate: `.7450` for mean paired NLL gain `.08` and
  SD `.50` (required at least `.60`).

All three frozen gates passed. The eventual augmented method must still have a
mean old-target NLL gain of at least `.05`, a positive lower endpoint from a
10,000-resample paired base-cluster bootstrap, and old-target accuracy
noninferior within `.02` relative to the equal-anchor baseline.

## PI interpretation

This removes a pseudoreplication risk and shows that the reserved 256-base
confirmation split has useful prospective power for the declared effect size.
It supplies no evidence that Qwen exhibits the effect. The next scientific gate
remains Qwen3.5-9B gradient G0 v2; policy G1 is conditional on G0, and the
capable EndoPAHF preflight/external assay remain downstream of both.

