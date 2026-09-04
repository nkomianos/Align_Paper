# SDPOv3: qualified generation, unsuccessful shared-user adaptation

Finished September4 in368.96seconds with exit0 and64 online updates.
`COMPLETE_POSITIVE_CONTROL` is the runner's completion label; the actual
`positive_control` boolean is **false**. Do not mistake completion for a pass.

## Frozen results

Explicit calibration:32/32 format-and-content success. Hindsight generation:
24/32 success,32/32 factual correctness,24/24 recovery among initially wrong
formats. Previously correct cases can regress under generic approval.

Held-out same-user evaluation:

| Metric | Before | After |
|---|---:|---:|
| Exact preferred format and correct facts | 16/64 | 0/64 |
| Correct facts | 61/64 | 64/64 |

The two plain-format users each lose8 successes; other users remain at0.
Paired user-cluster descriptive interval for the25-point decline is
[-62.5,0] percentage points. There are eight user clusters, not64 independent
human participants. Examples after training use a compact single-line
`asset = ...; zone = ....` layout that fails the unchanged strict grammar.
This observation does not authorize rescoring the frozen experiment.

All288 adapter tensors were inspected; squared parameter change4.2525357.
The final optimizer contains64 verified steps. Training occurred; failure is
not explained by an unchanged adapter. Further read-only gradient/bridge
diagnosis is pending. No further parser or calibration repair is queued.

## Interpretation

This shared adapter had to learn eight arbitrary user-ID/style associations
from eight updates per user. That additional routing burden differs from the
published single-profile online personalization setting. Also, this uses the
released simple-signal loss variant, not the launcher's default full distillation.
See [scope audit](SDPO_SHARED_USER_SCOPE_AUDIT_20260904.md).

The result fails to qualify this specific adaptation apparatus. It does not
refute SDPO, demonstrate preference shaping, or kill the original causal
question. Do not queue noisy/polite feedback arms on this unsuccessful control.
Even a successful control would not alone establish a novel paper contribution.

## Preservation and verification

Complete source, outputs, initial/final adapters, intermediate checkpoints,
optimizer states and logs are local under
`retrieved/sdpo_v3_20260904T0945Z`.
Remote/local archive SHA256:
`35ded5dd872b254e7e1c15b4f0c0e0a8167422fffb23302e2d14bc631a411486`.
Run manifest SHA256:
`3765136f5452a3831c9a5620938a8b20bd28d0ad18c59465179b5f2bebc0fd41`.

The committed v3 verifier checked34 files, native prompts/decoding, frozen
apparatus/checker, recorded losses, schedule, metrics, checkpoints and optimizer
metadata. No neural-forward, optimizer-update or RNG replay is claimed.
Large base weights were not locally rehashed by this verifier; tokenizer
metadata were checked against the previously retrieved pinned4B files.

At the post-run check there were no active GPU compute processes. The completed
experiments' unique evidence is secured; no instance termination was performed.
