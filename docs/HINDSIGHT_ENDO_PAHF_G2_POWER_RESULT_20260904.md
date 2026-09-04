# EndoPAHF G2 Development-Routing Power Result

Status: **qualified routing rule; not a neural result or paper green light.**

The rule was frozen at commit `aa30afd` before execution. Development uses 96
independent PAHF base tasks after averaging each task's four label rotations.
Because development only decides whether to spend the locked confirmation, it
uses a permissive mean paired NLL-gain threshold of `.03` without a confidence-
interval gate. The 256-base confirmation retains the separately frozen `.05`
effect and positive cluster-bootstrap lower-bound requirements.

The read-only verifier returned
`ENDO_PAHF_G2_DEV_ROUTING_RULE_POWER_QUALIFIED`. Evidence root:
`artifacts/hindsight_endo_pahf_g2_power_20260904_v1`; `MANIFEST.json` SHA-256:
`175b3ac2a5d8b35cf7e3959d393be9f0ce943b80913ec0b2088a5accf3048a64`.

Across 10,000 simulated development studies per condition:

- null routing rate: `.2041` (required at most `.25`);
- planned-signal routing power: `.9196` for mean `.08`, SD `.35` (required at
  least `.90`);
- noisy-signal routing power: `.8317` for mean `.08`, SD `.50` (required at
  least `.75`).

The 20.41% null routing rate is intentional and cannot be interpreted as a
final false-positive rate. It spends confirmation on some ambiguous DEV
outcomes instead of killing a useful effect prematurely. The already-qualified
confirmation rule controls the final decision.

