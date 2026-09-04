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
`artifacts/hindsight_endo_pahf_g2_power_20260904_v2`; `MANIFEST.json` SHA-256:
`663df8a48cc0e359be7256afd8ad7d3df57da0012cea409aa39480a149789b8c`.
The identical v1 numerical result is preserved but superseded because the final
transition-integrity check changed the hashed G2 source after v1 was written;
no threshold, random seed or simulated outcome changed.

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
