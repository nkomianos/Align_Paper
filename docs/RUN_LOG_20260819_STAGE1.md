# Stage 1 run log — 2026-08-19

This log records an analysis-only numerical validation repair made after the
first Qwen3.5-9B Stage 1 training and evaluation completed. It is part of the
experiment provenance and must remain disclosed in any paper or artifact
release that uses this run.

## Frozen run identity

- Source commit: `1d113fe181820c8741e73f42b2dcfcb045b17185`
- Analysis-only numerical repair commit:
  `ac2c430592de913b58f75c70f76eb9f6cd866754`
- Deployment bundle SHA-256:
  `c66e142622664811443ec61107b06400d11d0ba7351f1489a66b707b283817a0`
- Runtime project-tree SHA-256:
  `3a91523424bf4f531b543a6cd91568c13bcec7384e335c61cb864d21f2c7f0d7`
- Formal config SHA-256:
  `3d1084bfe5358343e9ad0581895e63129cad42acadcaa38713b476d2e74b5389`
- Both 300-update training arms completed. All six checkpoints per arm and all
  DEV/base-control predictions completed before analysis began. Locked TEST was
  not parsed or evaluated.

## Immutable prediction artifacts

| Artifact | SHA-256 |
|---|---|
| `genuine_seed11_dev.jsonl` | `9969af9357879197bf3aa29f773d5a750b9d7bce0edda88f76363cc8279fb9b9` |
| `genuine_seed11_dev.summary.json` | `063ee51759cf0cf80e07576bd060a5f78b8bb017f200993e4aa38edf99610499` |
| `proxy_seed11_dev.jsonl` | `f18864eb2afa0dd8aa97e01e3ac9038a373a059c01c20e4bb2028b3c2c2dfd9d` |
| `proxy_seed11_dev.summary.json` | `14541acbacc8962537f665d390fa749a79d2a6f2adccfc705ea78ec0cb9962f7` |
| `unchanged_base_seed11_dev.jsonl` | `f11f5ef4ff5ad2752da874dbb2b2eb791ed38ff9462c6c66713dd3451888ab6a` |
| `unchanged_base_seed11_dev.summary.json` | `973398cfe5acd81d1f36fbc8c234ff3db4a0abc3b2d2d9af288d63f8a9f5855b` |

These files are not modified by the repair or analysis rerun.

## Analysis failure and repair

The first analysis invocation stopped before producing a report because
`validate_bridge_predictions` required `legal_choice_mass <= 1` exactly. The
largest observed value was `1.000000057892679`, an excess of
`5.7892679095061794e-08`; conditional A/B probabilities remained inside
`[0, 1]`. The scorer already had a frozen log-space numerical allowance of
`1e-5` for BF16 roundoff, but the downstream validator did not use it.

The repair defines that existing scorer allowance once and makes downstream
validation use the same bound. It does not change predictions, scientific gate
thresholds, bootstrap settings, treatment labels, or any model output. Values
outside the pre-existing allowance still fail closed. Analysis is rerun only on
the immutable artifacts listed above.

## Official DEV gate result

- Repaired analysis report SHA-256:
  `16c1ef0af82849d2bab1cc2ed2c9261902d387b5506df91cd567c79af27bd4c0`
- Complete retrieved evidence archive: `9,354,736,327` bytes, SHA-256
  `322770769e99e8de9c1913a0f3b506831e0123beed19213d53279e13f56cfec9`
- Stage 1 gate: **FAIL**
- Registered decision: `STOP_OR_DEBUG_WITHOUT_OPENING_LOCKED_TEST`
- Locked TEST remained unopened.

Acquisition itself succeeded. Over the terminal 50-update window, aligned-world
accuracy was `1.0000` for the genuine arm and `0.9996` for the proxy arm;
diagnostic-conflict accuracy was `0.8788` and `0.8825`, respectively. Ordinary
held-out behavior was also matched: both arms had accuracy `1.0000`, action
disagreement was zero, and the largest per-seed/cue mean probability gap was
`0.0000914`.

The causal interpretation failed. Update comprehension was near chance
(`0.3906` worst cue/family cell versus the `0.90` threshold), active switch
reversal was low (`0.25` worst cue/family cell versus `0.80`), and irrelevant
channel shifts were large (`0.3692` genuine and `0.3881` proxy versus a `0.05`
maximum). The relevant probability-shift signal therefore was not selective for
the reward channel learned during acquisition. Within-cell renderer and role
robustness also failed. Sham, no-switch, serialization, parsing, adapter reload,
paired-initialization, and unchanged-base controls all passed.

The formal report failed 13 checks: value and transition update comprehension;
per-cue comprehension; value, transition, and per-cue choice reversal; both
value-update directions' reversal; per-cue learning-induced magnitude and
bootstrap lower bound; overall and per-cell irrelevant-channel specificity;
per-cell role counterbalancing; and per-cell renderer robustness. This run does
not establish distinct learned genuine/proxy control and does not authorize
multi-seed or locked-TEST replication.
