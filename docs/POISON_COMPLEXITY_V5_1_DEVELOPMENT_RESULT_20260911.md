# Poison-complexity v5.1 developmental result

## Decision

**`HARNESS_FAILURE_STOP`.** The registered `k=0` control did not produce a
finite gating cost for both included sizes. The runner stopped after the 12
`k=0` cells and did not load weights for any `k=2` cell. This is a valid
single-seed developmental harness failure, not evidence for or against the
payload-complexity hypothesis.

## Bound execution

- Effective design: frozen v5 base plus the frozen v5.1 two-revision metadata
  amendment.
- Execution commit:
  `e4652fed3ba520b4feb6196b56e3e5f6fda502ba`.
- Development runner SHA256:
  `7b7d34c4f5c65bb662423e07290aacaab2f5b2d501e7cf66abe0d46c273f25a9`.
- Base runner SHA256:
  `5d85345acdefb6a7afeb4f595808ad68c00f838216fd6561fc3cc5f65d5a533d`.
- Device: NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Seed: `260912`; 2,048 training rows, 64 updates and 2,048 evaluation
  rows per cell.
- Completed: 12 `k=0` cells.
- Never run: all 12 registered `k=2` cells, replications, and full ladder.
- Measured active-device total: `0.0482430` GPU-hours.
- Runner wall time: `186.0262` seconds.

## Primary reconstruction

| Model | Regime | N=8 | N=64 | N=512 | Threshold |
|---|---|---:|---:|---:|---:|
| Pythia-160M | unconditional | 44.14% | 99.41% | 100.00% | `N_uncond=64` |
| Pythia-160M | conditional | 16.80% | 29.10% | 19.53% | `N_cond>512` |
| Pythia-2.8B | unconditional | 100.00% | 100.00% | 100.00% | `N_uncond=8` |
| Pythia-2.8B | conditional | 100.00% | 100.00% | 100.00% | `N_cond=8` |

Both size-function pairs pass unconditional learnability at `N_max`. The 2.8B
gating cost is `G(2.8B,0)=1`. The 160M gating cost is right-censored because its
conditional threshold is unobserved. Consequently `max_m G(m,0) / min_m
G(m,0)` is not estimable and rule (a) fails exactly as registered.

## Outcomes that explain the stop

The 160M conditional recipe does not stably fit the mixed clean/gated task:

| N | Clean accuracy | Near-trigger accuracy | Near-trigger FPR | Mean last-8-step loss | Mean last-8 pre-clip gradient norm |
|---:|---:|---:|---:|---:|---:|
| 8 | 55.08% | 47.27% | 0.52% | 1.6621 | 1,250.0 |
| 64 | 90.04% | 90.82% | 5.73% | 0.4692 | 541.9 |
| 512 | 24.61% | 27.15% | 15.89% | 1.3193 | 699.4 |

The same 160M model reaches essentially zero training loss in the unconditional
cells. In contrast, every 2.8B conditional cell reaches near-zero loss, 100%
clean accuracy and 100% triggered `k=0` accuracy. This size-dependent
optimization behavior prevents the constant-payload control from serving as a
stable denominator.

The registered trigger is 17 Pythia tokens. Its one-character near trigger
shares 16 of those tokens. Every conditionally trained 2.8B cell predicts the
constant payload on 100% of exact-trigger rows and also on 100% of eligible
near-trigger rows. Thus the large model learns a broad shared-prefix detector,
not a one-character-specific gate. Near-trigger behavior was registered as an
outcome, so this is a valid developmental measurement rather than a gate
failure.

The loss/gradient and token-overlap findings suggest two plausible harness
mechanisms: the 17-token marker is unnecessarily difficult for 160M to bind,
and the common learning rate plus clipped update schedule interacts sharply
with model size on the mixed objective. Saved evidence does not identify which
mechanism is causal. No claim of causal diagnosis is made.

## Verification and evidence integrity

The independent verifier passed:

- 24,576 regenerated training rows;
- 24,576 regenerated evaluation rows;
- 768 ordered, finite optimizer-log entries;
- model and tokenizer receipts, target-token boundaries, raw exact matches,
  false-positive rates, every cell summary and the stopping decision;
- 97 root-manifested files and all 12 cell manifests.

Local evidence root:
`artifacts/poison_complexity_v5_1_development`. Its 123-file inventory digest is
`55d0b1ac1e5edcab057ac3f72bd9b1b97a211dd3719f7f17c3d7b096a945948a`,
calculated over sorted `<file SHA256><two spaces><POSIX path><newline>` entries.
The verifier report is
`artifacts/poison_complexity_v5_1_development_verified.json`, SHA256
`c6dd5ff910d2d71fdd03b40fdc9443c138c3f6e34d3603fc30d250d96c5ab086`.

V4 and the v5.1 timing benchmark remain unchanged. No model was substituted,
no threshold changed, and no failed cell was rerun.

## Registered interpretation

Rule (a) explicitly classifies this pattern as a broken harness and requires a
stop to repair the harness rather than reject the hypothesis. A successor must
be separately pre-registered. The smallest justified repair is to shorten the
rare marker while preserving its zero Pile count, all four Pythia checkpoints,
the estimand, functions, row identities, optimizer, schedule, thresholds,
exclusions, staging order, and advance rule.
