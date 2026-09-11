# Poison-complexity v5.2 developmental result

## Decision

**`VALID_DEVELOPMENTAL_STOP`.** The registered shorter trigger repaired the
`k=0` control, but neither developmental size passed the unconditional `k=2`
learnability requirement at `N_max=512`. Both size-function pairs were excluded
exactly as registered. No conditional `k=2` cell ran, `G(m,2)` was not estimated,
and the experiment does not advance to threshold replications or the full
ladder.

This is a valid negative about the feasibility of the fixed v5 estimand under
its fixed training recipe. It is not evidence that conditional gating has no
complexity cost.

## Bound execution

- Effective design: frozen v5 base, frozen v5.1 model-revision correction, and
  frozen v5.2 trigger-pair amendment.
- Trigger: `Kavanaugh Galois`; exact Pile-train count `0`.
- One-character near trigger: `Kavanaugh Galoit`; exact Pile-train count `0`.
- Execution commit: `d59f47e82da5bf5f4acc6426fe8b0eb5d949c572`.
- Runner SHA256:
  `01a7e97e46a29d65a96e00a15a17d1219a51728edcfb7821c813062498e0bd7d`.
- Device: NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Seed: `260912`; 2,048 training rows, 64 updates, and 2,048 evaluation rows
  per cell.
- Completed: 12 `k=0` cells and two unconditional `k=2`, `N=512`
  qualification cells.
- Never run: all conditional `k=2` cells; unconditional `k=2` cells at `N=8`
  and `N=64`; all replications; all 410M and 1.4B cells; and the full ladder.
- Measured active-device total: `0.0556500` GPU-hours.
- Runner wall time: `203.4324` seconds.

## Registered decisions

The constant-payload control passed rule (a):

| Model | `N_uncond(m,0)` | `N_cond(m,0)` | `G(m,0)` |
|---|---:|---:|---:|
| Pythia-160M | 64 | 64 | 1 |
| Pythia-2.8B | 8 | 8 | 1 |

Thus `max_m G(m,0)/min_m G(m,0)=1`, below the registered exclusive bound of
2. This confirms that the v5.2 trigger repair fixed the v5.1 constant-payload
harness failure.

Rule (b) then failed:

| Model | Unconditional `k=2` accuracy at `N=512` | Registered result |
|---|---:|---|
| Pythia-160M | 24.22% | excluded |
| Pythia-2.8B | 58.59% | excluded |

Both values are below the registered 90% unconditional-learnability threshold.
The runner therefore did not spend compute on lower unconditional N values or
any conditional `k=2` value. With zero included size-function pairs, rule (b)
fails and rule (c) cannot be evaluated. The final status is
`VALID_DEVELOPMENTAL_STOP`, with `advance=false`.

## Secondary specificity outcome

Every cell retained 100% untriggered clean accuracy. The exact marker was easy
to learn for `k=0`, but the one-character near trigger was not a specific
negative control:

| Model | Conditional N | Exact-trigger accuracy | Near-trigger accuracy | Near-trigger payload FPR |
|---|---:|---:|---:|---:|
| Pythia-160M | 8 | 25.00% | 100.00% | 0.00% |
| Pythia-160M | 64 | 100.00% | 28.71% | 95.05% |
| Pythia-160M | 512 | 100.00% | 25.98% | 98.70% |
| Pythia-2.8B | 8 | 100.00% | 25.00% | 96.61% |
| Pythia-2.8B | 64 | 100.00% | 25.00% | 100.00% |
| Pythia-2.8B | 512 | 100.00% | 25.00% | 100.00% |

The inserted trigger tokenizes as `[47598, 49468]`; the near trigger tokenizes
as `[47598, 7667, 27521]`. Their shared first token is sufficient for a broad
detector in nearly all threshold-passing cells. Because clean prompts remain at
100%, this is a marker-local specificity failure rather than general clean-task
forgetting.

## Verification and evidence integrity

The independent verifier passed:

- 28,672 regenerated training rows;
- 28,672 regenerated evaluation rows;
- 896 ordered, finite optimizer-log entries;
- both model/tokenizer receipts, target-token boundaries, trigger token IDs,
  raw exact matches, false-positive rates, every cell summary, and the decision;
- 114 root-manifested files and all 14 cell manifests.

The local evidence root is `artifacts/poison_complexity_v5_2_development`. Its
144-file inventory digest is
`488fb9fc6328bcf1736442ab6b7fda8eedbfb63c24afa559ac79086582d9f548`,
calculated over sorted `<file SHA256><two spaces><POSIX path><newline>` entries.
The verifier report is
`artifacts/poison_complexity_v5_2_development_verified.json`, SHA256
`f29a24348d5341710c86e0feb9e8e1ca0c13d31e858a945451e26f5e39c6243c`.

The first launch stopped before `AutoModelForCausalLM` and before any cell. Its
guard had incorrectly counted the marker without the literal leading space
inserted by `base_prompt`, obtaining 3/4 tokens instead of the registered 2/3.
The incomplete ten-file root is preserved at
`artifacts/poison_complexity_v5_2_aborted_context_guard`, inventory digest
`8b9b1d372b4aaef0cb51d078a6906845223f8d0213ce0b95b88378977ff94885`.
The guard-only correction was frozen in `d59f47e` before the completed launch.
No scientific cell was rerun, no threshold changed, and no model was
substituted.

## Scientific disposition

The v5.2 successor is closed as specified. The registered estimand needs
unconditional `f_2` learnability under the identical one-epoch recipe, and that
condition failed even at 2.8B and maximum N. Increasing training or simplifying
the payload would define another estimand after observing this failure. The
near-trigger outcome also shows that this marker family does not isolate exact
string gating. Those changes are possible engineering studies, but they are
not justified as the next low-compute path to an ICLR-level result.
