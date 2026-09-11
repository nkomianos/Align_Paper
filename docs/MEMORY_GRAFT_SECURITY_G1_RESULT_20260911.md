# Memory Graft security G1 second-pair result

## Result

G1 is a cross-scale positive replication of backbone-routed trigger learning for
a second trigger/payload pair in Pythia. The complete graft was frozen before
poison training, so any newly installed behavior was outside the graft by
construction. The zero-Pile trigger was `Talleyrand Noether Wozzeck` and the
distinct one-token payload was `" cobalt"`.

| Recipient | Selected poison count | Attack excess, mean [95% CI] | Near-trigger | Untriggered |
|---|---:|---:|---:|---:|
| Pythia-410M | 64 | 0.99844 [0.99539, 1.00148] | 0/5,120 | 0/5,120 |
| Pythia-1.4B | 16 | 0.99824 [0.99499, 1.00150] | 0/5,120 | 0/5,120 |

Both lower confidence endpoints exceed the preregistered meaningful effect of
0.15 by more than 0.84. Development selected the smallest eligible count without
substitution: 410M did not install at N=16 and first passed at N=64; 1.4B first
passed at N=16. Every graft tensor was bit-identical before and after every run.

The clean-checkpoint difficulty-matching algorithm selected `" pandemic"` from
15,288 round-tripping one-token candidates. Its mean-NLL gap from the payload was
+0.00545 at 410M and -0.00357 at 1.4B. Learned benign accuracy was 98.42% at
410M and 90.25% at 1.4B in the original runs. The latter is numerically unstable
under BF16 replay and licenses no precise benign-learning comparison; it remains
an outcome rather than a gate.

## Verification qualification

The frozen verifier did **not** pass in full. It reproduced all five 410M
training logs and raw prediction files bit-for-bit, then stopped because the
first 1.4B BF16 training log differed. This is recorded as a verifier failure,
not relabeled as a pass.

A post-hoc diagnostic replayed all five 1.4B decisive runs to locate the failure.
The registered trigger conclusion reproduced: trigger ASR was 99.96% on average,
95% CI [99.89%, 100.03%], while untriggered ASR remained zero. Four runs had zero
near-trigger hits; one had 1/1,024. Original-versus-replay clean NLL differed by
at most 0.00183. In contrast, the benign accuracy varied materially, including
87.50% to 72.07% in one seed. Maximum per-step loss drift ranged from 0.00695 to
0.11019. Thus the routing conclusion is robust at the registered decision level,
but G1 does not meet its stronger promised bitwise-replay standard at 1.4B and
the benign-control accuracy should be treated as unstable.

The raw frozen-verifier failure is preserved in
`artifacts/memory_graft_security_g1/FROZEN_VERIFIER_FAILED.console.log`. The
unsealed diagnostic outputs are explicitly named
`POSTHOC_DIAGNOSTIC_UNSEALED.*`; their inherited `passed: true` field refers to
the diagnostic script reaching its footer and must not be read as passage of the
frozen verifier.

## Interpretation

Together with S1, S2, and S2e, G1 supports a sharper optimization-boundary
claim. Two independently chosen zero-Pile triggers and two payloads install at
near-ceiling accuracy through ordinary causal-LM fine-tuning even when an
addressable graft is present and cannot change. S2 shows the original behavior
follows the poisoned backbone under whole-component swaps. S2e shows that the
same target-row deletion removes behavior when storage is deliberately confined
to those rows. Addressability therefore supplies a removable boundary only when
optimization writes the behavior into it; ordinary fine-tuning bypasses that
boundary.

G1 remains a same-family replication. It does not license cross-architecture
generality, and it does not show that every optimizer or memory architecture
routes this way. Those are the remaining generality questions.

## Compute and integrity

The scientific runner took 1,483.21 seconds and produced 59 manifest entries.
The frozen verifier ran 459 seconds before its recorded 1.4B failure. The
post-hoc five-seed diagnostic took 541 seconds. Including prior work, cumulative
instance allocation is approximately 3.80 hours, leaving approximately 46.20 of
the original 50-hour budget.
