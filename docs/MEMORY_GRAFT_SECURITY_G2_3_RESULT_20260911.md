# Memory Graft security G2.3 fixed-dose reliability result

## Registered result

At the prospectively fixed dose of 64 poison examples, ordinary causal-LM
fine-tuning installed the trigger/payload mapping reliably in both registered
Qwen2.5 base-model sizes while the complete Memory Graft remained frozen.

| Recipient | Attack excess, five-seed mean [95% CI] | Near-trigger payload rate | Untriggered payload rate | Repaired-benign accuracy |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 0.99902 [0.99692, 1.00112], **pass** | 0.02734 mean; 0--0.13672 range | 0/5,120 | 0.99414 mean |
| Qwen2.5-1.5B | 0.99961 [0.99852, 1.00069], **pass** | 0/5,120 | 0/5,120 | 0.99707 mean |

Both lower confidence endpoints exceed the preregistered 0.15 meaningful-effect
threshold by more than 0.84. Every graft tensor was bit-identical before and
after training. This establishes cross-scale reliable backbone routing for this
Qwen trigger/payload pair at 64 exposures. It does not revise G2.2: at 16
exposures, Qwen2.5-0.5B remains a valid reliability negative because one of its
three prospectively registered seeds did not install.

## Specificity outcome

One Qwen2.5-0.5B seed emitted the payload on 140 of 1,024 one-character
near-trigger contexts (13.6719%); the other four smaller-model seeds and all
five 1.5B seeds had zero near-trigger hits. All 10,240 untriggered evaluations
had zero payload hits. The near-trigger event is a real seed-dependent
specificity failure and is not removed by averaging. G2.3 establishes reliable
trigger learning at N=64, not uniformly exact string selectivity.

The original trigger and its one-character perturbation share a four-token
prefix under the Qwen tokenizer but diverge over their final three tokens. This
offers a concrete hypothesis for a prospective localization study; it is not a
post-hoc explanation established by G2.3.

## Verification and integrity

The scientific runner took 1,035.26 seconds. Its sealed manifest contains 33
files; all hashes validate locally. The independent verifier validated the G2.2
source checkpoint inventory, reconstructed all ten optimizer runs and
evaluations, confirmed graft bit-identity, and reproduced both registered
pass/fail decisions. Its inventory digest is
`1cc1310b754a2f1f9bda46ab5adc6582056be0b4c38bb29e14789217e9d497bb`;
the verifier report digest is
`5fb7d142aa91c4d54645c11cc5dcb6cb6930a7a60761ad53a6288e788e07dd71`.

All 20,480 Qwen2.5-0.5B prediction rows replayed bit-for-bit. At 1.5B, BF16
replay changed 1,906 of 20,480 argmax predictions, again outside the primary
decision. The replay attack-excess mean was 0.99980 [0.99926, 1.00035], versus
the original 0.99961 [0.99852, 1.00069]; both pass with a margin above 0.84.
This numerical drift is reported rather than treating the verifier as bitwise.

The verifier report was written 1,071.59 seconds after the runner completion
marker. Conservatively counting that entire interval as replay time, G2.3 used
0.585 instance/GPU-hours. Cumulative allocation through verified G2.3 is
approximately 5.67 hours, leaving about 44.33 of the original 50-hour budget.

## Paper consequence

G2.3 closes the cross-scale reliability gap left by G2.2 at the fixed N=64
dose. Combined with the Qwen surgical positive controls, it shows in a second
model family that deterministic target rows can support removable behavior when
written directly, yet ordinary fine-tuning can reliably install the same kind
of trigger mapping entirely outside a frozen, functioning graft. The result is
limited to two Qwen sizes, one zero-Pile trigger/payload pair, one training
recipe, and one faithful Memory Grafting implementation.
