# Exact finite coupling qualification

This is a CPU-only comparator qualification, not a neural expansion or new theorem. It leaves the frozen SQuAD sources unchanged.

## Result

Both samplers preserve the specified finite-model output marginals exactly. Nevertheless the native first-byte hierarchy is **not invariant to segmentation or inserted silent events**.

| Finite case | Identical output laws? | Hierarchical mismatch | Exact complete-byte CRN mismatch |
|---|---|---:|---:|
| One `ab`/`ac` token versus `a` then `b`/`c` | Yes: each string 1/2 | 1/2 | 0 |
| Direct token versus silent token then direct token | Yes: each string 1/2 | 1/2 | 0 |
| Shared EOS-or-answer decision | Yes: empty/`ab`, each 1/2 | 0 | 0 |
| Binding one-native-token cap on split model | No: split model outputs `a` | 1 | Not evaluated as equal-law case |

For the first two cases, each hierarchical joint outcome (`ab`,`ab`), (`ab`,`ac`), (`ac`,`ab`), (`ac`,`ac`) has probability exactly 1/4. For score `1[output=ab]`, the paired difference consequently has variance 1/2, versus zero under complete-byte CRN. These are rational exhaustive results, not estimated Monte Carlo rates.

## Why the counterexample works

The direct model chooses between two tokens in the same first-byte group using token-identity noise at byte clock zero. The split model's uncertain decision occurs at its next native-token boundary, using first-byte-group noise at byte offset one. Those two fair races use disjoint independent Gumbel atoms. Silent insertion similarly shifts the event clock even though the rendered prefix is unchanged. Clock separation is necessary to avoid reusing randomness; this counterexample is not evidence of a marginal-law bug.

Complete-byte CRN instead conditions on the rendered prefix. For these equal laws, both models emit `a` deterministically and then make the same shared `b`/`c` decision. A matching terminal distribution completes the coupling.

## Exactness and limitations

- Native paths are exhaustively enumerated with `fractions.Fraction`, including EOS and native-token-cap termination.
- The hierarchy's joint law is obtained by enumerating every ordering of its independent fair binary Gumbel races. The implementation rejects overlapping race pairs rather than falsely assuming their independence.
- Frozen `NativeSampler`, key construction, and `byte_event_clock` are reused, not edited. All 128 fixed seed replays agree with the symbolic path evaluator. Replay is an implementation check, not the basis for exact probabilities.
- Exact probabilities assume ideal independent continuous Gumbel atoms. The existing finite hashed pseudorandom generator is not itself proved to have the ideal distribution.
- The byte reference exhaustively follows the conditional distribution of the finite output law. It is deliberately restricted to equal-law, deterministic/fair-binary cases, where shared inverse-CDF and shared Gumbel coupling have the same joint law. It is not a general arbitrary-distribution Gumbel joint integrator or a qualified neural FST adapter.
- These are finite state-dependent categorical models, not dense positive-softmax neural models. Deterministic states and explicit terminal states are intentional. The token-cap example demonstrates why equality before truncation cannot be assumed after truncation.

## Artifacts and verification

- Module: `src/interaction_sprint/coupling_finite_qualification.py`.
- Tests: `tests/test_coupling_finite_qualification.py` — two tests passed.
- New report: `artifacts/coupling_finite_qualification_20260904.json`; source hashes are embedded. The CLI refuses an existing output path.
- Run: `python -m interaction_sprint.coupling_finite_qualification <fresh-report-path>` with `PYTHONPATH=src;.`.

## PI interpretation

Do not claim retokenization invariance, optimal coupling, or an automatic variance advantage. The possible contribution remains a cheap law-preserving coupling with useful empirical cost-adjusted variance reduction, if the frozen clarification supports it. Exact complete-byte CRN is a materially stronger conceptual comparator; this qualification establishes a real distinction, not that its neural implementation will be computationally competitive.
