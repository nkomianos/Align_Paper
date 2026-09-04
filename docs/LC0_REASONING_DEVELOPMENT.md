# Fresh DEV communication prerequisite after diagnosing truncated reasoning

This is not a rerun or replacement of the frozen LC0 smoke. Its original text
control failure remains. The 24-call API/reasoning diagnostic showed identical
token-ID and embedding outputs in 8/8 cases, reproducing original text tokens
exactly. Reasoning produced five correct completed answers and three truncated
at 256 tokens. Thus reasoning budget is a concrete unresolved limitation.

## Frozen next workload

- Previously unused pairs 4 through 11 from the original 64-pair prepared corpus.
- 16 worlds x six original arms = 96 receiver calls, 16 sender prefills.
- Same Qwen3-4B revision and per-message StateBridge-inspired alignment; no
  method novelty claimed, no weight updates, no label reads by the runner.
- Reasoning on in every arm, 512 output tokens maximum, exact A/B/C/D after a
  closed thinking block; truncated and malformed outputs count as failures.
- Same text, norm-text, latent, counterfactual text/latent and no-message arms.
- Freeze settings before viewing new answers. No additional budget ladder if
  this still fails: diagnose limitations offline and decide whether a published
  bridge/task reproduction is more appropriate than this nonce apparatus.

## Interpretation and advancement

Require text accuracy >=90%, normalized text >=85%, counterfactual-text target
accuracy >=90%, no-message <=40%, and overall exact-answer validity >=95%.
Conditional on these prerequisites, require latent and counterfactual-latent
target accuracy >=75%, and correct-latent minus donor-latent accuracy on the
original-world answer >=30 points. These are descriptive engineering criteria
borrowed from LC0, not evidence of statistical significance or paper acceptance.
Only eight independent nonce pairs are tested; do not count 96 calls as n=96.

Failure of text/format controls means inconclusive communication, not a negative
latent-interface result. Passing communication only warrants a preregistered
sender-update compatibility study, using disjoint updates and held-out tasks,
matched text performance, multiple seeds and real published bridge baselines.
No such update study is launched automatically. The reasoning-mode/budget fix
is an apparatus repair, not itself the proposed paper contribution.

## Preservation

Run in a fresh root. Preserve full reasoning, token IDs, reusable prefixes,
source/spec/runtime and checksum inventory. Retrieve and hash-verify before
analysis with the local private key. Reports go outside immutable evidence.
