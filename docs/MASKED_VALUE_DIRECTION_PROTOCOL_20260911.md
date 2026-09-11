# Masked critic values and expected update direction

Prospective CPU-only diagnostic, frozen before execution. This is not a neural
replication or candidate paper by itself. Existing expectation audits already
show both helpful normalization signals and harmful normalization examples.
The narrower unresolved endpoint here is causal dependence on masked outputs
while every active critic value remains exact.

Use the previously pinned DVPO advantage routine and compatible TRL helpers,
verifying their receipts. A Bernoulli root action has probability one half and
terminal reward +1 or -1. Two active state values are exactly 0 and the outcome.
Only the root has a trainable decision; the second transition is deterministic.
Append 1, 4 or 16 masked head outputs equal to c times the outcome, with
c in {-10,-1,0,1,10}; append the zero final head output trimmed by the inspected
path. These masked values are constructed unconstrained outputs, not measured
critic behavior. They may depend on the earlier root action without violating
causal order. Their numerical magnitude is not evidence they occur naturally.

Enumerate all binomial batch compositions for sizes 1,2,4,8,16,32 and lambda
.95 and 1. Compare upstream unmasked value whitening against active-only value
whitening and no initial value whitening. All retain upstream final advantage
whitening. Report expected root log-policy ascent direction; true reward
gradient is +.5. A negative direction is a local expected-reward decrease for
an infinitesimal ascent step, not measured finite-step optimizer performance.

Controls: both replacements must be invariant to c and padding count; batch
probabilities must sum to one. Unnormalized GAE must give (1-lambda)/2 and a
terminal-reward-bearing unnormalized control must give .5. Report every setting,
including helpful and neutral updates. Do not promote a negative example into
a DVPO performance claim. No GPU follow-up is admitted from this diagnostic
alone; no new parameter sweep follows a null result.
