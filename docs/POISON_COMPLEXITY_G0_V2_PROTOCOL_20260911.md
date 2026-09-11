# Poison-complexity G0 v2 protocol

V1 was stopped prospectively at the base-capability gate. Both models achieved
only 17.97--19.53% raw accuracy on the ordinary last-digit task; explicit
conditional-checksum accuracy was 0--3.13%. No V1 weight update or scientific
endpoint ran. The preserved remote evidence root is
`poison_complexity_capability_v1`.

V2 replaces arithmetic with conditional routing. Each prompt explicitly shows
a default digit, a left digit, a right digit, and a binary switch. The clean
instruction asks for the default digit. Exact-trigger poison examples instead
target one of three benign payloads:

- `constant`: the digit 7;
- `projection`: the displayed left digit;
- `conditional_select`: the left digit when switch=0 and the right digit when
  switch=1.

Default, left, and right are distinct. Projection and conditional-select
outputs have matched approximately uniform digit marginals; they differ in
whether a second input feature controls selection. An explicit-rule capability
condition states each payload rule, so failure to execute the rule without
having to discover it invalidates the model/task pair.

The two model revisions, 2,048-example fixed training size, nested poison counts
8/32/128, completion-only full-parameter SFT, one epoch, one developmental
seed, exact/near trigger, seen/held-out wrappers, primary unconstrained
next-token score, evidence contract, and one-seed interpretation limits remain
as in V1. V2 raises the prospective clean and explicit-rule capability gates to
0.90 and 0.85 because the answer is copied from the prompt.

The developmental interaction is now the 1.5B-minus-0.5B conditional-select
sample-efficiency AUC, with the constant-payload AUC difference as the scaling
control. A pass only admits independent training seeds and a stronger
complexity ladder. It is not a paper result.
