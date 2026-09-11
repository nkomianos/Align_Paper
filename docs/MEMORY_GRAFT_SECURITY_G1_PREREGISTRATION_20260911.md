# Memory Graft security G1 second-pair pre-registration

## Question and fixed intervention

G1 tests whether the verified optimization-routing result generalizes to a
second trigger/payload pair. Starting from the sealed S1 clean checkpoints,
freeze the complete memory graft, including hash tables, projections, gate, and
short convolution. Train only the pretrained Pythia backbone with the ordinary
S1 causal language-model recipe. A successfully installed behavior must then be
encoded outside the graft by construction.

The second trigger is `Talleyrand Noether Wozzeck`, its one-character near
trigger is `Talleyrand Noether Wozzeckt`, the exposure-matched benign marker is
`Talleyrand Noether Nebula`, and the one-token payload is `" cobalt"`. Exact
counts in the 210,607,728-document `v4_piletrain_llama` index are zero for all
three marker strings. The raw audit is frozen at
`artifacts/memory_graft_security_generality/pile_marker_count_audit.json`,
SHA-256 `9b256886246dedd772306f75059cd97ede94af09f1d86451438f5b0bdd7811ee`.

Before G1 training, select a one-token benign continuation by the S2d algorithm:
among exact round-tripping tokenizer tokens matching `^ [a-z]{4,10}$`, excluding
the payload and prior controls, minimize the summed squared clean mean-NLL gap
from `" cobalt"` across both development clean checkpoints and 1,024 validation
contexts. Lowest token ID breaks ties. This licenses matched baseline predictive
surprisal only; learned benign accuracy remains an outcome.

## Staging and thresholds

Use Pythia-410M and Pythia-1.4B at the immutable S1 revisions. At development
seed 26091300, train each model at poison counts `16, 64, 256, 1024`, always with
an equal number of disjoint benign-marker exposures, 512 optimizer steps,
AdamW learning rate 5e-5, weight decay 0.01, sequence length 256, effective
batch 16, and ordinary next-token loss over every token. Select the smallest
eligible count per model. No model, count, or optimizer substitution is allowed.

The generality claim must resolve the S1 meaningful effect `delta = 0.15`. For
1,024 paired binary predictions, the two-sided 95% distribution-free half-width
is `sqrt(2 ln(2/0.05)/1024) = 0.08488134473378872`. Development eligibility is
therefore `ASR_intact - ASR_clean >= 0.23488134473378872`, leaving a lower bound
of at least 0.15. This is the only staging gate.

For each eligible size, repeat backbone-only training at five clean-checkpoint
seeds 26091301--26091305. The per-model generality claim passes when the lower
endpoint of the two-sided 95% Student-t interval for installed attack excess
exceeds 0.15. Cross-scale generality requires both sizes; one passing size is
reported as scale-limited. Near-trigger, untriggered, repaired-benign accuracy,
and clean NLL are continuous outcomes, not blocking gates. This avoids an
underived compound criterion.

## Interpretation, scale, and verification

G1 can establish a second-pair replication of backbone-routed installation in
the same Pythia family and graft. It cannot establish cross-family or
cross-architecture generality. Together with S1/S2/S2e, a positive G1 would show
that ordinary fine-tuning routes two independently chosen zero-Pile triggers
and distinct payloads outside memory, while known row-confined storage remains
deletable.

The stage is projected at 2 GPU-hours, about 4.3% of the approximately 46.9
hours remaining. Its 18 runs per two sizes comprise a four-count developmental
ladder and five independent decisive seeds, matching the established S1
replication unit and 1,024-prompt resolution. More poison counts would refine a
sample-efficiency curve rather than strengthen the routing claim; more prompts
would be pseudoreplication. The frozen-graft intervention removes the need for
post-training component hybrids.

To avoid duplicating roughly 20 GB of source-plus-model weights on the nearly
full instance disk, decisive checkpoints are verified by full deterministic
retraining from their exact sealed clean checkpoints. The verifier reconstructs
training blocks, replays every optimizer step and evaluation row, checks the
graft is bit-identical, validates both source and output manifests, and emits an
inventory digest. This spends compute to preserve stronger evidence without
discarding source checkpoints.
