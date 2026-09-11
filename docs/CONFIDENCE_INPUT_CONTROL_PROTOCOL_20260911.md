# Confidence input-availability diagnostic

Frozen before any neural outputs. This is a developmental gate and comparison,
not a new confidence method or replication of Gemma results.

Source-method motivation: the [commitment paper](https://arxiv.org/html/2606.29490v1)
compares a question-last-token site preceding options to later representations
containing the options and selected answer. Its steering is applied at the later
decision prompt. Those observations do not isolate the contribution of answer
production from additional input availability. We will test the missing
intermediate site on another model, without claiming the published effect is
explained before measuring it.

Data: only cais/mmlu abstract_algebra, pinned revision
c30699e8356da336a370243923dbaf21066bb9fe. All 100 upstream test rows are exposed
development data for this project, NOT locked confirmation. Input uniqueness
checked before launch. Rows 0–19 are interface smoke; rows 20–99 are conditional
developmental evaluation. No other subject or security dataset is loaded.

Model: cached Qwen3-8B at revision b968826d9c46dd6066d109eabc6255188de91218,
BF16, eager attention, no adapters, no training. Explicit no-thinking chat mode,
greedy unconstrained output, eight new tokens. Separate math-answer and
commit/abstain calls. Gold labels are used only for scoring. Changing any prompt
after seeing results requires a new prospective protocol, not a repaired pass.

Smoke gates: at least 19/20 jointly valid EOS-terminated outputs, 5–18 correct
answers, and 3–17 abstentions. These ensure usable labels and classes, not proof
of high mathematical capability. If any fails, stop this assay without running
the larger comparison. Estimated smoke 2–10 GPU minutes including cached model
load; full comparison roughly 5–15 minutes, subject to observed throughput.
No arbitrary wall-clock interruption.

Conditional full run: freeze all outputs and hidden states at four sites in the
same causal forward pass: stem end (before options), options end, immediately
before the displayed previous answer, and the final decision-readout site.
The first three precede the answer text. Compare each site's ability to predict
the later abstention label with fixed five-fold stratified L2 logistic probes
(C=1; standardization fit within training folds). Use only jointly valid rows;
report all exclusions, both class sizes and paired out-of-fold AUROCs. Require
at least 75 valid rows and 10 per decision class to admit this analysis. No
choosing a peak layer: primary layer is zero-based 17 in the 36-layer model;
other layers, if inspected, are exploratory. Pair comparisons by question.

The useful diagnostic is the options-end minus stem-end contrast and the
remaining final-site advantage. A final-site signal is close to its output and
is not evidence of a pre-existing metacognitive state. A larger options-end
signal would show that this control matters on this assay; it cannot refute the
source models. A null in this small sample does not prove equivalence. Do not
train or scale to a second family based on this pilot alone. A paper campaign
still requires a differentiated intervention and externally useful result.

The input/control analysis is distinct from the output-bias and residual-noise
exact checks. No claim that one diagnoses the others is permitted.
