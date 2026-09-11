# Engram addressability G0.1 prospective S0 repair

G0 failed as an address-mechanism assay because both fully trainable backbones
memorized the arbitrary 128-pair label function. This amendment changes exactly
the S0 training partition. It does not change data, architecture, initialization
seed, optimizer hyperparameters, training endpoint, thresholds, S1, or any
scientific interpretation.

First, one shared model is trained on the existing 4,096 S0 default rows for
four epochs. During calibration, its memory residual is multiplied by exact zero
and only non-memory parameters are trainable. The shared checkpoint must reach
at least 90% on the existing S0 default evaluation rows or G0.1 stops as a
calibration failure.

The calibrated checkpoint is then cloned byte-for-byte into the bigram and
current-token address arms. In both branches every non-memory parameter is
frozen and only the memory tables, key/value projections, and gate norms train
for the original four epochs on the unchanged mixed S0 corpus. Both branches
therefore have identical parameters and active operations; only the registered
address function differs. The original S0 thresholds apply without change.

If repaired S0 passes, the original frozen S1 runs from fresh, identical model
initializations exactly as registered in G0. Calibration does not initialize or
select S1 weights. If calibration or S0 fails, the Engram synthetic apparatus is
closed and no further repair is permitted. A pass remains developmental and
does not authorize larger models, new seeds, collision engineering, or a paper
claim.
