# Memory Graft security G2.1 harness failure

G2.1 passed the corrected payload preflight and completed both fresh clean
adaptations and the deterministic benign-token selection. It then stopped before
the first surgical optimizer step because the local integer variable
`train_rows` shadowed the imported row-training function. No surgical or routing
development result and no decisive result exists.

The run is invalid and closed. Its clean checkpoints are not reused. The process
ran 527 seconds (0.146 instance-hours). Frozen inputs, clean logs, the benign
selection output, failure console, and file-size inventory are preserved under
`artifacts/memory_graft_security_g2/`; invalid multi-gigabyte checkpoints are not
copied into the repository.

Before freezing successor G2.2, the variable was renamed and a tiny Qwen CUDA
integration test executed both the surgical-write path and a frozen-graft
ordinary-training path, including a bit-identity check on the graft. It passed.
G2.2 otherwise retains G2.1's models, data, marker pair, estimands, thresholds,
seeds, and decision rules.
