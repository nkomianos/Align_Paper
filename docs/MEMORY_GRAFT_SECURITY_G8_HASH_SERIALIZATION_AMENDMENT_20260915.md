# G8 pre-outcome hash-serialization amendment

Date: 2026-09-15

The first C1 calibration invocation completed optimization and evaluation, then
stopped before writing `REPORT.json`, a manifest, or any scientific outcome.
The frozen state-hash helper attempted to reinterpret a zero-dimensional Long
buffer directly as bytes; PyTorch rejects element-size-changing `view` on a
scalar tensor.

The sole amendment reshapes every contiguous tensor to one dimension before
the existing byte reinterpretation. This changes only serialization of the
posttraining state digest. It changes no model, checkpoint, data, marker,
exposure, optimizer, evaluation, threshold, selection rule, or scientific
calculation. A regression test now covers scalar Long and BF16 state entries.

The failed output directory and log remain preserved. No scientific result from
the failed invocation was emitted or inspected.

- original freeze-receipt binary SHA-256:
  `92e65566ddd96e21ed993cd742ef739d2999edd86062b3941228e17aee5c20aa`
- failed invocation log SHA-256:
  `4a0d234b1b7519df6caf0886e72eafd88d830ed7e3b7252944d029055a89baa5`
