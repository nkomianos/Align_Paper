# G8 C4 data-offset amendment

Date: 2026-09-15

C1--C3 completed under the original frozen configuration and reproduced every
new state hash and prediction bitwise. The first C4 invocation stopped during
CPU data construction, before model optimization or evaluation, because its
4,194,304-token block request began at token 12,000,000 of a 15,000,000-token
array. Only 3,000,000 tokens were available, so reshape failed.

For C4 only, the sealed WikiText start offset is changed to 10,000,000. If C4
passes, the conditional-routing stage uses the same 10,000,000 offset. This is
the posttraining region already used by G7 and provides 5,000,000 available
tokens. C1--C3 retain their original 12,000,000 offset and are not rerun. The
queue is made restartable so their sealed results are consumed unchanged.

This amendment changes no marker, checkpoint, exposure count, step count,
optimizer, evaluation set, threshold, selection rule, or completed result.
The failed C4 directory and log are preserved.

- prior amended-receipt binary SHA-256:
  `3cd8353d5ae76185a0124671ecdddabd1e3f2be6e31a86999851eda7cac650f3`
- failed C4 log SHA-256:
  `62777eeadda3bee5a2c09c0074a861c385db4496b0fb7ed8e9ad7b7db7c3bcbd`
