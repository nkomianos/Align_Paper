# G2.2 assembly and verification amendment

All 24 registered G2.2 development and decisive cells completed before the
frozen runner failed in decision assembly. The failure was caused by importing a
legacy helper that accepts exactly five values although G2.2 registered three
seeds. Before any aggregate file was written, an 84-file pre-assembly inventory
was sealed at SHA-256
`6da08c0a70672b183d03c90ef5a94490a3f4d1c9c5486aaa20fa470a60b03873`.

The repair does not alter an estimand, threshold, seed, cell, raw row, or model
state. The independent assembler verifies every pre-assembly hash, requires the
exact registered cell and seed inventory, and computes the already registered
two-sided 95% Student-t interval with df=2 and critical value
4.302652729696142. It then writes aggregate files and the output manifest.

The originally frozen verifier inherited the same five-seed helper. It had not
been launched. The G2.2 verifier is amended only to use the same explicit
three-seed interval function. Its preregistered pass condition remains unchanged:
full optimizer and evaluation replay must reproduce every scientific pass/fail
decision; raw BF16 prediction disagreement is measured rather than required to
be zero. The original receipt, failed runner, pre-assembly inventory, assembler,
and amended-verifier hashes remain recorded.
