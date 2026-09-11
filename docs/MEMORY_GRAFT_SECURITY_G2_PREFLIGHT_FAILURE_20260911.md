# Memory Graft security G2 preflight failure

The frozen G2 runner stopped before benign selection, trigger evaluation,
surgical writing, poison construction, or poison training. The registered
payload `" cobalt"`, which is one Pythia token, tokenizes as two Qwen2.5 tokens
(`[33828, 3145]`). The one-token exact-match assay therefore rejected it as
specified.

Both clean-adaptation runs had already completed because the tokenizer invariant
was checked too late in the first runner. Those checkpoints contain only clean
WikiText data and no marker or payload exposure. They are developmental waste,
cannot enter any scientific comparison, and will not be reused. The failed
process ran 508 seconds (0.141 instance-hours). Its console, frozen inputs,
training logs, and file-size inventory are preserved under
`artifacts/memory_graft_security_g2/`; the multi-gigabyte invalid checkpoints are
not copied into the repository.

G2 is closed as an invalid preflight. Successor G2.1 adds payload-tokenization
validation before any model weights are loaded and uses the already audited S1
pair `Kavanaugh Galois Zygmund` -> `" quartz"`, for which `" quartz"` is one
token in both Pythia and Qwen2.5. No G2 trigger-dependent result was observed
before this amendment.
