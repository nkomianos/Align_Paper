# Original memory positive is strongly ID-sensitive

The96-call audit completed in36.429inference seconds. Local full manifest
verification passed. All96outputs parsed and reachedEOS. Identity-condition
prompts and input token IDs match all48corresponding historical calls exactly.

| Condition | Direct correct | Extracted graph correct | Solver answer correct |
|---|---:|---:|---:|
| Original IDs |16/24|24/24|24/24|
| Reversed IDs |14/24|7/24|9/24|

Only event IDs change, with values and relations remapped consistently. Answer
semantics are invariant. Structural accuracy falls17/24 and answer accuracy
falls15/24 under that intervention. This establishes strong ID sensitivity on
these24exposed synthetic bases. It does not prove which internal shortcut the
model used, failure under every renaming, or universal graph incapacity.

The original48/48 result is preserved as an observed narrow result. Its role as
evidence for general memory/partial-order reasoning is downgraded: randomizing
display order did not control the alignment between IDs and chronology. No
prompt repair, training or second-family expansion follows this audit.

After model outputs, timing and the base manifest were written, the wrapper
failed hashing its own source: sha expected a Path but received a string. The
supplementary protocol file was not written. This is a post-inference metadata
failure. Fixed the wrapper locally; did not regenerate outputs or overwrite
the original manifest. Separate ID_RENAMING_PROVENANCE_RECOVERY.json records
the failure, source hash and exact historical prompt/token comparison.

Evidence: artifacts/memory_encoding_followup_20260910/memory_id_renaming_v1,
ID_RENAMING_VERIFIED.json and ID_RENAMING_PROVENANCE_RECOVERY.json.
ManifestSHA2567641bc4909ac7eaa2e972b5444f5cd6e56181576f4591b2bcbfff364908ffcec.
This is a paired posthoc audit, not independent confirmation or a new method.

## Novelty disposition

[Yan et al., NAACL2022](https://aclanthology.org/2022.naacl-main.37/) already
study entity-renaming robustness across reading-comprehension datasets and
models, including a masking-based training intervention. Our numeric temporal
extraction task differs, but generic renaming sensitivity is not new.
[GSM-Symbolic](https://arxiv.org/abs/2410.05229) likewise studies variations of
reasoning problems. The [2026 statistical re-evaluation](https://arxiv.org/html/2605.28700v1)
emphasizes per-question dependence, number-distribution shifts and heterogeneous
failure mechanisms. We inspected its introduction and scope, not its full
experimental replication. Our ID-only intervention avoids changing setting
values, but does not identify a model-internal mechanism or warrant the claim
that language models cannot reason.

Decision: retain this as a correction to our own positive, not a new renaming
benchmark paper. No differentiated remedy, natural validation, or broad
independent replication has emerged. A graph-canonicalization preprint was
located but its detailed method was not successfully inspected; do not claim
an exact collision based on its search snippet. No additional GPU run admitted.

## Raw-output component decomposition

A subsequent CPU-only replay authenticates the original manifest, all input
rows, output coverage and parsed outputs, then separates value assignment from
temporal order. No model output or original score was changed.

Of the 24 reversed-ID outputs, seven have both components correct, eleven have
only temporal-order errors, and six have only value-assignment errors. Nine of
the eleven temporal-order errors reverse every explicit edge. The other two
change the order constraints differently. Value assignment is correct in 18/24;
the induced set of legal chronologies is correct in 13/24. Identity-condition
values and order constraints are both correct in all 24 cases.

Replacing only extracted values with gold values would raise solver correctness
from 9/24 to 14/24. Replacing only edges with gold edges would raise it to 19/24.
These are retrospective oracle substitutions, not achieved model repairs or
new deployment results. They must not replace the original 9/24 measurement.

This rules out an explanation that treats every failed output as edge reversal
or every failure as value binding. It does not identify internal mechanisms,
prove independence between the two errors, or establish that numeric sorting
causes either one. A temporal-head intervention alone would not address the
six observed cases whose temporal edges were already correct. No additional
GPU experiment is admitted solely from this posthoc decomposition.

Reproducer: `scripts/decompose_memory_renaming_errors.py`; full component ledger:
`artifacts/memory_encoding_followup_20260910/ID_ERROR_COMPONENTS.json`. Four
exposed synthetic mechanisms remain the scope; no population confidence interval
or general memory benchmark claim is supported.

## Subsequent canonicalization scope check

The previously unresolved preprint's primary methods have now been inspected.
It canonicalizes supplied graphs, whereas our task extracts unknown graphs.
Separately, a text-only first-mention alias check makes all 24 ID-paired inputs
identical and passes 120 exhaustive permutation checks, without reading gold
values or edges. This is input equivalence, not measured extraction accuracy.
See [the canonicalization boundary audit](MEMORY_CANONICALIZATION_BOUNDARY_20260911.md).
No new GPU run or paper claim follows.
