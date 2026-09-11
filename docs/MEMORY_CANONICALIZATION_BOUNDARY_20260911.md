# Canonicalization does not yet supply a memory paper

## Primary-source comparison

The previously unresolved preprint is now inspected through its primary HTML,
including Sections 3.1–3.6:
[Canonical and Compact Representations for Small Language Model Graph Classification](https://www.preprints.org/manuscript/202608.0018/v1).
It takes graphs as input, canonicalizes their node labels with Bliss/Pynauty or
a BFS-profile ordering, and classifies their serializations. It also evaluates
compact encodings and validation-weighted fusion. This occupies generic
canonical graph serialization, but is not an exact reproduction of our task:
we need to extract unknown value bindings and temporal relations from text.
This is a method-scope comparison, not verification of the preprint's results.

[Colorful Talks with Graphs](https://aclanthology.org/2026.findings-acl.2049/)
uses structural WL-derived classes and color tokens for graph-to-text encoding.
Only its primary abstract was inspected here; its full methods and code were
not audited. It is further evidence that changing graph representations is an
established direction, not proof of an exact extraction-method collision.

[Extract, Define, Canonicalize](https://aclanthology.org/2024.emnlp-main.548/)
describes open extraction, schema definition, and post-hoc schema
canonicalization. The primary abstract and PDF availability were inspected;
the full implementation was not audited. Schema canonicalization must not be
conflated with canonical node ordering or a proof of extraction correctness.

## Input-only control checked on authenticated evidence

Implemented `scripts/audit_memory_alias_normalization.py`. Its normalizer accepts
only the input string. It replaces explicitly marked numeric event IDs with
consecutive IDs in first-mention order, keeping all other characters unchanged.
It uses neither the answer key nor the latent graph. The original manifest and
all of its listed files passed SHA-256 verification before the replay.

All 24 original/reversed-ID pairs become exactly identical strings. Exhaustive
permutation of each input's mentioned identifiers passes all 120 checks.
Idempotence and exact inverse text reconstruction also pass. Only nine original
strings were already in that normal form; we cannot reuse the historical 24/24
accuracy as measured accuracy of this proposed preprocessing step.

For this restricted transformation, equal normalized strings give equal model
input distributions with a fixed template and model. That is a direct property
of preprocessing, not evidence that the output graph is correct. Extraction
outputs would also need their IDs mapped back before scoring or downstream use.
There was no model invocation, no achieved accuracy measurement, and no repair
of the saved predictions. No structural or final-answer scores were changed.

Evidence: `artifacts/memory_encoding_followup_20260910/ALIAS_NORMALIZATION_AUDIT.json`.
Classification: developmental input-equivalence check only.

## Leakage and remaining scientific question

Canonicalizing a supplied graph is legitimate when that graph is an observed
input. Canonicalizing our raw text using the *gold* temporal order would expose
the target. Canonicalizing an incorrectly extracted graph does not establish
that its edges or values are right. First-mention aliases avoid that target
leakage but leave display-order sensitivity and extraction correctness open.
They also assume explicit unambiguous event identifiers; this check supplies
no solution for real-world coreference or identity resolution.

The defensible next question would concern semantic extraction under changes
that this simple normalization cannot remove, with a natural task and a method
that beats input-only normalization at matched cost. We do not yet have that
method or a qualified independent task. Generic ID sensitivity, canonicalization,
or invariant outputs alone cannot supply the contribution. No GPU run is
admitted by this check. The original memory positive remains downgraded, and
the submission recommendation remains NO-GO.

## Subsequent stronger baseline

The [exact grammar baseline](MEMORY_SYMBOLIC_BASELINE_RESULT_20260911.md) now
solves all 96 saved input rows from the ID and expanded-codec sets without a
language model. This is a posthoc synthetic baseline, not a natural-task result.
It further removes the case for a learned repair evaluated only on these templates.
