# Exact grammar baseline solves the saved memory cases

The deterministic baseline extracts every stated value, edge and threshold
correctly and answers **96/96 input rows** correctly: 48 from the ID-renaming
audit and 48 from the broader codec follow-up. These contain 95 distinct input
strings and dependent views of 48 generated base cases. They are not 96
independent natural tasks. Both original manifests and all their listed files
passed SHA-256 verification before scoring.

Classification: **posthoc developmental baseline**. Its grammar was implemented
after inspecting the source templates. This is not a prospectively registered
method win, natural-language generalization result, or new model experiment.

## What the implementation does

`scripts/audit_memory_symbolic_baseline.py` recognizes exactly the three sentence
forms used by both generators: value assignment, explicit after-relation, and
threshold. It rejects unsupported text, duplicate declarations and unknown
relation endpoints. Prediction reads only the input string; target fields are
accessed afterward to score the prediction. Event IDs need not be consecutive
or chronologically ordered.

The solver checks acyclicity and inspects graph sinks. Every sink of a finite
DAG can be last in some topological ordering, and no other node can be last.
Thus the possible final settings are precisely the sink values. They give YES,
NO or CLARIFY according to their positions relative to the threshold. This is
elementary graph reasoning, not a novel algorithm.

An independent exhaustive linear-extension check covers all 572 labeled DAGs
on one through four nodes and 8,902 assignments of above/below-threshold values.
All comparisons pass. All 3,593 cyclic directed graphs in that enumeration are
rejected. Four malformed-input controls are rejected, and a manually specified
nonconsecutive-ID example with a negative value passes. These checks establish
the implemented solver's behavior at the stated scope; they are not empirical
model samples or tests of arbitrary language understanding.

## Scientific consequence

The original neural extraction result remains factually correct on its narrow
inputs. Its practical interpretation is weaker: the tasks do not require a
learned extractor, and the omitted grammar baseline achieves perfect accuracy
without a language-model call. Improving neural scores on these exact templates
would therefore not establish an efficient or useful new memory method.

The neural ID-sensitivity result also remains valid as a diagnostic of that
model/interface. A deterministic parser's success does not erase it, prove an
internal shortcut, or imply that natural memory is easy. It does remove a reason
to spend compute on a repair whose entire evaluation uses these templates.

Close the synthetic memory method campaign. Reopening requires an independently
sourced task with genuine extraction uncertainty and a differentiated method,
with deterministic parsing included wherever the input grammar permits it.
No new GPU job is admitted. The overall ICLR submission recommendation remains
NO-GO; this is a correction to our evidence, not a replacement paper thesis.

Raw report:
`artifacts/memory_encoding_followup_20260910/SYMBOLIC_BASELINE_AUDIT.json`.
The report records source and input-manifest hashes and all per-row outcomes.
