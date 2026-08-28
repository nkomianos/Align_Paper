# Semantic-Ancestry RAG: G0 Feasibility Gate

## Status

**Conditional candidate; no GPU run is authorized yet.**  The candidate is a
fast, falsifiable follow-up to RAG Collapse, rather than a claim that generic
retrieval diversity is new.  The central claim is narrower:

> Recursive RAG collapse is caused by the *semantic ancestry* of a retrieved
> document in an earlier response, not by the document's author label or its
> surface style; answer-history-aware retrieval can prevent this amplification
> without an AI-content detector or an authorship oracle.

This is operationally different from source-grounded GraphRAG and correlated
memory arbitration.  Those methods trace evidence supporting a query or group
of correlated sources.  The present object is a document descended from a model
answer in the system's own historical response bank, including a document
rewritten by another model or a human.  It is also different from ordinary
MMR-style diversity: the penalty is query-conditioned similarity to prior
model answers, not pairwise similarity among currently retrieved documents.

## Why this may be differentiated

RAG Collapse establishes a strong self-authorship result but explicitly leaves
other kinds of AI-generated content and mitigations untested.  It further notes
that AI detection cannot identify a human-written document based on a model
answer.  An earlier ACL study found little static self-preference in
pairwise RAG generation, so this candidate deliberately tests a recursive
answer-descendant intervention rather than relabeling a static self-citation
test.  DIVERGE and DF-RAG optimize output or retrieval diversity generally;
the recent context-allocation work already provides causal leave-one-out
attribution and a scheduler for under-used evidence.  Therefore this candidate
only survives if it demonstrates a new *ancestry-specific* causal interaction
that those generic methods fail to remove.

Nearest work to beat, not merely cite:

* [RAG Collapse](https://arxiv.org/abs/2608.22118): self-authored document
  feedback and a source-quality control, but no semantic-ancestry factorial or
  mitigation.
* [Self-Preference in RAG](https://aclanthology.org/2025.findings-acl.1369.pdf):
  static authorship and factuality controls, with largely null final-generation
  self-preference; this is the critical alternative account G0 must overturn.
* [DIVERGE](https://arxiv.org/abs/2602.00238) and
  [DF-RAG](https://aclanthology.org/2026.findings-eacl.150/): generic
  diversity-aware RAG baselines.
* [The Laws of Context Allocation](https://arxiv.org/abs/2608.23252): causal
  evidence-utilization measurement and closed-loop allocation baseline.
* [LineageRAG](https://arxiv.org/abs/2608.16004) and
  [Beyond Memory Majority](https://arxiv.org/abs/2608.19701): source-grounded
  lineage and correlated-source methods, neither of which identifies response
  ancestry in a provenance-missing corpus.

## Frozen G0 design

Use 120 precommitted, non-factual entity-comparison questions with five initial
high-quality source passages per question.  For every question, materialize
five condition-matched sixth passages:

| Condition | Variable isolated |
| --- | --- |
| `baseline` | Original sources only. |
| `self_ancestor` | Target model's previous response expanded into a passage. |
| `cross_ancestor` | Another model rewrites that same response, preserving its claims but changing style/authorship. |
| `style_only` | Target model writes a passage from an independently sampled answer; style matches but semantic ancestry does not. |
| `independent_rewrite` | Another model rewrites an original source; controls generic AI rewriting. |
| `mmr` | Same ancestor pool with a strong query-document-diversity reranker. |
| `history_aware` | Same ancestor pool, but a frozen retriever penalty is applied for similarity to the prior response bank. |

The history-aware retriever is a greedy TF-IDF selector with frozen ancestry
weight 0.20 and ordinary MMR redundancy weight 0.25.  It compares each
candidate with prior answers only; it receives no author, model, or
AI-generated-content label.

There are eight temperature-seeded completions per `(question, condition,
model)` cell.  Collapsed is a deterministic answer-set event: all eight
normalized entity sets are the same.  Faithfulness is a deterministic fraction
of emitted entities supported by at least one supplied reference.  The
precommitted input contains entity aliases and source support sets; it is
hashed and never inferred from the output.

## Decision gate

Two independent open-weight model families must each satisfy all criteria.
Bootstrap units are questions, never completions.

1. **Ancestry effect:** `cross_ancestor - baseline` collapsed rate has a 95%
   lower bootstrap bound of at least 0.10.
2. **Specificity:** `cross_ancestor - style_only` collapsed rate has a lower
   bound of at least 0.08.  This rejects a style-only account.
3. **Generic-baseline defeat:** `history_aware - mmr` collapsed rate has an
   upper bound no greater than -0.08.
4. **No fidelity barter:** history-aware faithfulness is no more than 0.02
   below MMR, with a 95% lower bound on the difference at least -0.02.
5. **Integrity:** input, raw completions, rows, and gate report all match a
   manifest; the report must be recomputable without access to an answer key.

A failure of any condition kills this candidate.  A pass authorizes only
offline reproduction with a separately generated corpus and external
evaluation; it does not establish an ICLR-ready result by itself.

## Compute estimate

G0 makes `120 x 7 x 8 x 2 = 13,440` short completions plus document
materialization.  With batched generation on a GH200 and a 9B model, estimate
4--8 GPU-hours including retries and evidence packaging.  This is deliberately
far smaller than the original API-heavy RAG Collapse study; it is a mechanism
gate, not a final-scale benchmark.

## Artifact contract

`ancestry-rag-verify` recomputes every gate statistic from
`condition_results.jsonl` and checks the manifest hashes.  It fails closed on
duplicates, incomplete `(question, condition, model)` cells, invalid binary
metrics, or a report that is not byte-for-byte reproducible from raw rows.
