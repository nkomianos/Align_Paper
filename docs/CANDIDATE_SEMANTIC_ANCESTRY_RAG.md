# Semantic-Ancestry RAG: G0 Feasibility Gate

## Status

**Conditional candidate; G0 is running under a preserved, non-overwriting
evidence root.**  The candidate is a
fast, falsifiable follow-up to RAG Collapse, rather than a claim that generic
retrieval diversity is new.  The central claim is narrower:

> Recursive RAG collapse is caused by the *semantic ancestry* of a retrieved
> document in the system's earlier response, not by the serving model's identity,
> document author label, or surface style; answer-history-aware retrieval can prevent this amplification
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

This makes the contribution conditional and falsifiable: an ancestry effect
that disappears under the same-rewriter independent-summary control is merely a
source-quality/coverage effect; an effect that MMR removes is merely generic
near-duplicate redundancy. Neither supports a paper.

### Active G0 is developmental evidence only

The sealed G0 uses Qwen3.5-9B to form the prior-answer history, Mistral-7B to
rewrite that history, and those same two families as its serving arms. This is
not a data-integrity failure, and the running job must remain unchanged. It is
an identification limitation: in the Mistral serving arm, the `cross_ancestor`
passage is also authored by Mistral.

More importantly, the active implementation's `style_only` passage is an
unrewritten answer produced by Qwen, while its `cross_ancestor` passage is a
Mistral rewrite. It therefore does **not** hold the rewriter's surface style or
transformation process fixed. The `independent_summary` uses Mistral but has a
different direct-summary prompt, so it does not repair that specificity test.

Accordingly, the active run is a developmental, bounded effect-size diagnostic
only. Its raw evidence and the sealed runner decision must be preserved, but
neither a positive nor a negative result is a candidate go/no-go decision. It
can at most justify, or deprioritize, construction of the clean pre-registered
G0b below. This reclassification was recorded before inspection of the
Mistral-family outcome and applies regardless of the observed result.

Nearest work to beat, not merely cite:

* [RAG Collapse](https://arxiv.org/abs/2608.22118): self-authored document
  feedback and a source-quality control, but no semantic-ancestry factorial or
  tested mitigation. Its Section 10 says it cannot distinguish matching content
  from style and specifically proposes testing whether one model over-cites
  another's generated references. Section 15 calls generic diversity a possible
  mitigation but does not test it; Section 15.2 identifies human-edited
  answer-derived "knowledge collapse" as a case that AI detection cannot catch.

  A source-level audit of the released v1 TeX on 2026-08-28 confirms this
  boundary rather than relying on the abstract: Section 10 controls eight
  LLM-judged quality dimensions and AI-generation status, then concludes that
  it cannot tell whether the residual preference is matching reasoning/content
  or style, and lists cross-model citation as future work. Its mitigation
  section only lists filtering and generic diversification; it contains no
  factorial test or answer-history-aware selector. This audit does **not** make
  the present candidate novel by itself: G0 still has to establish an
  author-invariant ancestry effect and a history-aware gain over MMR.
* [Self-Preference in RAG](https://aclanthology.org/2025.findings-acl.1369.pdf):
  static authorship and factuality controls, with largely null final-generation
  self-preference; this is the critical alternative account G0 must overturn.
* [Spiral of Silence in RAG](https://aclanthology.org/2024.acl-long.798.pdf):
  iterative accumulation of generic LLM-generated text and retriever dominance.
  This means an ordinary feedback-loop reproduction is not publishable; G0 must
  isolate an author-invariant, response-descendant effect and show that a
  history-aware selector beats its generic retrieval controls.
* [Retrieval Collapses When AI Pollutes the Web](https://arxiv.org/abs/2602.16136):
  ecosystem-level synthetic-content exposure and source-diversity loss.  This
  candidate cannot be presented as another web-pollution result: it must show
  an answer-history causal effect under matched relevance, style, and source
  coverage, then reproduce that effect on a real corpus whose provenance is
  independently auditable.
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
| `self_ancestor` | The RAG system's primary response model expands its previous response into a passage. |
| `cross_ancestor` | Another model rewrites that same response, preserving its claims but changing style/authorship. |
| `style_only` | The primary response model writes a passage from an independently sampled answer; style matches but semantic ancestry does not. |
| `independent_summary` | The same rewriter independently summarizes the original packets without ever seeing the earlier answer; this matches source-derived semantic coverage without response ancestry. |
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

Two independent open-weight serving-model families, evaluated on the same
frozen system-history corpus, must each satisfy all criteria.
Bootstrap units are questions, never completions.

1. **Ancestry effect:** `cross_ancestor - baseline` collapsed rate has a 95%
   lower bootstrap bound of at least 0.10.
2. **Specificity:** `cross_ancestor - style_only` collapsed rate has a lower
   bound of at least 0.08.  This rejects a style-only account.
3. **Lineage versus content:** `cross_ancestor - independent_summary` has a
   lower bound of at least 0.08.  The control is written by the same rewriter
   from the original packets but without exposure to the first answer, so a
   positive effect cannot be attributed only to a comprehensive semantic
   summary or to generic rewriter behavior.
4. **Generic-baseline defeat:** `history_aware - mmr` collapsed rate has an
   upper bound no greater than -0.08.
5. **No fidelity barter:** history-aware faithfulness is no more than 0.02
   below MMR, with a 95% lower bound on the difference at least -0.02.
6. **Integrity:** an immutable CUDA/model-class/revision preflight is bound to
   each family root. Inputs, raw completions, deterministically re-scored rows,
   and gate report all match a manifest; the report must be recomputable without
   access to an answer key.

The sealed runner may emit `KILL_SEMANTIC_ANCESTRY_CANDIDATE`; that string and
all evidence must be preserved. After the recorded control audit, however, it
is not a scientific candidate decision. The next test is G0b, not external G1.

G0b must be frozen before launch and must: (1) have each serving model create
its own prior-answer history; (2) use an external rewriter that is neither
serving family; (3) make `cross_ancestor` by passing that prior answer through
the rewriter; (4) make `style_only` by passing an independently generated,
same-question shadow answer through the **same rewriter and rewrite prompt**;
and (5) counterbalance two independent external rewriter/shadow-answer model
pairs. The independent-summary control must use the same rewriter, with a
matched source-only construction, and all transformations must be materialized
before either serving model is evaluated. This makes the ancestry contrast
separable from rewriter identity, output form, and generic source coverage.
Only a positive, fully role-disentangled and style-matched G0b can authorize
the external replication protocol.

## Compute estimate

G0 makes `120 x 7 x 8 x 2 = 13,440` short completions plus document
materialization.  Its current sealed runner generates one completion at a time
to preserve per-cell evidence, so the credible planning range is **several
GPU-hours per family**, not a sub-hour smoke test.  That cost is deliberately
far smaller than the original API-heavy RAG Collapse study; it is a mechanism
gate, not a final-scale benchmark.  Any speed optimization belongs in a new,
equivalently verified protocol revision—not in the active run.

## Artifact contract

`ancestry-rag-preflight` validates the exact pinned model classes, revisions,
CUDA host, and runtime dependencies before weights are downloaded. Each serving
run copies that attestation into its evidence root. `ancestry-rag-verify`
recomputes every gate statistic from the raw completions, checks the manifest
hashes, and rejects an unbound preflight. It fails closed on duplicates,
incomplete `(question, condition, model)` cells, invalid binary metrics, edited
scored rows, or a report that is not byte-for-byte reproducible.

For any *future* run started from the current code, the runner writes a
non-analyzable `RUNNING.json` and appends each raw completion to
`raw_completions.partial.jsonl` as it is generated.  These files preserve
interrupted work but cannot be assembled or verified as a result; only a
complete run atomically promotes the raw file and emits a manifest.  This
durability improvement does not modify the already-running frozen G0 root.

The initial source packets are deterministic fictional entity-comparison
contexts generated by `ancestry-rag-build-base`.  This makes source support and
entity scoring exact while eliminating web memorization.  A G0 pass must then
replicate on a separately frozen real-source corpus before the candidate can be
considered a paper.

## Conditional external-replication preparation

The repository now includes `ancestry-rag-extract-stackexchange`, a
fail-closed **source-packet extractor**, not a G1 evaluator. Given a specific,
time-pinned Stack Exchange `Posts.xml` dump, it retains only explicit CC BY-SA
4.0 questions and answers satisfying predeclared score, date, length, and
multi-answer criteria; it emits source URLs, licenses, dump and output hashes,
and a deterministic snapshot-keyed selection order. It refuses to overwrite a
shortlist or manifest.

This shortens the turnaround after a valid G0 pass, but it intentionally does
not create G1 model inputs, answer aliases, condition passages, or a decision.
Those choices must be frozen only after the verified G0 result, together with
the selected community, snapshot identifier, attribution appendix, semantic
scoring design, and independent-review plan. A shortlist alone is never
external evidence and must not be cited as a replication.

### Required G1 safeguards if G0 passes

G0's faithfulness check is deliberately only a **non-inferiority guard**: a
history-aware selector can lower the collapse event simply by omitting the
descendant passage.  It does not establish that the selector preserves answer
utility.  Thus a positive G0 cannot support a mitigation claim until the
external protocol pre-registers all of the following:

1. a task-level, ground-truth answer-utility endpoint in addition to
   source-support faithfulness, with an equivalence/non-inferiority margin
   relative to MMR;
2. a retrieval-quality and coverage audit, including the fraction of queries
   on which the selector excluded the descendant and the relevance loss of
   each selected packet;
3. a content-matched history control (the same selector penalty against a
   non-ancestral answer of matched query relevance), so that generic removal
   of a useful summary cannot masquerade as semantic-ancestry mitigation; and
4. evaluation on a held-out question set whose scoring rules and source-packet
   construction are fixed before any external serving result is examined.

These are confirmatory safeguards, not criteria that may be relaxed to rescue
G0.  If the frozen G0 is positive but this stronger external protocol cannot
show utility preservation and ancestry-specific benefit over its controls, the
candidate is killed rather than reframed as a generic diversity heuristic.

Importantly, multiple answers to one Stack Exchange question are **not**
independent sources for a corroboration claim. This transfer setting would test
whether the answer-ancestry mechanism survives naturally written text and a
public data license; it cannot support a claim about independent-source
authority. Any such authority claim would require a separate provenance-aware
dataset and a new protocol.

Preparation is staged and evidence-preserving: `ancestry-rag-prepare` first
has a fixed *system response model* answer the base packets, has a separately
named rewriter render the relevant and irrelevant descendants, and freezes the
shared corpus before either serving model is evaluated.  The serving runner,
assembler, and verifier then operate only on that frozen corpus.  No author or
model label is supplied in any serving prompt.

## Launch-ready model plan

The frozen defaults are [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) and
[mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3).
Qwen3.5-9B is the requested recent 9B model; it is Apache-2.0 and the official
card supports direct Transformers loading.  Its multimodal default loader is
not permitted in this study: every Qwen arm is pinned to revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, native text-only
`Qwen3_5ForCausalLM`, and `enable_thinking=false`.  Mistral-7B-Instruct-v0.3 is
a distinct Apache-2.0 family, pinned to
`c170c708c41dac9275d15a8fff4eca08d52bab71`, and gives
the required independent serving-model test.  Neither selected repository is
gated, so a Hugging Face token should not be needed.  This is preferable to
using Llama or Gemma here: their official Hub repositories require accepting a
gated license, which would make a token and account state part of the gate.

The non-overwriting execution sequence is:

1. Run `ancestry-rag-preflight` against the frozen config; preserve its new
   JSON attestation.
2. Build the 120 source packets with `ancestry-rag-build-base`.
3. Prepare one shared answer-history corpus with `ancestry-rag-prepare`.
4. Run `ancestry-rag-run` once per serving family, binding the preflight output.
5. Merge exactly those two roots with `ancestry-rag-assemble` and independently
   recompute the result using `ancestry-rag-verify`.

Qwen's current model card documents a multimodal loading route; the gate
instead requires the repository's native Qwen3.5 text class because a visual
wrapper could silently change the tested model.  The remote environment will be
checked before download or generation; a dependency or model-load failure is a
hardware/software preflight failure, not a scientific result.

The exact one-shot, non-overwriting host sequence is
`scripts/run_semantic_ancestry_rag_g0_remote.sh`. Set
`ANCESTRY_RAG_RUN_ROOT` to a fresh absolute path and run it only after the
repository is at the frozen commit. It preserves the preflight, prepared corpus,
both family roots, aggregate, and independently recomputed verification result.
