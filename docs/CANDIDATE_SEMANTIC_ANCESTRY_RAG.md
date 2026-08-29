# Semantic-Ancestry RAG: G0 Feasibility Gate

## Status

**Killed by the preregistered role-separated G0b feasibility gate.** The
candidate remains a useful negative result, but is not a paper direction in
its current form. The completed G0b evidence was retrieved under a fresh,
non-overwriting local root and independently recomputed after a packaging
defect was repaired in a separate recovery aggregate; neither the original
remote root nor its incomplete aggregate was changed. The candidate was a
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

### Verified G0b result and PI decision

The four raw G0b cells passed their individual manifest, frozen-config,
runtime-preflight, role-plan, and deterministic re-scoring checks. The remote
assembler had omitted `cell_report.json` when copying cells into its aggregate,
so the original aggregate correctly failed closed. The local recovery copied
the preserved source cells into a *new* aggregate using the repaired copier;
the recovered aggregate then verified byte-for-byte from raw completions.

The result is formally `KILL_SEMANTIC_ANCESTRY_CANDIDATE`. Cross-ancestor
collapse exceeded baseline in all four crossed cells (estimates 21.7--53.3
points; all 95% lower bounds at least 11.7 points). This is an encouraging
mechanism signal, not a publishable conclusion: one Mistral/SmolLM3
construction failed both the same-rewriter style control (lower bound -21.7
points) and independent-summary control (lower bound 5.0 points, below the
8-point criterion). More decisively, the frozen history-aware TF--IDF selector
failed to beat MMR in every cell; it was no better in two cells and could be
worse in another. Faithfulness was preserved, but that cannot rescue an
ineffective mitigation.

Therefore do not run G0c, tune the selector, or start external replication
from these outputs. A future project may separately study the robust-looking
response-descendant effect with a new question and preregistration, but it may
not reuse these data to claim a semantic-ancestry mitigation paper.

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
* [Self-Generated Documents for RAG](https://aclanthology.org/2025.findings-naacl.149/):
  studies when a model's generated documents are useful alongside retrieved
  evidence, including document-type and stylistic factors. It does not test
  whether a serving system disproportionately follows a document descended
  from *its own prior answer* after the document's author and surface form are
  removed. G0b's same-question, same-rewriter shadow control is designed to
  isolate precisely that distinction.
* [Generator-Aware RAG Attribution Bias](https://aclanthology.org/2025.findings-acl.1087/):
  measures sensitivity to explicitly supplied source-author identities. This
  is a necessary negative comparator: G0b exposes neither authorship nor model
  labels to the serving prompt, so a positive result cannot be described as an
  author-label attribution effect.
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

### Implemented corrected feasibility protocol (G0b)

The repository now contains a separate, non-overwriting G0b implementation;
it does not alter the preserved developmental G0 root. Its initial purpose is
to decide whether the large Qwen developmental effect survives proper causal
controls, not to establish the final paper result. It uses 60 fictional source
packets, four samples per cell, two serving families, and two crossed external
construction pairs:

| Serving history | Rewriter | Same-question shadow-answer generator |
| --- | --- | --- |
| Qwen3.5-9B | SmolLM3-3B | Granite-3.0-2B-Instruct |
| Qwen3.5-9B | Granite-3.0-2B-Instruct | SmolLM3-3B |
| Mistral-7B-Instruct-v0.3 | SmolLM3-3B | Granite-3.0-2B-Instruct |
| Mistral-7B-Instruct-v0.3 | Granite-3.0-2B-Instruct | SmolLM3-3B |

For each cell, the serving family creates its own ancestor answer. The shadow
model independently answers the *same question* from the same source packets.
The external rewriter applies the exact same rewrite prompt to each answer,
yielding the `cross_ancestor` and `style_only` passages. Thus those conditions
hold rewriter identity, prompt form, question relevance, and source access
fixed; the intended difference is whether the serving model's own prior answer
is the semantic parent. A same-rewriter, source-only summary remains a separate
coverage control.

`scripts/run_semantic_ancestry_rag_g0b_remote.sh` is the only launch sequence.
It freezes a new CUDA preflight, materializes all transformations before serving
evaluation, checkpoints every completion, then verifies all four cells from raw
text. It must use a fresh run root and must not be launched concurrently with a
sealed diagnostic on the same GPU. A G0b pass is still only permission to do an
offline reproduction and external, utility-scored evaluation; a failure kills
this semantic-ancestry formulation rather than being reframed as generic RAG
diversity.

### Precommitted interpretation of a core-effect / mitigation split

G0b separates two hypotheses: the *mechanism* (a rewritten answer descendant
can disproportionately influence the answer) and the current lightweight
TF--IDF selector (a practical mitigation).  If the fully verified G0b result
replicates the causal contrast across all four cells but fails only the
`history_aware - mmr` criterion, that is a failure of this selector--not
permission to tune its 0.20 penalty or weaken its margin after observing the
result. The paper cannot claim this selector as a mitigation.

At most, that outcome may authorize a separately frozen G0c, with a different
deployment-relevant selection problem: a fixed retrieval budget must choose
among base evidence, a **unique but answer-descended** passage, and a
same-rewriter unique non-descended control. Generic MMR must be forced to
retain the unique descendant because it adds query coverage; a history-aware
method may then win only if it rejects that descendant while retaining a
matched useful alternative. G0c must measure both selection behavior and
task-level utility/non-inferiority. It may not reuse G0b inputs, thresholds,
or output-derived parameter choices, and requires a new literature audit and
preregistration before GPU use.

The prospective external pair is
[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) revision
`a07cc9a04f16550a088caea529712d1d335b0ac1` and
[Granite-3.0-2B-Instruct](https://huggingface.co/ibm-granite/granite-3.0-2b-instruct)
revision `5ad66c190631382717bd92d7b052adb1a7b669e7`. Both are Apache-2.0,
ungated, and expose native `AutoModelForCausalLM` loading, so a future pinned
preflight can reject unreviewed custom model code. Phi-4-mini-instruct was
considered but deliberately excluded from this protocol because its official
loading instructions require `trust_remote_code=True`.

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
