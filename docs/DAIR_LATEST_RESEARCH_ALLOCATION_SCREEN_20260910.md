# Latest weekly list: research-allocation follow-up

The user-specified DAIR list was pinned at
789f644f9cf213c192e5625fecbc57e580a4cced. Its latest listed week is
August31–September6. The ten titles/links were inventoried; this does not mean
ten primary methods were audited or ten experiments were run. Existing
Declarative Attention and Trace-as-State assessments were not rerun. The only
new primary follow-up in this check was the research-preference paper below.
No security experiments or payload investigations were pursued.

[AI Research Preference Models](https://arxiv.org/html/2608.13940v1) uses frozen
LLMs to choose among unexecuted candidate implementations, optionally using
small pilot experiments. Its main evidence concerns online child selection
within AIRA-dojo. It separately studies selection among already executed nodes:
AppendixD finds little benefit over choosing by validation score, and explicitly
distinguishes this boundary from its main result. The limitations acknowledge
offline selection bias, inference overhead and limited demonstrated portability.
Thus neither generic pilot-based experiment selection nor the final-selection
negative is a new discovery available to us. No benchmark results were replicated.

Our inference: an RPM's preferred candidate is not a certificate of novelty,
scientific validity or acceptance likelihood. Benchmark score optimization
requires a fixed trustworthy endpoint; our outstanding problem is establishing
a differentiated, valid research question. Adding another LLM judge would not
resolve missing controls or external labels. The paper may inform workflow,
but does not justify a GPU campaign or a replacement for independent evidence.

No new qualified experiment emerged from this bounded check. Uninspected weekly
titles remain uninspected leads, not negatives. Source files and hashes are in
`artifacts/dair_latest_screen_20260910/`. The initial console preview failed on
Unicode encoding after successful file saving; the subsequent preview succeeded.

## 12:12 follow-up: state and skill-evolution boundaries

Two more primary methods/limitations were read, without executing their systems.
This increases the new primary follow-ups from the latest list to three; it does
not mean the other seven entries have all been reviewed or tested.

[SKILL.state](https://arxiv.org/html/2608.26263v1) uses a domain schema, structured
state patches and the latest observation instead of a growing transcript. Its
limitations already identify unknown schemas, observations whose relevance is
recognized later, historical-output tasks and concurrent updates. It also
reports state-overwrite and schema errors and proposes constrained decoding.
Generic state compression or syntax repair is therefore not our new contribution.
This check covered benign state semantics and complexity only; no security
benchmark, exploit example or payload was investigated.

Our mathematical qualification: total prompt work is proportional to
sum_t (|P|+|state_t|+|observation_t|). A fixed number of schema fields alone does
not bound token size: one field can contain an ever-growing list. A linear-in-
horizon claim therefore needs bounded serialized state and observations, or must
report their actual growth. This is an elementary accounting condition, not a
new theorem or evidence that its reported measured savings are wrong. We have
not established state growth in its released implementation.

[WikiSkill](https://arxiv.org/html/2608.27454v1) separates immutable traces,
accumulated wiki knowledge and reversible skill changes. Its stated limitations
include direct skill injection rather than retrieval, strict validation
improvement excluding neutral proposals, no automated wiki pruning, and limited
very-long-horizon evaluation. Generic validation-gated skill evolution is already
covered. None of these acknowledged limitations by itself establishes a new
research gap or a failure of the published result.

Possible follow-up hypotheses would concern the benefit of neutral intermediate
skill changes or bounded knowledge retention under evolving objectives. These
require comparisons to established evolutionary search and memory selection,
independent validation after adaptive selection, and an accessible task-specific
endpoint. No such experiment is implemented or admitted by this source screen.

## 12:46 follow-up: selective forgetting

[Selective Forgetting](https://arxiv.org/html/2608.28978v1), Sections3–5 and
AppendixA.1, already reports that its extraction-based graph loses to flat
retrieval. The paper scopes this to its pipeline, not all graph memory.
Its retention experiment prunes once after ingesting all500 haystacks into one
persistent graph, then evaluates questions; it is not repeated pruning over a
prospective deployment stream. The text acknowledges cross-conversation
interference and names a same-size random-pruning control as missing future work.
It also identifies stale-attribute handling as a weakness. Generic pruning,
verbatim preservation and latest-value correction therefore do not establish our
novelty. None of its results were independently reproduced here.

Our decision: do not turn this into another expensive graph-extraction campaign.
A future retention claim would require a matched storage budget, random and
recency controls, query/context token accounting, conversation isolation, and
new query batches after each retention decision. Different objectives need
separate evaluation: answer quality, raw storage and simultaneous correctness
are not interchangeable. A small interval including zero is not proof of exact
performance equivalence. This is a baseline/protocol requirement, not a new
method or a demonstrated flaw in the reported paired comparison.

Four new primary follow-ups from the latest weekly list have now been reviewed.
No new GPU-ready candidate emerged. Primary HTML and hash receipt saved alongside
the other weekly-screen artifacts. Do not count these source screens as model
experiments, or infer that all remaining weekly entries were tested.

## 13:26 follow-up: WikiSkill release and counterfactual feasibility

Exact-name searches for WikiSkill plus GitHub, code/release, and author Liyan
Tang, together with the [primary paper record](https://arxiv.org/abs/2608.27454),
did not establish an author-released experimental trace archive. This is a
bounded search result, not proof that no archive exists. The readily found
[implementation](https://github.com/ashutoshsinghpr7/wikiskill) is a separate
implementation of the paper, not authenticated original experimental evidence.
Its README advertises a small demonstration benchmark and live runs; those
claims and implementation fidelity were not independently reproduced. Nothing
was installed or executed from external repositories.

The proposed neutral-update question has an additional identification barrier.
An archive containing a rejected neutral proposal and its immediate score does
not reveal the quality of descendants that would have been proposed after
accepting it. Re-scoring the same candidate cannot measure that longer-term
effect. Nor does a recorded zero validation difference establish population
equivalence. These are our methodological deductions, not reported failures of
WikiSkill's experiments.

The smallest informative future assay would branch from a common frozen parent
and proposal, randomly retain or reject an eligible neutral update, then spend
equal proposal and evaluation budgets on both branches. The primary endpoint
would be independent held-out terminal utility, with the parent/task as the
cluster rather than each descendant. A no-change branch and a budget-matched
search control are necessary; qualification and neutrality rules must be frozen
on development data. Repeated validation selection requires a separate final
evaluation set. A single constructed success or synthetic search landscape
would remain developmental. Runtime is not estimable from the currently
available evidence without a chosen capable task, model, and pilot throughput.

No offline shortcut or differentiated method has been established, so this
candidate remains unrun and is not admitted to the GPU queue. Generic neutral
evolution is not claimed as novel. No reason to repeat this release search
without new author-linked availability evidence.

## 13:42 follow-up: CORAL attribution and feasibility

Read [CORAL2609.02730v1](https://arxiv.org/html/2609.02730v1), Sections3–7.
Its production harness combines LLM proposals, persistent memory and deterministic
budget projection. Both deployments report A/B comparisons. Cadence and memory
horizon are defaults; systematic selection is acknowledged future work. Reported
token costs are estimates, not logged usage. Standardized predeployment evaluation
is explicitly identified as an open problem.

Our inference: successive rounds of an adaptive system do not alone isolate
memory's causal contribution; changed context and exploration also matter. A
matched memory/no-memory comparison would address that attribution. This does
not invalidate the reported deployment A/B effects. Likewise, pre/post change
alone cannot establish causal attribution under drift, whereas concurrent
randomization can address that particular concern. The inspected description
does not supply enough detail to audit every attribution tool output.

No accessible production endpoint or raw randomized records were established
here. A synthetic resource-allocation simulator could qualify infrastructure,
but would not reproduce these deployments or by itself yield a novel paper.
No campaign admitted. This brings new primary follow-ups from the latest weekly
list to five; source review is not an experiment.

## 13:58 follow-up: Harness-of-Harness evaluation boundary

Read [Harness-of-Harness2609.01481v1](https://arxiv.org/html/2609.01481v1)
method overview and AppendicesB.2,B.5,B.7,B.9. It already compares repeated
vanilla coding passes with planning/coding/testing loops. Its fixed terminal
artifact is not selected using benchmark scores; benchmark evaluator outputs
are withheld from development. Generic independent testing and repeated
continuation are therefore occupied controls, not a newly discovered remedy.

The comparison matches development passes, not exact total model work. Token
accounting is reported separately and excludes benchmark evaluation. Each
task-condition has one valid run; the stated bootstrap resamples tasks together
with their score components. That interval does not measure repeated-run
variability conditional on a fixed task. These are scope distinctions, not
evidence that its numerical results are wrong or that it used pseudoreplication.

The described anonymous package excludes raw run artifacts and benchmark data.
No raw replication performed. Published frontier harness/model runs are not a
measured local8B capability qualification or a GH200 runtime estimate. A new
study would need a distinct intervention, fixed-budget evaluation and repeated
independent runs. No such candidate emerged. Six new primary follow-ups from
the latest weekly list are now screened; no GPU work admitted.
