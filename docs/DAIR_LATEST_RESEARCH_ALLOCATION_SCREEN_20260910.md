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
