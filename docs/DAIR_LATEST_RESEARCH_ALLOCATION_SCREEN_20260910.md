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
