# External temporal validation: measurement and novelty boundary

The Qwen structured-memory result remains a narrow synthetic positive; the
Nemo replication did not qualify. This audit tests whether a natural temporal
QA release provides the missing external labels. No GPU experiment was run.

## Source evidence

[TORQUE](https://aclanthology.org/2020.emnlp-main.88/) supplies temporal questions
over news passages. The authors' dataset repository was pinned at
ab27019cc6a317fde3c879900499f02acce8b16d. Only README, Apache2.0 license and official
data/dev.json were fetched. Their bytes match the pinned Git tree and saved
SHA256 receipts. Training and test contents remain unopened.

Independent development-file audit found145 passages from79 source article IDs,
1483 questions and571 passage/cluster pairs.434 questions are defaults.323
consensus answers are empty;184 of those have at least one nonempty individual
annotation.1161 questions have nonidentical individual answer sets. All checked
consensus span offsets match passage text and the marked event inventory.
These structural checks do not authenticate consensus semantic correctness.

The annotation structure is a collection of answer spans and individual judgments,
not a gold graph enumerating every consistent chronology. Disagreement can arise
from scope, interpretation, annotation errors or actual uncertainty. It cannot
be relabeled automatically as CLARIFY; neither can an empty answer. The first
inspected passage also contains expectations and possible future events, rather
than only completed point events. The current memory solver does not model those
semantics. Numeric updates and thresholds are not supplied by this source.

Accordingly, TORQUE is not admitted as direct external replication of our current
memory assay. It can support a separately defined temporal QA study. Multiple
questions within one passage and article require clustered analysis. No model
score, new labels or natural-memory claim has been produced.

Reproduce the structural audit with scripts/audit_torque_development.py using
artifacts/torque_source_20260910; DEV_SUITABILITY.json contains exact counts.

## Additional prior-art check

[TG-LLM](https://aclanthology.org/2024.acl-long.563/) and its
[author implementation](https://github.com/xiongsiheng/TG-LLM) already separate
text-to-temporal-graph translation from reasoning over that graph. Rechecked
the primary abstract and implementation overview. This is not a full raw
replication, and its learned reasoning is not identical to our exact solver.
It nevertheless rules out presenting the generic two-stage graph representation
as our new contribution. Standard certain-answer semantics were already noted
in the earlier memory audit.

## Decision

Do not enlarge the memory claim by treating any temporal dataset as equivalent
validation. A paper needs a nontrivial contribution beyond graph extraction plus
existing symbolic semantics, as well as an independently validated task. The
current positive is insufficient, and no natural-data GPU run is admitted from
this source audit. Replacing Nemo's relation encoding also remains a prospective
apparatus repair, not a successful replication or an accepted remedy.
