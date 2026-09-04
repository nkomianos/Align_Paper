# Nested-prefix prior-belief evaluation: feasibility, not run approval

## Design correction

Replacing early text with late text cannot isolate an effect of *additional*
dialogue: it also removes early evidence. The primary contrast now preserves
all content through the third user turn and adds content through the sixth user
turn. Both prefixes stop on a user message, before the next assistant reply.
Full role-labeled context is primary; user-only context is a sensitivity analysis
because removing assistant questions can make human replies uninterpretable.
Equal-length excerpts remain a separate control, not the primary contrast.

Implemented strict flat-transcript parsing, nested prefixes, and a minimal
prior-rating prompt. It takes only a belief statement and role/content records;
no survey ratings, demographic metadata, incentives or participant identifiers
are accepted as arguments. Transcript text remains untrusted input. Three tests
cover nesting, future exclusion, user-only filtering and invalid boundaries.
Flat markers are still not proof of authentic original message boundaries.

## Outcome-blind feasibility counts

Using the pinned CSV without reading rating values for selection:

- 1,151 released rows.
- 543 lack six user turns; eight fail alternating-marker validation.
- 600 meet structural requirements; 569 also pass the attention check.
- Primary non-personalized subset: 305 rows across 27 query IDs, comprising
  C3=101, C4=104 and C6=100.
- Personalized diagnostic subsets: C1=97, C2=86, C5=81.

Receipt: `artifacts/puppet_schema_20260904_v1/cohort_feasibility_v1.json`.
These are new developmental inclusion rules, **not** a reconstruction of the
paper's 1,035-person cohort. No prediction or endpoint association was computed.
No full prompt dataset was persisted or transmitted.

Requiring six turns conditions on conversation length, which may itself depend
on treatment and the user. This cohort cannot support a clean causal comparison
of randomized conditions. Its target is descriptive paired reader performance
among eligible conversations. Report exclusions and short-conversation sensitivity
separately rather than silently changing the cutoff after seeing model results.

## Primary-work comparison

The full [DToM-Track method](https://arxiv.org/html/2603.14646v1) uses synthetic
LLM–LLM dialogue with scheduled belief changes and multiple-choice questions;
it filters for detectable updates and answerable questions. Therefore prior-belief
recall and a prior/current asymmetry are not new. Independently measured human
ratings and nested evidence interventions are a possible extension, whose
scientific value still depends on the eventual effect and measurement validity.

## Before an inference run

1. Resolve data-use scope and adopt an explicit new-cohort policy without claiming
   to reproduce the source paper's filtering.
2. Freeze query-level DEV/confirmation splits before reading model outcomes;
   never tune prompts or thresholds on confirmation endpoints.
3. Pin model/tokenizer and check every complete paired prefix fits the context.
   Reject oversized pairs explicitly; no silent truncation of early evidence.
4. Add query-only, third-turn, sixth-turn, and time-aware sixth-turn comparisons.
   Evaluate paired pre-rating error as primary; post-rating error and directional
   alignment are descriptive. Include the shared-measurement null.
5. Use local inference, no fine-tuning on participant data, and no persuasion
   optimization. Keep identifiers and raw text out of published artifacts.

The available counts make a modest measurement pilot feasible; they do not prove
adequate power or justify a paper greenlight. No GPU/API process has been launched.
