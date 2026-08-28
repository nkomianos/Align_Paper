# ICLR 2027 Submission Readiness

**Status on 28 August 2026: not submission-eligible.**  No active candidate
has yet passed its causal gate and independent external replication.  A code
bundle, a local unit test, or a null result cannot support a genuine abstract.

## Official constraints

ICLR's [2027 call for papers](https://www.iclr.cc/Conferences/2027/CallForPapers)
sets an abstract deadline of **18 September 2026, 11:59 PM AOE** and a paper
deadline of **25 September 2026, 11:59 PM AOE**.  The
[author guidelines](https://www.iclr.cc/Conferences/2027/AuthorGuidelines)
require a genuine abstract and state that no new authors may be added after the
abstract deadline.  The submission is double blind.

## Evidence required before an abstract is defensible

1. A differentiated central claim survives the literature screen.
2. The frozen semantic-ancestry G0 result passes for both independent serving
   model families, with raw-completion, preflight, and aggregate verification.
3. The effect replicates on a separately frozen public-source corpus whose
   answer space can actually exhibit evidence concentration.  **No corpus is
   frozen yet.**  HotpotQA is deliberately not the planned transfer corpus:
   its single factual-answer supervision would conflate correct evidence use
   with the diversity-collapse mechanism.  A suitable option, subject to a
   pre-registration and attribution review, is a time-pinned Stack Exchange
   data-dump slice of multi-answer recommendation/explanation questions.  The
   public network terms say the Creative Commons Data Dump is CC BY-SA, and
   Stack Exchange's licensing guidance specifies that post-era licenses and
   attribution requirements must be retained.  See
   [network terms](https://opendata.stackexchange.com/legal/terms-of-service/public)
   and [licensing guidance](https://opendata.stackexchange.com/help/licensing).
   The repository contains a non-executing, CC BY-SA-4.0-only extraction tool
   for this contingency, but no Stack Exchange dump, shortlist, or G1 protocol
   has been frozen.
4. The proposed history-aware selector defeats generic retrieval-diversity and
   context-allocation controls without a faithfulness trade-off.
5. A full anonymous draft reports failures and fixed decision rules, includes
   all model/data/licensing details, and is internally reproducible.

The current candidate satisfies only the *implementation* prerequisite for
item 2. It has **no experimental result**, so it is neither a paper nor an
abstract candidate.

## Compressed decision schedule, if compute is explicitly authorized

| Milestone | Earliest meaningful decision | Consequence |
| --- | --- | --- |
| Run immutable G0 | After a clean host preflight | Any gate failure kills the candidate. |
| Rebuild and verify evidence | Immediately after each family completes | A manifest or raw-rescoring mismatch invalidates the run. |
| Freeze external corpus and replicate | Only after a verified G0 pass | Any transfer failure kills the candidate. |
| PI submission review | Only after both gates pass | Decide whether the evidence merits a genuine abstract. |

No placeholder abstract should be submitted.  If the two experiments cannot be
completed and audited before the abstract deadline, the correct decision is to
continue the research toward a later venue rather than submit an unsubstantiated
ICLR paper.

## User-owned items needed only at submission time

* Final author list and affiliations, with every author maintaining an OpenReview
  profile before 18 September.
* Continued GPU access for external replication and any approved extensions
  after a verified G0 pass.
* Confirmation that the authors accept the relevant dataset licenses and ICLR's
  double-blind submission requirements.
