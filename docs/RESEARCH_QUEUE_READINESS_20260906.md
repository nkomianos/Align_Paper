# Readiness before renting GPU time

**Historical snapshot, superseded:** the four exploratory runners are now
implemented. See [current implementation and execution contract](RESEARCH_PILOTS_IMPLEMENTATION_20260906.md).
The protocol-only table below records the earlier state; it is not current readiness.

The full research queue is NOT deployment-ready. The research shortlist is a
set of hypotheses and protocols, not four implemented experiments.

| Stage | Current readiness | Runtime / spending limit |
|---|---|---|
| Hindsight three-arm calibration | Existing code, verifier, data package, wheels and CPU tests. GPU path still needs host qualification. | Planning estimate 10-30 minutes, unbenchmarked; 1 hour hard cap. |
| Compensating-update pilot | Protocol only; no frozen data, runner or verifier yet. | Proposed 1-2 hours, 3 hour cap; not launchable. |
| CLARA rule-edit pilot | Algebra audit code only; baseline comparison and natural-data qualification pending. | CPU first; no GPU allocation yet. |
| Unreliable-reference pilot | Protocol only; organism/reference protocol unresolved. | Proposed 2 hour cap; not launchable. |
| Specification monitor | Protocol only; dataset labels/splits and baseline audit pending. | Proposed 1-2 hours; not launchable. |

`python scripts/research_queue.py` prints machine-readable readiness.
It can launch only the existing Hindsight calibration, after bounded offline
host/input checks. The actual calibration supervisor enforces allocation time,
process termination, and the 50-hour ceiling. It stops after that diagnostic;
unfinished entries cannot be started by passing their names. The queue wrapper
does not stop cloud billing and has not been GPU-tested.

For deployment use the original verified calibration package and
`CALIBRATION_FAST_DEPLOY_20260905.md`; the new queue wrapper is optional and is
not retroactively part of that package's source hash. Do not overwrite or relabel
the existing verified bundle. A new package must be sealed after new executable
pilots are complete.

Preparation estimate, not a delivery guarantee: several focused hours for one
new properly tested pilot; roughly 1-2 working days for multiple candidates,
potentially longer if data or model-organism prerequisites fail. Developing all
four before selecting any is not the efficient route under 50 H200 hours.
Prioritize the compensation implementation and CLARA CPU falsification; keep
the remaining ideas as conditional research entries.

If only running the prepared calibration, allow approximately 5-15 minutes for
transfer/import checks on a compatible instance with cached weights. A missing
~19 GB model cache or incompatible image makes that estimate inapplicable.
There is no reason to rent an idle GPU while CPU-only implementation continues.
