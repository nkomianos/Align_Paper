# Agent-security re-scout: avoid restarting occupied directions

This is literature triage, not experimental evidence. No attack was executed,
external system targeted, or GPU job launched.

Three broad candidate framings were checked against primary-paper records:

1. Persistent-memory revocation and resurrection: OBLIVION already introduces
   workflow-level revoked-skill reconstruction from residual carriers.
   https://arxiv.org/abs/2608.08264 . MemSecBench evaluates memory poisoning across
   persistence, consequence, and repair: https://arxiv.org/abs/2607.27080 .
   Do not reopen generic removal/resurrection as a new benchmark claim.
2. Asynchronous monitoring races: Async Control already stress-tests asynchronous
   control of agents and releases a harness:
   https://arxiv.org/abs/2512.13526 . A timing attack alone is not a new contribution.
3. Benign authorization overreach: Overeager Coding Agents studies out-of-scope
   actions on benign tasks: https://arxiv.org/abs/2605.18583 . ScopeJudge is a
   directly relevant cost-aware pre-execution monitoring benchmark with expert
   tool-call labels: https://arxiv.org/abs/2607.07774 . An ordinary scope classifier
   or another synthetic permission test would be insufficient.

These overlaps do not exhaust agent-security research. They change the next
action: inspect released real execution traces and the information actually
available to monitors at each decision, rather than invent another toy prompt
injection suite. A possible question is whether monitor evaluations remain
valid under delayed tool returns and partial observability, but its exact novelty
and the ability to reconstruct decision-time observations are unverified.

Next bounded audit: inspect Async Control's released schema, licenses, and trace
availability; determine whether it already measures that question and whether
any useful counterfactual analysis is possible without running an attacker.
This is a lead, not an experiment queue entry or paper greenlight.
