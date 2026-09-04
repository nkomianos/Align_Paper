# Async Control release audit and next data source

Inspected repository tree at `7bc4e699248ed226462def2f2a76c73c5060493a`:
https://github.com/UKGovernmentBEIS/async-control . It contains monitor code,
prompts, a trained ensemble artifact, trajectory-generation/analysis scripts and
a deployment simulator, but no raw `.eval` trajectories in the inspected tree.
The README describes MIT licensing. No attack prompts were decrypted or run.

The full-trajectory scorer passes the available TaskState messages through a
parser to an LLM. The analysis script expects separately supplied `.eval` logs.
Full-trajectory scoring is deliberate for retrospective monitoring, not evidence
of future-information leakage. The paper explicitly models detection windows
and discusses immediate harm and stateful monitoring as deployment issues.
Source: https://arxiv.org/html/2512.13526v1 . Therefore a simple full-versus-prefix
comparison cannot be pitched as correcting an undisclosed flaw in this paper.

ScopeJudge instead releases 100 complete labeled trajectories:
https://huggingface.co/datasets/dreadnode/scopejudge . Its card describes expert
votes at individual tool calls, retained source records and MIT licensing.
Timestamps are replaced during anonymization; they cannot establish actual
wall-clock tool latency. The source system prompt is retained in records but
excluded from the benchmark evaluation prompt. These are important parsing
constraints, not grounds to assume its evaluation is wrong.

Next bounded work: inspect the actual ScopeJudge schema and released prompt
construction to determine what observations precede each labeled call. Do not
execute tools found in trajectories, use included example credentials, or contact
their endpoints. Any analysis remains offline and observational. Distinguish
expert hindsight labels from an online monitor's information, and check whether
the paper already studies that distinction before claiming novelty.
