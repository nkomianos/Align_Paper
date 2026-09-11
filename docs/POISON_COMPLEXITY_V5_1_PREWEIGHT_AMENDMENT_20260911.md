# Poison-complexity v5.1 pre-weight metadata amendment

The frozen v5 pre-registration correctly selected the four non-deduplicated
Pythia repositories and their final `step143000` checkpoint labels. During an
independent metadata replay, before any Pythia tokenizer or model weight was
loaded, two manually transcribed resolved commit IDs failed to match those
registered branches.

V5 remains preserved at commit
`30dcb2300d98700aa688daf6d8ab5442f73bb7ed`. V5.1 prospectively corrects only
the two resolution strings:

| Model | Incorrect v5 transcription | Correct `step143000` commit |
|---|---|---|
| `EleutherAI/pythia-410m` | `38b3436f703f7e2793d7a4f742b46e38db725dd6` | `bba6a464f54bbf08fc174cfb351d9794d58af21d` |
| `EleutherAI/pythia-1.4b` | `7ba6d9aa3926be2241e6e1e45f42b99c27514c19` | `9cc5c8c8148a4e0115d9e29c6b4f21124cfe748a` |

The 160M and 2.8B pins already match their registered `step143000` branches and
do not change. The repositories, four model sizes, checkpoint step, trigger,
estimand, functions, row construction, optimizer, thresholds, exclusions,
replication plan, staging plan, and advance rule remain byte-bound to the v5
base configuration and pre-registration.

The execution harness must load the frozen v5 base configuration, verify its
hash, load this amendment, verify its hash, require the two incorrect values to
be present, and replace only those exact values. Any further mismatch stops
execution. V5.1 is a prospective clerical correction, not post-result model
selection; no model-dependent output existed when it was made.
