# Paired update test (Stage B)

Status: implemented and CPU-tested, not run. Stage A has not trained any sender.
GPU access must be reconfirmed after the earlier termination clearance.

## Release and isolation

`scripts/prepare_c2c_paired_ticket.py` runs locally. It re-verifies the completed
text comparator and both Stage A archives with the private qualification key.
Only when both fixed seeds qualify does it emit a release ticket with each
merged model's complete file digests and its original evidence-manifest digest.
The GPU receives the ticket and its separately recorded SHA-256, not private
answer keys. The paired runner rejects an absent seed, failed qualification,
changed ticket, or altered/missing/extra model file. This is integrity checking,
not a cryptographic attestation against a malicious operator.

Use the reserved 256 public validation cases, SHA-256
`66fca7bc7e5725f380a74db8454a1bc49f2304d7b5c267cd3ebadb965cc2e83f`.
Final DEV is disjoint from qualification/training by ID and normalized content;
it is not a new uncontaminated benchmark or locked source TEST split.

## Arms, costs and controls

For each question, evaluate the unchanged receiver, then each sender version:

| Sender version | Arms |
| --- | --- |
| Original pinned sender | Sender alone; frozen C2C; disabled fuser; background generation; receiver with that background |
| Merged qualified update | Same five arms, with the same receiver and fuser weights |

That is 11 generations per question, **2,816 per training seed, 5,632 total**.
Each generation uses the fixed non-thinking chat template, greedy decoding and
64-token cap. Direct-question and background-query token IDs must be identical
across all three tokenizers. Context-augmented text inputs necessarily differ
when the generated background changes; those exact messages are preserved.
Version order alternates by predetermined case index. A fresh generation call
resets the wrapper cache. Only sender membership changes; the receiver and
projector weights never train.

Text-transfer time is background generation PLUS receiver answering. Model
loading, one-time qualification and training costs are reported separately, not
silently presented as free deployment benefits. Serial generation measurements
are not an optimized serving benchmark. Record token caps in both text stages;
do not adjust the cap after seeing the channel contrast.

## Fixed analysis and interpretation

Per seed, report paired changes in sender-alone, C2C and text-transfer accuracy.
Primary descriptive contrast:

`(new C2C - old C2C) - (new text transfer - old text transfer)`.

Bootstrap entire question tuples, stratified by dataset, 5,000 resamples with
seed 202609043. Do not treat arms as independent observations. Report each
training seed separately; question-level uncertainty is not uncertainty over
the population of possible updates.

The following developmental rules are frozen in `DECISION_RULE`:

- All answer arms parse >=95%; both disabled-fuser arms agree with the standalone
  receiver's stripped completion >=98%.
- Original C2C exceeds the receiver by >=5pp on this new slice, establishing a
  usable interface rather than assuming the earlier baseline generalizes.
- Sender-alone accuracy falls no more than 3.125pp on this slice. Qualification
  on a separate set does not guarantee retention on final DEV.
- A candidate extra-loss signal requires contrast <=-10pp, an actual C2C accuracy
  decline >=10pp, and a paired bootstrap upper endpoint below zero. A better text
  arm alone cannot establish bridge damage. Both updates must satisfy this.
- Across seeds, unchanged-reference generated token sequences must agree >=98%.
  A reproducibility failure prevents advancement.

Report every individual result, dataset subgroup, parse failure and token cap.
Do not pool away a failed seed. A below-threshold result means no required signal
on this recipe, not proof that all updated models are compatible. A positive
contrast still does not isolate geometric drift: text content and capabilities
can change differently across channels.

## Commands

Use a fresh committed checkout and the isolated torch 2.7.1+cu128 /
Transformers 4.52.4 environment. All source bytes must match the recorded commit.
On Linux set `PYTHONPATH=src:.`. The entry point is:

```sh
python -m scripts.run_c2c_paired_update \
  --upstream /path/to/pinned/C2C --assets /path/to/verified/asset_evidence \
  --cases /path/to/public/final_eval.json \
  --ticket /path/to/local_issued_ticket.json --ticket-sha256 RECORDED_DIGEST \
  --update-root /path/to/verified/stage_a_seed_202609041 \
  --seed 202609041 --output /fresh/paired_seed_202609041
```

Run the second fixed seed in its own fresh root. Retrieve both complete roots
and compare transport hashes before invoking `scripts.verify_c2c_paired_update`
with both `--root` paths, the local `--key`, the original `--ticket`, pinned
`--upstream`, explicit `--expected-commit`, and a fresh `--output` outside inputs.
Do not overwrite a failed/partial root or auto-expand.

## What is still missing

This stage has **no repair arm**. A repeatable failure would justify the separately
implemented [Stage C repair comparison](C2C_CACHE_REPAIR_PROTOCOL.md), not a paper
go. Its calibration and baseline protocol is declared before training begins:
identity, diagonal/ridge/orthogonal head-local alignment, fuser output matching,
and the text fallback measured here. Existing alignment methods are baselines,
not new contributions. The broader study still needs useful
repair, independent tasks/families, cost comparisons including sender-alone,
and a defensible novelty argument.

Engineering checks include a tiny native Qwen3/C2C smoke under Transformers
4.52.4: old-to-new-to-old sender switching reproduces the original generation,
disabled fusion matches the standalone receiver, and both sender forward paths
are exercised. This is a CPU integration test, not GH200 or scientific evidence.
