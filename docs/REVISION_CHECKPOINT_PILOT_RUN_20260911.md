# AWS revision pilot run state

At 2026-09-11 00:38:56 UTC, PID 12065 was confirmed live (Rl, elapsed 47 seconds)
on the authorized AWS instance. It loaded the model and entered generation.
This is a live-process observation, not completion or a scientific result.

- Host: ubuntu@44.203.253.82; root: /home/ubuntu/align_research_20260910.
- Output: revision_checkpoint_pilot_v1; log: revision_checkpoint_pilot_v1.log.
- Frozen protocol/runner commit: 353318b.
- Remote runner SHA256: 1661df546dd064b90badfe9ae704c2d45a5a748bdef2b7ec8ff7232eab4dfadd.
- Adapter SHA256 is asserted before the run; base and adapter file hashes are
  written by the runner. No credentials are copied into the job or output.
- The existing 15-minute Monitor AWS ICLR research heartbeat now refers to this
  exact process, output path and protocol. It should not relaunch the process on
  transient observation failure, and should remain quiet during normal progress.

The GPU was confirmed idle before launch. No GH200 connection was made. No
follow-up training is queued. Retrieve the complete output after termination,
verify its manifest and provenance, run the scorer locally, and apply the frozen
capability and interaction gates. Preserve any failed or incomplete run as such.

Transformers reports ignored sampling-only flags during greedy generation;
do_sample=False is explicit. This warning is not a change to the frozen greedy
protocol. No result is available at this recorded observation.

## Pre-scoring design correction

While the job remained live, a direct inspection of the frozen input table found
that nominal bases 4 and 7 are identical: both use binary strings with two ones,
old N=16, and current N=19. All their eight histories coincide. There are seven
unique parameter cases across the two families, not eight unique cases.

The original generation is left unchanged. The frozen nominal-denominator
scores and gates will still be reported, alongside a unique-case table. The
verifier identifies groups from the complete input histories, not output
correctness. A nominal qualifying pattern cannot by itself admit follow-up when
duplicate cases could supply repeated support; its route becomes design
reassessment. No extra cases are added after launch and no independent-sample
significance claim is made. The prospective protocol is retained as originally
written so this mistake remains auditable.

## Runtime observation during the adapter half

The first matched batch contains 10,315 generated tokens for base (longest
completion 1,797) versus 13,242 for adapter (longest 2,823). Its elapsed inference
is about 48.20 seconds for base and 137.33 seconds for adapter, using successive
logged batch timings. This is operational progress information, not an accuracy
comparison or an intrinsic compute-efficiency result. The adapter is unmerged,
and lengths, padding, context growth and implementation overhead differ. No
profiler has isolated their contributions. The original 5–25 minute planning
estimate may be exceeded; the fixed pilot continues without a wall-clock kill.
