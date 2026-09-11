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
