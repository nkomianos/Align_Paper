# Prospective text-transfer comparator

Status: implemented and CPU-tested; NOT launched. The user was given GPU
termination clearance. Confirm continued availability before a new remote run.

The verifier's full path was exercised using synthetic T2T records against the
real preserved baseline (not new model outputs). It rejects rehashed duplicate
records, incorrect message transfers, invalid timing, dataset labels and caps.
It checks recorded token counts, not independent tokenizer re-execution; the
source-pinned runner is responsible for the recorded tokenization.

The released-C2C run is fully secured and shows a benefit to its smaller receiver,
but the stronger sender alone is substantially better. We must measure the
standard text alternative before attributing practical value to latent transfer.

`scripts/run_c2c_text_baseline.py` uses the same pinned Qwen3-4B sender and
Qwen3-0.6B receiver, isolated environment, upstream prompt builder and 128 public
validation examples as the completed baseline. Each question makes two greedy
64-token calls: generate background from the question alone, then show that
background as the assistant turn between the original question-for-background
and the final multiple-choice question. This is the upstream TwoStageInference
structure; loaders are changed only to pin versions and disable remote code.
The 256 calls record both complete messages and token sequences. No answer key
is read by the runner, no model parameters change, and output roots must be new.

The prospective explicit-label scorer accepts a bare answer letter or a label
followed by punctuation and option text; an explicit `The correct answer is`
prefix is optional. It does not guess a label from free prose or ambiguous
`A or B` output. This fixes the identified scoring mismatch for future work;
historical strict reports remain unchanged. The verifier applies this same
scorer to preserved historical answers and the new comparator. It reconstructs
every prompt and transferred message using the pinned upstream prompt function.

Report answer accuracy, parse failures, background/answer truncation and the sum
of BOTH generation times for T2T. Historical stage times exclude shared model
loading and new times do likewise. This is a serial implementation comparison,
not a serving benchmark or measurement with optimized kernels. If background
generation truncates substantially, record that limitation rather than changing
the budget mid-run. Any alternate budget belongs to a new declared study.

Runner SHA-256: `17cd70b2ff39b8b3f15f0e220d0e23a5fc6988d03c313060a342dcef14644cdd`.
Scorer SHA-256: `f9bb7fb3a112004cf52c4d086ac74c42d76e5006511f9c8dc2cb74dfe4712c63`.
Run as a module from a fresh archive of this repository so its `scripts` imports
resolve; do not rely on the remote older scientific checkout. Preserve source
commit and archive SHA before deployment. Use `PYTHONPATH=src;.` on Windows and
`PYTHONPATH=src:.` on Linux. Read-only verification entry point:
`scripts/verify_c2c_text_baseline.py` with `--root`, `--base-root`, `--prepared`,
`--upstream` and a fresh `--output` outside evidence roots.

Estimated generation time: roughly 5–10 minutes on the same GH200, based on
128 sender backgrounds of at most 64 tokens and the measured prior model speed,
plus short receiver answers. This is an estimate, not a measured run duration.
The comparator does not automatically authorize update training or establish
any new paper claim.
