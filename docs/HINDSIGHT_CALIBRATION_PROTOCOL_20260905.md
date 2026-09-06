# Learning-only acquisition diagnostic, v1

Status: prospective, NOT RUN. This is apparatus calibration, not an ICLR result.
User budget on5September: approximately50H200 hours, superseding the earlier
100GH200-hour ceiling for future work. No host is contacted during preparation.
The user will turn on the instance; do not start or keep a rental alive for this work.

## Question and scope

Does direct supervision acquire the fixed old label, and does freezing the
privileged teacher change the failed updating-teacher behavior? The prior
70-update assay is invalid acquisition, not a residual-method negative. This
diagnostic cannot establish latent preferences, causal welfare, or released
multi-token SDPO performance. Frozen-vs-current isolates target dynamics within
our first-answer-token reverse-KL recipe; it does not establish a general cause.

Only the previously used learning.json is read, pinned to SHA256
986b2241a2a3137cf4d97383d97a9d1ac84d8dc0f482dcb1f5ed4199cb065a81.
No old DEV or confirmation file is needed or packaged. A hash-only ordering of
630 base IDs with salt learning-only-calibration-20260905-v1 selects128training
bases, then32held-out learning bases;470are unused. All four rotations stay
together. These are post-audit DEVELOPMENTAL splits of already-exposed data,
not new independent confirmation or unseen-user evidence. Display-name groups
are not verified person identities. All held-out tasks remain fixed after launch.

## Frozen execution

Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a,
BF16, SDPA, no generation, no kernels/cache, thinking disabled, no truncation
above1024tokens. Common hypothetical-task prefix is unchanged from v2.
Same rank8/alpha16 adapters, identical initialization, new AdamW state each arm,
LR1e-4, betas(.9,.999), epsilon1e-8, no weight decay. Eval model mode with
student gradients. Seed202609051. Each arm sees512rows once in32updates,
effective batch16, microbatch2. Each effective batch contains four complete
rotated bases: exactly four A/B/C/D targets. Accumulate weighted microbatch
mean gradients, clip once at1.0, then update once. No LR sweep or retries.

Arms: supervised full-vocabulary native-label cross-entropy; reverse
KL(student||initial frozen privileged teacher); same reverse KL against the
current detached privileged teacher. Every arm has access to the same128base
labels. Freeze teacher logits at initial weights before ANY optimizer exists;
reuse their exact full-vocabulary float32 bytes. The comparison matches student
updates/examples, not forward compute: report caching/teacher/evaluation costs.
No teacher-swap operation occurs with a live student graph.

Before training, score initial student and privileged teacher on128held-out
rotation rows. Teacher must have at least28/32correct in each rotation,
mean conditional correct probability>=.70, choice mass>=.10, and mean paired
conditional probability range<=.20. Failure stops with invalid interface;
no third prompt repair is authorized by this protocol.

The same fractional teacher gates must also pass on all512training rows while
caching their initial logits (at least112/128correct per rotation). This prevents
mistaking bad training targets for a failure of distillation. Caching and scoring
use the same forward calls; no repeated teacher-inference stage is necessary.

Score both views at steps8,16,32; only step32 controls the decision. Save final
training scores to distinguish memorization from held-out acquisition. All
forward logits, IDs, actual token arrays, rendered prompts, intermediate/final
adapters, optimizer states, configuration, source closure and model hashes are
retained. No p-values or seed replication are inferred from label rotations.

## Routing and kill criteria

Per arm, acquisition requires held-out full-label NLL gain>=.10 over initial
student, strictly positive full/conditional correct-probability gains, and mean
paired conditional range<=.20. These are practical routing thresholds, not
statistically powered tests. No best-checkpoint selection or gate relaxation.

1. Supervised fails: stop method expansion. Inspect training-vs-held-out evidence;
   distinguish task generalization/budget/implementation without assigning cause.
2. Supervised passes, frozen fails: investigate objective or information transfer;
   no justification to blame moving-teacher feedback alone.
3. Supervised and frozen pass, current fails: target-dynamics investigation is
   warranted within this recipe, especially if privileged capability also falls.
4. All pass: apparatus qualifies narrowly; still require a distinct contribution
   and consequential external task before label-efficiency experiments.

Teacher capability is reported at every checkpoint and finally for every arm;
it is a diagnostic, not a retroactively added acquisition gate. No further
experiments are automatically launched for ANY outcome. Nonfinite values, OOM,
token overflow, timeout or interrupted execution are invalid runs, not negatives.

## Budget and deployment

One GPU only, at least90GiB available device capacity; microbatch2 reduces
activation risk. H200 throughput is NOT inferred from previous GH200 timings.
Planning estimate10–30minutes with cached weights, but unbenchmarked for this
runner; independent supervisor enforces<=3600seconds including model load,
forward logging, scoring and sealing. Hard-kill grace is inside this cap.
Require30GiB free disk for raw logits. No online download occurs in the run.

The launcher also limits time by the actual allocation start and previously
spent H200 allocation hours against50hours. Idle/setup time counts. There is
no scientifically valid fixed conversion from GH200 to H200 hours; keep the
historical GH200 receipt separate and include any provider-charged new rental
usage in the new ledger. An idle rental still bills; process termination is
not provider instance termination. Do not leave the instance running between
research decisions. This launcher cannot terminate the provider instance.

Suggested conditional allocation:1h calibration,4h candidate screening,20h
main methods/controls,15h independent replication/confirmation,5h robustness,
5h setup/failure reserve. These are ceilings, not automatic authorization of
an undefined queue. Re-estimate from the first measured run and stop losers early.

## Verification limits

CPU tests establish data exclusion, balanced schedules, decision boundaries,
teacher detachment, gradient accumulation, resets and timer arithmetic. Linux
process-group termination is tested separately. Full H200 model loading and
training are explicitly NOT tested until the instance is available.
The offline result verifier checks sealed bytes, source rows, saved-logit
arithmetic, evaluation contexts, schedules and routing. It does not claim a
neural checkpoint replay or independently establish actual training execution.
Float32 CPU/GPU score arithmetic permits absolute discrepancy up to1e-5;
the verifier recomputes the routing from raw-logit-derived summaries and requires
the same discrete decision. A threshold crossing is not waved through using
that numeric tolerance. Saved token inputs are preserved for later inference
replay; this small verifier does not certify their source-to-token derivation.
