# GPU cross-vocabulary coupling DEV

This is a separate, larger-model DEV authorized on September 4 for the available
GH200 window. It does not modify or replace the active laptop experiment. It is
not an acceptance gate and does not claim to improve either model's accuracy.

## Fixed experiment

- Qwen3-4B cached revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- SmolLM2-1.7B-Instruct revision `31b70e2e869a7173562077fd711b654946d38674`,
  downloaded publicly with `token=False` into `/home/ubuntu/coupling_models`.
  Weight SHA256 `f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8`.
- Existing eight predetermined public SQuAD development questions; no outcome
  filtering or new parser. Sixteen fixed seeds, four policies, two models: 1024
  completions. Both have reversible byte-level BPE vocabularies, checked by code.
- Same independent, token-clock, byte-clock and first-byte-hierarchical policies;
  temperature 1, full native softmax, EOS stop, 32 native tokens, batch eight.
- Model float32, GPU eager attention, native KV cache; TF32 disabled. This avoids
  introducing a reduced-precision change to the sampling definition merely for
  speed. Cache-vs-full and batch-vs-serial checks must pass before any sampling.
- Model logits remain on GPU. Float64 Gumbel selection and group mass accumulation
  use GPU tensors. The existing deterministic CPU noise stream is transferred to
  GPU; this is an auditable prototype, not an optimized throughput implementation.
  CUDA timing is synchronized. Noise-transfer and group-reduction cost is included.
  Floating point accumulation is not a claim of bitwise CPU/GPU equivalence for
  arbitrary near ties. Ideal arithmetic gives native categorical marginals.

These are practical public checkpoints available quickly, not final modern-model
confirmation. No cross-vocabulary capable pair was already cached: the cached
Qwen3 models share a vocabulary. SmolLM2 is intentionally not described as new.

## Tests and launch requirements

Three CPU tests passed: 800 policy/seed selections and log probabilities agree
with the original NumPy sampler; a 6000-draw marginal-frequency sanity check;
nonfinite logits fail closed. Run the same test file on CUDA before the full job.
The CUDA parameterization is automatic when CUDA is available. Model cache tests
run separately inside the experiment. These tests are engineering qualification,
not a scientific result.

Root coordinator must stage source and the unchanged prepared public inputs,
then launch once into a fresh evidence directory. No worker launched inference.
Use `/home/ubuntu/Align_Paper/.venv-interaction/bin/python` (inspected Torch 2.7.0,
Transformers 5.6.2). With `PYTHONPATH` pointing at the staged `src` directory:

```
python -m interaction_sprint.gpu_coupling_runner FRESH_ROOT --config configs/gpu_coupling_dev_v1.json
```

The generator never opens the separate answer key. The existing committed
`scripts/analyze_squad_coupling.py` accepts this identical evidence schema; its
offline verification requires the original pinned sources, model files and
prepared answer key. All files are manifest-hashed; transfer complete evidence
and compare SHA256 before instance termination. Verification is not full neural
or full sampler replay because this job does not archive all per-step logits.

Provisionally allow 20–60 minutes including qualification and evidence hashing,
but treat this as a planning range, not measured timing. There are at most 4096
batched generation forwards across both models; refine from actual early batch
times. Do not expand automatically or consume the entire eight-hour window on
this alone. Leave enough time to copy and verify evidence before termination.

## Interpretation

Retain the original criteria: assess within-question variance of score A minus
score B, not pooled difficulty; compare against the best simple coupling arm;
audit covariance separately from noisy marginal sample variances; include cost.
Any apparent gain on these eight questions is descriptive. No new task/seed/score
tuning after viewing outputs. A practical gain would justify a larger genuinely
independent confirmation and exact-byte baseline, not a claim of likely acceptance.
