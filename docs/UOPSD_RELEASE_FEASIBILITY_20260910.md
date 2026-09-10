# Author-linked u-OPSD adapters: available, audited, not behaviorally qualified

The [author project page](https://williamium3000.github.io/u-opsd/) links to the
[u-OPSD Hugging Face organization](https://huggingface.co/u-opsd), which exposes
four adapters. This corrects any broad inference that no usable post-training
checkpoint exists in this research area. It does not supply the unavailable
checkpoint from the earlier thinking-collapse paper or prove collapse here.

The author's code repository at8140c234586309a183bfb21cd6a298aeafa44f8d still
announces a future code release. Models and training implementation availability
are separate facts. Discovery used a bounded100-result public model-name search;
this is not an exhaustive Hugging Face inventory.

Prepared locally, without GPU use:

- Model: u-opsd/qwen3-8b-thinking, revision16ba69e32700ddaeb03ec0a2cf717dfca89296b9.
- Adapter:349,244,256bytes, published LFS SHA256
  84917a93ee7b9840bb6c21ea6918f6102d13844ccbbd121c3bf3a840faf57c7c, verified.
- Safe tensor CPU audit:504 finite, nonzero tensors;252 rank64 module pairs.
- Every A/B matrix matches the expected projection dimensions across all36
  layers of our pinned Qwen3-8B base configuration. No external pickle loaded.
- Trainer state global_step75 agrees with the model card. The historical base
  revision is null, so shape compatibility is not historical base authentication.

The [8B card](https://huggingface.co/u-opsd/qwen3-8b-thinking) describes threshold
0.3, random disagreeing target selection, a shortest agreeing reference, and a
step75 checkpoint. Its reported thinking-mode improvement is small and single-seed.
The [4B card](https://huggingface.co/u-opsd/qwen3-4b-thinking), inspected at
88113d15d161ebc3c1873b7d9badf50005012b28, describes an all-agree variant, longest
agreeing reference and step25 checkpoint. Its weights were not downloaded.
These are different training variants, not a matched model-size replication.

The [paper's Appendix D](https://arxiv.org/html/2608.06296v1) says unanimous valid
rollouts contribute no loss under its described setup, whereas the released4B
card describes training on all-agree prompts. This is a protocol correspondence
question, not proof that a published score is false. Do not label either adapter
as the default algorithm without stating its released configuration. Code-level
historical verification remains unavailable in the inspected author repository.

Next admission: use the adapter only after defining an outcome-level question
distinct from existing reference-bias, uncertainty-suppression and math-perturbation
work. Freeze fresh development groups, native thinking generation, scoring and
termination qualification before results. Report effects of this adapter on an
explicitly pinned base; do not claim exact historical training replication. The
first run would benchmark generation throughput and completion on a tiny frozen
capability set before assigning a larger runtime. No such run is yet admitted.

Do not spend GPU time on another entropy-only comparison: the previous forward
diagnostic already showed why target movement is insufficient. A publication
still needs a new useful claim, verified outcomes, controls and independent
confirmation. No behavioral result, training run, or submission qualification
was produced by this release audit. AWS and GH200 were not contacted in it.

Evidence: artifacts/opsd_checkpoint_search_20260910; CPU audit script:
scripts/audit_released_uopsd_adapter.py. The first NumPy tensor-read attempt failed
because BF16 is unsupported there; the corrected CPU PyTorch safe-tensor read
completed. Downloaded bytes were unchanged and matched the published hash.
