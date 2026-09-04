# Single-profile calibration: as-executed sampling, not intended greedy decoding

The 16-case calibration completed with zero updates and intact evidence. The receipt audit passes, but the intended greedy-policy protocol did not execute. No training is approved by either audit.

## Cause and scope

Transformers 4.57.6 copies the supplied generation configuration and replaces values equal to global defaults with model-specific defaults, when the model configuration was saved with Transformers >=4.50 and `use_model_defaults` is unspecified. The frozen wrapper supplied `do_sample=False` inside a GenerationConfig, not as a direct `generate` keyword. False is the global default, so Qwen's saved `do_sample=True` wins. The original passed object remains False and was therefore logged faithfully, but it is not the effective configuration. This behavior is explicit in the [pinned official generation implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/generation/utils.py).

Policy effective parameters: sampling=True, temperature=0.6, top_k=20, top_p=0.95, BOS=151643, EOS=151645, maximum new tokens=2048. Simulator: sampling=True, temperature=0.7, top_k=20, top_p=0.95, same BOS/EOS, maximum new tokens=128. The simulator's nondefault temperature survives the merge. No live runner, checkpoint, dependency, or evidence was changed.

The deviation was identified from configuration warnings before reviewing response outcomes. The parent decided to finish the bounded calibration, preserve it as sampled, and withhold training approval. This is not evidence of failure of the scientific hypothesis or of SDPO itself.

## Audit trail

- Evidence: `retrieved/sdpo_single_profile_20260904T1045Z/sdpo_single_profile_calibration_retry_20260904T1040Z`.
- Final manifest SHA-256: `03b08fc8548cdf0e11bbb8a48b9f2f91e1ce824f00c71d1782db2dc76f969386`.
- `passed_config_receipt_verification.json` beside the root verifies pinned source/data selection/model metadata, token/text contexts, native termination, and AB/BA arithmetic. It checks the passed configuration only.
- `effective_config_forensic.json` beside the root reports `VERIFIED_AS_EXECUTED_SAMPLED_CALIBRATION_PROTOCOL_DEVIATION`, covering all 16 cases and 64 generation calls.
- `artifacts/sdpo_generation_config_forensic_20260904/receipt.json` is a separate CPU-only probe in the actual Transformers 4.57.6 environment. It archives the exact installed method source, module hashes, both model generation-config receipts, passed and effective configs, and complete case/condition coverage. The text-source companion is for inspection; the original method string inside the JSON has its exact SHA bound.
- Installed merge-method SHA: `fbd277c3bc70bcc944dee222b653e0702234bdeebb8826d19e9613997db9f6ef`.
- Both model generation-config file hashes: `2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2`, matched to original experiment model receipts.

The independent auditor replays the visited supplied-config branch and agrees exactly with the installed-method probe. Four prospective regression tests cover False being overridden, simulator temperature preservation, direct-keyword precedence, and disabling model defaults. These tests do not modify the completed experiment.

This is an external forensic attestation, not a claim that effective configuration was captured at generation time. Actual generate keyword assumptions come from pinned call sites. Neither audit independently replays RNG, neural generation, weights, or judge probabilities. Outcome/style review remains separate and must label this as an as-executed sampled calibration. A future intended-greedy run would require a new frozen protocol and verified effective configuration; it must not overwrite or relabel this run.
