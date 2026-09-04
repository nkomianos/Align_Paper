# Single-profile released-default reproduction: staged protocol

No neural inference launched by the implementer. Root reviews and freezes source/data before calibration; training requires a separately written approval bound to the completed calibration manifest. This is known-method apparatus validation, not paper evidence.

## Immutable assets and provenance

- Upstream `lasgroup/user_interactions` commit `3b17d2a67bd2565b9fbda495fd16a485406aa954`. Exported exact git bytes for updater, config, user simulator and style judge, with `PINNED_SOURCE.json`.
- Policy Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`; file hashes checked against the preserved v3 model receipt.
- User/judge Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, publicly downloaded and hashed.
- Dataset repository revision `b8f7d168b6f4e95b2a92e84768bd6c955bed2f29`; original23 comparison batch files from the public URLs in that script, all raw bytes retained and hashed. No test split is accepted. Model download/script/source receipts retained.

The HF revision pins the loader script, **not immutable versions of its Azure blobs**. The separately recorded SHA-256 values pin the actual downloaded source bytes used here. Reproducing post grouping requires those raw files plus the staging code, not just the selected split files; both are preserved locally and remotely.

Fresh prepared **v2** root `/home/ubuntu/sdpo_single_profile_assets_v2_20260904`; previous v1 preserved. Original preprocessing's post-or-article fallback, prompt prefix, normalized-prompt deduplication and strict `<1024 CHARACTERS` filter are retained. Additional conservative source-post-ID deduplication drops edited variants across and within splits. This is an explicitly documented additional filter. Eligible3975 training posts and1605 validation posts remain. Deterministic hash ordering selects16 train-source calibration cases, next64 adaptation cases, first32 validation-source evaluation cases, before any outputs. No inferred outcome or model-based selection is used.

## Actual released code, not a reimplemented objective

The wrapper instantiates the upstream `OnlineSDPOUpdater`, `StyleUserSimulator` and `StyleJudge`, and invokes actual `train_step`. Settings are explicitly fixed: full_distillation, student top20 plus tail reverse KL, AdamW5e-6/epsilon1e-6/weightdecay0/gradclip1, LoRA256/512 attention AND MLP, one update per episode, one concise/casual/beginner profile. Retokenization of decoded completion and appended EOS remain upstream behavior, unlike our prior native bridge.

Single-GPU SDPA and no vLLM are the execution departures. Default2048-token policy cap and released128-token feedback cap remain; no shorter outputs are silently substituted. Evaluation is greedy with fixed seeds; training uses the released sampling configuration. Model/context, feedback and judge truncation are checked and cause an explicit stop instead of silent clipping. Native generation IDs/configs and training token log probabilities are retained by observational wrappers; these do not alter the loss. Adapter-only snapshots plus optimizer states every16 updates replace merged full-model checkpoint exports, with no cleanup.

Explicit calibration uses an assistant-facing instruction conveying the desired preference. It does NOT paste the simulator's 'you are a user' identity into the policy system message. This is a calibration oracle, never a base or training-policy prompt.

## Calibration and root review

Generate ordinary, explicit-preference and hindsight responses for all16 frozen natural prompts, plus actual user feedback. Run the released symmetric judge in both answer orders and retain scores for explicit-versus-ordinary, hindsight-versus-ordinary and hindsight-versus-explicit. Pairwise preference alone does not label a response absolutely satisfactory.

An **assistant factual/style audit**, not a human study, reviews all16 tuples under the following rubric before approval: materially unsupported entities/numbers, changed negations, contradictions of the source, or misleading omissions are content failures. Style satisfaction requires a short direct summary, accessible everyday wording rather than unexplained jargon, and a conversational rather than formal register. Exact word counts or punctuation are not hidden criteria. Ambiguous cases remain ambiguous and are not reassigned to produce favorable strata. Preserve individual decisions with reasons.

Preservation and correction recovery are reported separately according to audited original satisfaction. Requiring at least four cases in each stratum is an engineering informativeness check, not a scientific null or model failure; insufficient strata stop as insufficient calibration headroom without replacement prompts. Proposed engineering thresholds (subject to root freeze before outputs): no material content-corruption cases, preservation>=80%, recovery>=75%. These test conditional teacher validity, not whether parameter adaptation will improve. No predictive-training-gain threshold is used for calibration.

Calibration always exits with zero updates and requires root review. No automatic training or follow-on hypothesis. Approval JSON must bind `calibration_manifest_sha256` and state `approved: true`; the training process also checks source/data/upstream/persona equality. All64 updates run only after approval; no checkpoint selection or outcome-driven extension. Held-out raw win/tie/loss, judge order consistency, factual audit and lengths determine the limited replication interpretation, not acceptance prospects.

## Environment and execution

Fresh `/home/ubuntu/sdpo_repro_env_20260904`, inheriting systemtorch2.7.0 without modifying the previous environment. Transformers4.57.6, PEFT0.15.1, Accelerate1.6.0, Datasets3.5.0 installed; Pillow10.4.0 resolves old systemPillow's missing `Image.Resampling`. Full package freeze must be archived before inference. Import/data-only preflight succeeded after that environment fix; first failed preflight preserved. Model loading/forward feasibility remains to be checked by root's approved calibration.

```sh
python sdpo_single_profile_reproduction.py calibration ASSETS_V2 PINNED_UPSTREAM FRESH_CAL_ROOT \
  --policy PINNED_QWEN4B --simulator PINNED_QWEN8B --policy-manifest VERIFIED_V3_MODEL_JSON
```

`training` uses the same command arguments plus `--calibration-root CAL_ROOT --approval ROOT_APPROVAL_JSON` and a different fresh output root. Two-hour phase budget with ten-minute reserve is checked at saved boundaries; root owns external cutoff15:40UTC and stuck-call handling. Runtime must be estimated from actual calibration throughput, not from the old format experiment. No model generation has occurred during staging.
