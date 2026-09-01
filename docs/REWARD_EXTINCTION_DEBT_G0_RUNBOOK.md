# Reward-Seeking Extinction Debt G0 Runbook

## 1. CPU preparation

Run once, before inspecting or tuning any TEST result:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
python -m reward_extinction_debt_g0.prepare `
  --output-dir artifacts/reward_extinction_debt_prepared
```

Keep `private_answer_key.json` on the analysis machine. Transfer only:

- `sealed_corpus.json`
- `ORACLE_PREFLIGHT.json`

Record all preparation hashes. Never regenerate a corpus after a scientific
result is known.

## 2. Freeze and pin

After the final CPU test commit:

```powershell
git rev-parse HEAD
Get-FileHash configs/reward_extinction_debt_g0.yaml -Algorithm SHA256
```

The package code-tree SHA-256 is the SHA-256 of a canonical JSON mapping from
every `src/reward_extinction_debt_g0/*.py` relative path to its file SHA-256.
The config embeds this tree hash. The launcher separately requires the final
Git commit, config hash, and code-tree hash because a commit cannot safely
embed its own hash.

## 3. GH200 launch

Use the already verified CUDA environment containing exactly Transformers
5.15.0 and PEFT 0.20.0. Qwen3.5 runs text-only with thinking disabled and the
released PyTorch DeltaNet fallback.

```bash
export REWARD_EXTINCTION_DEBT_RUN_ROOT=/home/ubuntu/reward_extinction_debt_g0_YYYYMMDDTHHMMZ
export REWARD_EXTINCTION_DEBT_SEALED_CORPUS=/home/ubuntu/staged_reward_extinction_debt/sealed_corpus.json
export REWARD_EXTINCTION_DEBT_ORACLE_PREFLIGHT=/home/ubuntu/staged_reward_extinction_debt/ORACLE_PREFLIGHT.json
export REWARD_EXTINCTION_DEBT_PINNED_GIT_COMMIT=<final-commit>
export REWARD_EXTINCTION_DEBT_PINNED_CONFIG_SHA256=<config-sha256>
export REWARD_EXTINCTION_DEBT_PINNED_CODE_TREE_SHA256=<code-tree-sha256>
export REWARD_EXTINCTION_DEBT_PYTHON=/home/ubuntu/Align_Paper/.venv/bin/python
bash scripts/run_reward_extinction_debt_g0_remote.sh \
  > /home/ubuntu/reward_extinction_debt_g0_YYYYMMDDTHHMMZ.log 2>&1
```

Use a new absolute root every time. The runner refuses overwrites and dirty
tracked worktrees. It preserves every zero, induction, alignment-candidate,
selected, and reacquisition adapter.

## 4. Monitoring

Routine monitoring should inspect only process state, GPU utilization, log
tail, filesystem growth, and the `COMPLETE` marker. Do not open raw TEST scores
during the run. A final run has completed only when both `MANIFEST.json` and
`COMPLETE` exist and the process has exited successfully.

## 5. Retrieval and verification

Copy the entire evidence directory without modification into a fresh local
directory. Compare the remote and local manifest hashes, then run:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
python -m reward_extinction_debt_g0.verify `
  --config configs/reward_extinction_debt_g0.yaml `
  --answer-key artifacts/reward_extinction_debt_prepared/private_answer_key.json `
  --root retrieved/reward_extinction_debt_g0_YYYYMMDDTHHMMZ `
  --destination analysis/reward_extinction_debt_g0_YYYYMMDDTHHMMZ
```

The verifier refuses mutated evidence, reconstructs probabilities from raw
sequence log-likelihoods, opens TEST once with the committed private key, and
recomputes the decision. Never run the verifier inside the evidence root.

## 6. Sequencing with Phantom Rollback

The experiments are independent. Run Phantom Rollback first if both frozen
inputs are already staged; it is inference-only and establishes whether the
instance/model runtime is healthy. Secure and retrieve its evidence before
starting Extinction Debt. Extinction Debt then owns the GPU exclusively because
it repeatedly loads and trains LoRA checkpoints.

Do not launch an expansion experiment automatically after either gate. Preserve
the model cache, logs, evidence, and every checkpoint until the PI decision is
recorded and retrieval is verified.
