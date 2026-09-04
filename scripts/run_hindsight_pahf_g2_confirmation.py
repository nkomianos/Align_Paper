"""Evaluate frozen G2 adapters on locked EndoPAHF confirmation inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

from interaction_sprint.hindsight_neural_anchor import (
    LORA_ALPHA, LORA_RANK, MODEL_ID, MODEL_REVISION, install_qwen35_lora,
)
from interaction_sprint.hindsight_pahf_g2_v2 import (
    G2_BATCH, expected_g2_arm_names, summarize_g2_stage,
)
from interaction_sprint.hindsight_pahf_interface import OPTION_LETTERS
from latent_contract.sender_update import load_adapter
from run_hindsight_pahf_g2_dev import INPUT_MANIFEST_SHA256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state(path: Path):
    import torch
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"invalid adapter checkpoint: {path.name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--dev-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.input_root / "MANIFEST.json") != INPUT_MANIFEST_SHA256:
        raise SystemExit("EndoPAHF v3 input manifest mismatch")
    args.root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).parents[1]
    started = time.time()

    def write(name: str, value: object) -> None:
        (args.root / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )

    def write_manifest() -> None:
        files = [path for path in args.root.iterdir() if path.is_file() and path.name != "MANIFEST.json"]
        write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})

    verification = subprocess.run(
        [sys.executable, str(repository / "scripts" / "verify_hindsight_pahf_g2_dev.py"),
         "--input-root", str(args.input_root), "--g1-root", str(args.g1_root),
         "--preflight-root", str(args.preflight_root), "--root", str(args.dev_root)],
        check=False, capture_output=True, text=True,
    )
    if verification.returncode != 0:
        write("FAILED.json", {
            "stage": "development_prerequisite_verification",
            "returncode": verification.returncode,
            "stdout": verification.stdout,
            "stderr": verification.stderr,
        })
        write_manifest()
        raise RuntimeError("G2 development prerequisite failed verification")
    dev_receipt = json.loads(verification.stdout)
    if (
        dev_receipt.get("verified") is not True
        or dev_receipt.get("decision") != "ENDO_PAHF_G2_V2_DEV_QUALIFIED"
    ):
        write("FAILED.json", {
            "stage": "development_prerequisite_decision",
            "receipt": dev_receipt,
        })
        write_manifest()
        raise RuntimeError("confirmation requires qualified G2 development")
    prerequisite = {
        "root_at_launch": str(args.dev_root.resolve()),
        "manifest_sha256": sha256(args.dev_root / "MANIFEST.json"),
        "result_sha256": sha256(args.dev_root / "RESULT.json"),
        "receipt": dev_receipt,
    }
    write("development_prerequisite.json", prerequisite)

    confirmation_path = args.input_root / "confirmation.json"
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    write("input_receipt.json", {
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "confirmation_sha256": sha256(confirmation_path),
        "confirmation_rows": len(confirmation),
        "confirmation_opened": True,
        "opened_only_after_qualified_dev": True,
    })
    expected_arms = expected_g2_arm_names()
    spec = {
        "version": "endo-pahf-g2-confirmation-v2-full-learning",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "arms": expected_arms,
        "batch": G2_BATCH,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "adapter_manifest_sha256": prerequisite["manifest_sha256"],
        "panel_aggregation": "arithmetic mean of four panel probability vectors",
        "confirmation_opened": True,
        "training_or_selection_on_confirmation": False,
        "paper_green_light": False,
    }
    write("spec.json", spec)
    source_paths = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_cluster_stats.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_g2.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_g2_v2.py",
        repository / "src" / "latent_contract" / "sender_update.py",
        repository / "scripts" / "run_hindsight_pahf_g2_dev.py",
        repository / "scripts" / "verify_hindsight_pahf_g2_dev.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_pahf_g2_confirmation.py",
        repository / "scripts" / "run_hindsight_pahf_g2_confirmation_remote.sh",
    ]
    write("source_hashes.json", {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in source_paths
    })

    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU is required")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=args.hf_home, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=args.hf_home,
        dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": torch.cuda.current_device()}, low_cpu_mem_usage=True,
        use_kernels=False,
    ).eval()
    token_sequences = [tokenizer.encode(letter, add_special_tokens=False) for letter in OPTION_LETTERS]
    if any(len(sequence) != 1 for sequence in token_sequences):
        raise RuntimeError("A/B/C/D are not all single tokens")
    answer_ids = torch.tensor([sequence[0] for sequence in token_sequences], device=model.device)
    install_qwen35_lora(model)

    def render(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    plain = {str(row["id"]): render(str(row["prompt"])) for row in confirmation}

    def evaluate(name: str, state_path: Path) -> list[dict[str, object]]:
        load_adapter(model, _load_state(state_path))
        predictions: list[dict[str, object]] = []
        with torch.no_grad():
            for start in range(0, len(confirmation), G2_BATCH):
                batch = confirmation[start:start + G2_BATCH]
                texts = [plain[str(row["id"])] for row in batch]
                lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
                if max(lengths) > 1024:
                    raise RuntimeError("confirmation prompt exceeds 1024-token ceiling")
                encoded = tokenizer(
                    texts, return_tensors="pt", padding=True, add_special_tokens=False,
                ).to(model.device)
                output = model(**encoded, use_cache=True, logits_to_keep=1).logits[:, -1, :]
                full_log = output.float().log_softmax(-1)
                choice_log = full_log[:, answer_ids]
                normalized_log = choice_log.log_softmax(-1)
                mass = choice_log.logsumexp(-1).exp()
                for index, row in enumerate(batch):
                    predictions.append({
                        "id": row["id"],
                        "normalized_choice_log_probabilities": normalized_log[index].tolist(),
                        "full_vocabulary_choice_mass": float(mass[index]),
                    })
        write(f"{name}_confirmation_predictions.json", predictions)
        print(json.dumps({"arm": name, "wall_seconds": time.time() - started}), flush=True)
        return predictions

    try:
        arm_predictions = {}
        for name in expected_arms:
            checkpoint = (
                args.dev_root / "initial_adapter.pt" if name == "baseline"
                else args.dev_root / f"{name}_adapter.pt"
            )
            arm_predictions[name] = evaluate(name, checkpoint)
        raw_state = _load_state(args.dev_root / "raw_immediate_adapter.pt")
        transition_state = _load_state(args.dev_root / "transition_sanity_adapter.pt")
        if raw_state.keys() != transition_state.keys():
            raise RuntimeError("transition adapter key mismatch")
        adapter_difference = max(
            float((raw_state[key] - transition_state[key]).abs().max()) for key in raw_state
        )
        summary = summarize_g2_stage(
            confirmation, arm_predictions, stage="confirmation",
            transition_adapter_max_abs_difference=adapter_difference,
        )
        write("RESULT.json", {
            **summary,
            "development_prerequisite": prerequisite,
            "runtime": {
                "wall_seconds": time.time() - started,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
        })
        write_manifest()
        print(json.dumps({
            "decision": summary["decision"],
            "controls": summary["controls"],
            "wall_seconds": time.time() - started,
        }, indent=2), flush=True)
    except Exception as exc:
        write("FAILED.json", {
            "stage": "confirmation_evaluation",
            "error": type(exc).__name__,
            "message": str(exc),
            "wall_seconds": time.time() - started,
        })
        write_manifest()
        raise


if __name__ == "__main__":
    main()
