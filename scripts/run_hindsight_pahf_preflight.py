"""Run the frozen Qwen3.5-9B EndoPAHF capability preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from interaction_sprint.hindsight_neural_anchor import MODEL_ID, MODEL_REVISION
from interaction_sprint.hindsight_pahf_interface import OPTION_LETTERS
from interaction_sprint.hindsight_pahf_preflight import (
    BASE_COUNT,
    CONTEXTS,
    SELECTION_SALT,
    build_jobs,
    select_base_panel,
    summarize_preflight,
)


INPUT_MANIFEST_SHA256 = "915dbc068573c50990c42db8aa48a1ba5158f12c4684cea4d0d10727d151e5c2"
BATCH_SIZE = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.input_root / "MANIFEST.json") != INPUT_MANIFEST_SHA256:
        raise SystemExit("EndoPAHF v2 input manifest mismatch")
    args.root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    repository = Path(__file__).parents[1]

    def write(name: str, value: object) -> None:
        (args.root / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )

    development = json.loads((args.input_root / "development.json").read_text(encoding="utf-8"))
    selected = select_base_panel(development)
    jobs = build_jobs(selected)
    spec = {
        "scope": "EXACT_SDPO_INTERFACE_PREFLIGHT_NOT_PAPER_EVIDENCE",
        "preflight_version": 2,
        "interface": "plain_prompt_plus_frozen_hindsight_block",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "base_count": BASE_COUNT,
        "variants_per_base": 4,
        "contexts": list(CONTEXTS),
        "selection_salt": SELECTION_SALT,
        "batch_size": BATCH_SIZE,
        "confirmation_opened": False,
        "paper_green_light": False,
    }
    write("spec.json", spec)
    write("selected_cases.json", selected)
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_interface.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_preflight.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_pahf_preflight.py",
    ]
    write("source_hashes.json", {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    })

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
    sequences = [tokenizer.encode(letter, add_special_tokens=False) for letter in OPTION_LETTERS]
    if any(len(sequence) != 1 for sequence in sequences):
        raise RuntimeError(f"A/B/C/D are not all single tokens: {sequences}")
    choice_ids = torch.tensor([sequence[0] for sequence in sequences], device=model.device)
    scores: list[dict[str, object]] = []
    for start in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[start:start + BATCH_SIZE]
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": str(job["text"])}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for job in batch
        ]
        lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in rendered]
        if max(lengths) > 1024:
            raise RuntimeError("preflight prompt exceeds 1024-token ceiling")
        encoded = tokenizer(
            rendered, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(model.device)
        with torch.no_grad():
            log_probs = model(
                **encoded, use_cache=True, logits_to_keep=1,
            ).logits[:, -1, :].float().log_softmax(-1)
        for offset, job in enumerate(batch):
            choice_log = log_probs[offset, choice_ids]
            normalized = choice_log.softmax(0)
            target_index = OPTION_LETTERS.index(str(job["target"]))
            scores.append({
                "id": job["id"],
                "base_id": job["base_id"],
                "rotation": job["rotation"],
                "context": job["context"],
                "target": job["target"],
                "choice_token_ids": choice_ids.tolist(),
                "normalized_choice_probabilities": normalized.tolist(),
                "normalized_target_probability": float(normalized[target_index]),
                "normalized_correct": int(normalized.argmax()) == target_index,
                "full_vocabulary_choice_mass": float(choice_log.logsumexp(0).exp()),
            })
    write("scores.json", scores)
    result = summarize_preflight(scores)
    result["wall_seconds"] = time.time() - started
    result["choice_token_ids"] = choice_ids.tolist()
    write("RESULT.json", result)
    files = [path for path in args.root.iterdir() if path.is_file()]
    write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
