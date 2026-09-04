"""Run the frozen EndoPAHF A/B/C/D interface rehearsal on cached Qwen3-0.6B."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interaction_sprint.hindsight_pahf_interface import (
    OPTION_LETTERS,
    REHEARSAL_COUNT,
    REHEARSAL_SALT,
    context_specs,
    select_rehearsal_rows,
    summarize_scores,
)


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
BATCH_SIZE = 4
THREADS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    repository = Path(__file__).parents[1]

    def write(name: str, value: object) -> None:
        (args.output_root / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )

    development = json.loads((args.input_root / "development.json").read_text(encoding="utf-8"))
    selected = select_rehearsal_rows(development)
    input_manifest = sha256(args.input_root / "MANIFEST.json")
    spec = {
        "scope": "CPU_REHEARSAL_ONLY_NOT_PAPER_EVIDENCE",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "batch_size": BATCH_SIZE,
        "threads": THREADS,
        "selection_count": REHEARSAL_COUNT,
        "selection_salt": REHEARSAL_SALT,
        "input_manifest_sha256": input_manifest,
        "contexts": ["immediate", "delayed_expression", "delayed_transition"],
        "paper_green_light": False,
    }
    write("spec.json", spec)
    write("selected_cases.json", selected)
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_interface.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_pahf_interface_cpu.py",
    ]
    write("source_hashes.json", {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    })

    torch.set_num_threads(THREADS)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=args.hf_home,
        local_files_only=True, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=args.hf_home,
        local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).eval()
    token_sequences = [tokenizer.encode(letter, add_special_tokens=False) for letter in OPTION_LETTERS]
    if any(len(sequence) != 1 for sequence in token_sequences):
        raise RuntimeError(f"choice labels are not single tokens: {token_sequences}")
    choice_token_ids = torch.tensor([sequence[0] for sequence in token_sequences])

    jobs = [
        {"id": str(row["id"]), **context}
        for row in selected for context in context_specs(row)
    ]
    score_rows: list[dict[str, object]] = []
    for start in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[start:start + BATCH_SIZE]
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": str(job["text"])}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for job in batch
        ]
        encoded = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.no_grad():
            log_probs = model(**encoded, use_cache=True).logits[:, -1, :].float().log_softmax(-1)
        for offset, job in enumerate(batch):
            choice_log_probs = log_probs[offset, choice_token_ids]
            normalized = choice_log_probs.softmax(0)
            target_index = OPTION_LETTERS.index(str(job["target"]))
            score_rows.append({
                "id": job["id"],
                "context": job["context"],
                "target": job["target"],
                "choice_token_ids": choice_token_ids.tolist(),
                "normalized_choice_probabilities": normalized.tolist(),
                "normalized_target_probability": float(normalized[target_index]),
                "normalized_correct": int(normalized.argmax()) == target_index,
                "full_vocabulary_choice_mass": float(choice_log_probs.logsumexp(0).exp()),
                "target_full_vocabulary_probability": float(choice_log_probs[target_index].exp()),
            })
    write("scores.json", score_rows)
    result = summarize_scores(score_rows)
    result["wall_seconds"] = time.time() - started
    result["choice_token_ids"] = choice_token_ids.tolist()
    write("RESULT.json", result)
    files = [path for path in args.output_root.iterdir() if path.is_file()]
    write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
