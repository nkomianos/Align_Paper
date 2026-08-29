"""Durable text-model collection for the effect-consistency gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from under_extinction.io import canonical_json, sha256_file, write_json
from under_extinction.modeling import QWEN35_MODEL_ID, chat_prompt_text

from .corpus import EffectCase, load_cases


def _seed(*parts: str | int) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % (2**63 - 1)


def _prompt(tokenizer: Any, model_id: str, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if model_id == QWEN35_MODEL_ID:
        return chat_prompt_text(tokenizer, messages)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("chat template produced an empty prompt")
    return rendered


def _load_model(model_id: str, revision: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_id == QWEN35_MODEL_ID:
        try:
            from transformers import Qwen3_5ForCausalLM
        except ImportError as exc:  # pragma: no cover - host dependency
            raise RuntimeError("the pinned Transformers runtime lacks Qwen3_5ForCausalLM") from exc
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()}, low_cpu_mem_usage=True, use_kernels=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()}, low_cpu_mem_usage=True,
        )
    return tokenizer, model.eval()


def collect(cases: Sequence[EffectCase], *, model_id: str, revision: str, samples: int = 6, temperature: float = .8, max_new_tokens: int = 160) -> Iterable[dict[str, Any]]:
    if samples < 4 or not 0 < temperature <= 2 or max_new_tokens < 16:
        raise ValueError("invalid collection parameters")
    import torch

    tokenizer, model = _load_model(model_id, revision)
    for case in cases:
        text = _prompt(tokenizer, model_id, case.prompt)
        encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(model.device)
        input_length = int(encoded["input_ids"].shape[1])
        for sample_id in range(samples):
            sampling = sample_id != 0
            seed = _seed(model_id, revision, case.task_id, sample_id)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "do_sample": sampling,
                }
                if sampling:
                    kwargs.update(temperature=temperature, top_p=.95)
                with torch.inference_mode():
                    output = model.generate(**encoded, **kwargs)
            generated = output.sequences[0, input_length:]
            probabilities: list[float] = []
            for token, logits in zip(generated, output.scores, strict=False):
                probabilities.append(float(torch.softmax(logits[0].float(), dim=-1)[int(token)].item()))
            yield {
                "task_id": case.task_id,
                "sample_id": sample_id,
                "completion": tokenizer.decode(generated, skip_special_tokens=True),
                "token_confidence": float(sum(probabilities) / len(probabilities)) if probabilities else 0.0,
            }


def run(*, inputs: str | Path, output: str | Path, model_id: str, revision: str, samples: int = 6, temperature: float = .8, max_new_tokens: int = 160) -> dict[str, Any]:
    root = Path(output)
    if root.exists():
        raise FileExistsError("refusing to overwrite an effect-consistency evidence root")
    cases = load_cases(inputs)
    root.mkdir(parents=True)
    input_copy = root / "frozen_inputs.jsonl"
    input_copy.write_bytes(Path(inputs).read_bytes())
    partial, running_path = root / "raw_completions.partial.jsonl", root / "RUNNING.json"
    running = {"status": "INCOMPLETE_DO_NOT_ANALYZE", "records_completed": 0, "model_id": model_id, "revision": revision}
    write_json(running_path, running)
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for record in collect(cases, model_id=model_id, revision=revision, samples=samples, temperature=temperature, max_new_tokens=max_new_tokens):
            handle.write(canonical_json(record) + "\n")
            handle.flush()
            running["records_completed"] += 1
            write_json(running_path, running)
    raw = root / "raw_completions.jsonl"
    partial.replace(raw)
    running_path.unlink()
    manifest = {
        "kind": "effect_consistency_uq_family",
        "model_id": model_id,
        "revision": revision,
        "case_count": len(cases),
        "samples": samples,
        "inputs_sha256": sha256_file(input_copy),
        "raw_sha256": sha256_file(raw),
    }
    write_json(root / "MANIFEST.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args(argv)
    print(canonical_json(run(**vars(args))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
