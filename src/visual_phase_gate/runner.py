"""Durable VLM inference for every rendered patch phase."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

from under_extinction.io import canonical_json, sha256_file, write_json

from .corpus import PhaseCase, load_cases


QWEN_VL = "Qwen/Qwen3-VL-8B-Instruct"
GEMMA_VL = "google/gemma-4-12B-it"


def _seed(*parts: str | int) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % (2**63 - 1)


def _load(model_id: str, revision: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    if model_id == QWEN_VL:
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()},
            low_cpu_mem_usage=True,
        )
    elif model_id == GEMMA_VL:
        from transformers import Gemma4UnifiedForConditionalGeneration

        model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()},
            low_cpu_mem_usage=True,
        )
    else:
        raise ValueError("model is not frozen in the visual-phase runner")
    return processor, model.eval()


def collect(cases: Sequence[PhaseCase], *, image_root: str | Path, model_id: str, revision: str, same_image_samples: int = 4, temperature: float = .7, max_new_tokens: int = 12) -> Iterable[dict[str, Any]]:
    if same_image_samples < 4 or not 0 < temperature <= 2:
        raise ValueError("invalid VLM collection parameters")
    import torch

    processor, model = _load(model_id, revision)
    image_root = Path(image_root)
    for case in cases:
        image = image_root / case.image_path
        if not image.is_file():
            raise FileNotFoundError(image)
        repetitions = same_image_samples if case.phase_x == 0 else 1
        for sample_id in range(repetitions):
            messages = [{"role": "user", "content": [{"type": "image", "image": str(image)}, {"type": "text", "text": case.prompt}]}]
            template_kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            if model_id == GEMMA_VL:
                template_kwargs["enable_thinking"] = False
            inputs = processor.apply_chat_template(messages, **template_kwargs).to(model.device)
            input_length = int(inputs["input_ids"].shape[-1])
            sampling = sample_id != 0
            seed = _seed(model_id, revision, case.image_id, sample_id)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": sampling}
                if sampling:
                    kwargs.update(temperature=temperature, top_p=.9)
                with torch.inference_mode():
                    generated = model.generate(**inputs, **kwargs)
            completion = processor.decode(generated[0, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            yield {"image_id": case.image_id, "sample_id": sample_id, "completion": completion}


def run(*, inputs: str | Path, image_root: str | Path, output: str | Path, model_id: str, revision: str, same_image_samples: int = 4, temperature: float = .7, max_new_tokens: int = 12) -> dict[str, Any]:
    root = Path(output)
    if root.exists():
        raise FileExistsError("refusing to overwrite visual-phase evidence")
    cases = load_cases(inputs)
    root.mkdir(parents=True)
    input_copy = root / "frozen_inputs.jsonl"
    input_copy.write_bytes(Path(inputs).read_bytes())
    partial, running_path = root / "raw_completions.partial.jsonl", root / "RUNNING.json"
    running = {"status": "INCOMPLETE_DO_NOT_ANALYZE", "records_completed": 0, "model_id": model_id, "revision": revision}
    write_json(running_path, running)
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for record in collect(cases, image_root=image_root, model_id=model_id, revision=revision, same_image_samples=same_image_samples, temperature=temperature, max_new_tokens=max_new_tokens):
            handle.write(canonical_json(record) + "\n")
            handle.flush()
            running["records_completed"] += 1
            write_json(running_path, running)
    raw = root / "raw_completions.jsonl"
    partial.replace(raw)
    running_path.unlink()
    manifest = {
        "kind": "visual_patch_phase_family",
        "model_id": model_id,
        "revision": revision,
        "case_count": len(cases),
        "same_image_samples": same_image_samples,
        "inputs_sha256": sha256_file(input_copy),
        "raw_sha256": sha256_file(raw),
    }
    write_json(root / "MANIFEST.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--same-image-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args(argv)
    print(canonical_json(run(**vars(args))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
