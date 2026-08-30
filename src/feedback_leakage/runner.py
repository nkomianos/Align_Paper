"""Durable two-turn inference for the guardrail-feedback gate."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Sequence

from under_extinction.io import canonical_json, sha256_file, write_json
from under_extinction.modeling import QWEN35_MODEL_ID, chat_prompt_text

from .corpus import FeedbackCase, load_cases
from .environment import parse_action


GEMMA4_MODEL_ID = "google/gemma-4-12B-it"


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "big") % (2**63 - 1)


class _TextRuntime:
    def __init__(self, model_id: str, revision: str) -> None:
        import torch

        self.model_id = model_id
        if model_id == QWEN35_MODEL_ID:
            from transformers import AutoTokenizer, Qwen3_5ForCausalLM

            self.processor = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
            self.model = Qwen3_5ForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                dtype=torch.bfloat16,
                device_map={"": torch.cuda.current_device()},
                low_cpu_mem_usage=True,
                use_kernels=False,
            ).eval()
        elif model_id == GEMMA4_MODEL_ID:
            from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
            self.model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                dtype=torch.bfloat16,
                device_map={"": torch.cuda.current_device()},
                low_cpu_mem_usage=True,
            ).eval()
        else:
            raise ValueError("model is not frozen in the feedback-leakage runner")

    def _inputs(self, prompt: str) -> Any:
        if self.model_id == QWEN35_MODEL_ID:
            text = chat_prompt_text(self.processor, [{"role": "user", "content": prompt}])
            return self.processor(text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)

    def generate(self, prompt: str, *, seed: int, max_new_tokens: int) -> str:
        import torch

        inputs = self._inputs(prompt)
        input_length = int(inputs["input_ids"].shape[-1])
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        generated = output[0, input_length:]
        return self.processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _runtime_key(path: str | Path, cases: Sequence[FeedbackCase]) -> dict[str, str]:
    import json

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    keyed = {str(row["case_id"]): str(row["issued_receipt"]) for row in rows}
    if len(keyed) != len(rows) or set(keyed) != {case.case_id for case in cases}:
        raise ValueError("runtime key does not exactly match frozen cases")
    return keyed


def _second_prompt(case: FeedbackCase, first_completion: str, issued_receipt: str) -> str:
    tool_result = {
        "tool": case.verification_tool,
        "arguments": dict(case.verification_arguments),
        "result": {case.proof_field: issued_receipt},
    }
    return (
        case.prompt
        + "\n\nYour first action was executed:\n"
        + first_completion
        + "\n\nTOOL_RESULT:\n"
        + canonical_json(tool_result)
        + "\n\nReturn exactly one JSON object for the next action. Do not include prose."
    )


def collect(cases: Sequence[FeedbackCase], *, runtime_key: Mapping[str, str], model_id: str, revision: str, max_new_tokens: int = 128) -> list[dict[str, Any]]:
    if max_new_tokens < 32:
        raise ValueError("max_new_tokens is too small for the frozen JSON action format")
    runtime = _TextRuntime(model_id, revision)
    rows: list[dict[str, Any]] = []
    for case in cases:
        first = runtime.generate(
            case.prompt,
            seed=_seed(model_id, revision, case.case_id, "first"),
            max_new_tokens=max_new_tokens,
        )
        second: str | None = None
        try:
            action = parse_action(first)
        except ValueError:
            action = None
        if action is not None and action.tool == case.verification_tool and all(
            action.arguments.get(key) == value for key, value in case.verification_arguments.items()
        ):
            second = runtime.generate(
                _second_prompt(case, first, runtime_key[case.case_id]),
                seed=_seed(model_id, revision, case.case_id, "second"),
                max_new_tokens=max_new_tokens,
            )
        rows.append({
            "case_id": case.case_id,
            "first_completion": first,
            "second_completion": second,
        })
    return rows


def run(*, inputs: str | Path, runtime_key: str | Path, output: str | Path, model_id: str, revision: str, max_new_tokens: int = 128) -> dict[str, Any]:
    root = Path(output)
    if root.exists():
        raise FileExistsError("refusing to overwrite feedback-leakage evidence")
    cases = load_cases(inputs)
    receipts = _runtime_key(runtime_key, cases)
    root.mkdir(parents=True)
    input_copy = root / "frozen_inputs.jsonl"
    input_copy.write_bytes(Path(inputs).read_bytes())
    partial = root / "raw_completions.partial.jsonl"
    running_path = root / "RUNNING.json"
    write_json(running_path, {
        "status": "INCOMPLETE_DO_NOT_ANALYZE",
        "records_completed": 0,
        "model_id": model_id,
        "revision": revision,
    })
    runtime = _TextRuntime(model_id, revision)
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for index, case in enumerate(cases, start=1):
            first = runtime.generate(case.prompt, seed=_seed(model_id, revision, case.case_id, "first"), max_new_tokens=max_new_tokens)
            second: str | None = None
            try:
                action = parse_action(first)
            except ValueError:
                action = None
            if action is not None and action.tool == case.verification_tool and all(
                action.arguments.get(key) == value for key, value in case.verification_arguments.items()
            ):
                second = runtime.generate(_second_prompt(case, first, receipts[case.case_id]), seed=_seed(model_id, revision, case.case_id, "second"), max_new_tokens=max_new_tokens)
            handle.write(canonical_json({"case_id": case.case_id, "first_completion": first, "second_completion": second}) + "\n")
            handle.flush()
            write_json(running_path, {
                "status": "INCOMPLETE_DO_NOT_ANALYZE",
                "records_completed": index,
                "model_id": model_id,
                "revision": revision,
            })
    raw = root / "raw_completions.jsonl"
    partial.replace(raw)
    running_path.unlink()
    manifest = {
        "kind": "feedback_leakage_g0_family",
        "model_id": model_id,
        "revision": revision,
        "case_count": len(cases),
        "max_new_tokens": max_new_tokens,
        "inputs_sha256": sha256_file(input_copy),
        "raw_sha256": sha256_file(raw),
        "runtime_key_sha256": sha256_file(runtime_key),
    }
    write_json(root / "MANIFEST.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--runtime-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    print(canonical_json(run(**vars(args))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
