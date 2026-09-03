"""One explicit smoke or pilot run. No automatic chaining or model training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

from .pilot import (FRAMES, MODEL_ID, MODEL_REVISION, SETTINGS, VERSION,
                    dump, prepare, seal, validate_seal)


def planned_cases(cases, mode):
    if mode == "full":
        return cases
    if mode != "smoke":
        raise ValueError("unknown run mode")
    # Frozen before inference; covers several directions and the static case.
    return [c for c in cases if c["scene"] in ("scene-000", "scene-007", "scene-013", "scene-024")
            and c["condition"] in ("native_rgb", "joint", "oracle_joint")]


def token_contract(inputs, image_token_id: int) -> dict:
    grids = inputs.get("image_grid_thw")
    if grids is None or len(grids) != FRAMES:
        raise ValueError("must process exactly 16 image canvases")
    grids = grids.tolist()
    if len({tuple(g) for g in grids}) != 1:
        raise ValueError("unequal per-image resolution")
    n = int((inputs["input_ids"] == image_token_id).sum().item())
    if n <= 0:
        raise ValueError("could not count image tokens")
    return {"vision_tokens": n, "image_grid_thw": grids,
            "total_input_tokens": int(inputs["input_ids"].shape[-1])}


def run(prepared: Path, output: Path, mode="smoke"):
    validate_seal(prepared)
    if json.loads((prepared / "settings.json").read_text()) != json.loads(json.dumps(SETTINGS)):
        raise ValueError("settings differ from source")
    if not json.loads((prepared / "preflight.json").read_text())["estimator_numeric_check"]:
        raise ValueError("CPU apparatus validation failed: no GPU inference authorized")
    cases = [json.loads(line) for line in (prepared / "cases.jsonl").read_text().splitlines()]
    plan = planned_cases(cases, mode)
    if len(plan) != (450 if mode == "full" else 24):
        raise ValueError("unexpected frozen workload")
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    shutil.copytree(source, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(prepared, output / "inputs")
    # A copied nested manifest is hashed by the outer manifest too.
    manifest_path = output / "inputs" / "MANIFEST.json"
    manifest_path.rename(output / "inputs" / "INPUT_MANIFEST.json")
    dump(output / "plan.json", {"mode": mode, "case_ids": [c["case_id"] for c in plan],
                                "model_id": MODEL_ID, "revision": MODEL_REVISION})
    try:
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from huggingface_hub import HfApi

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; CPU tests use no model")
        resolved = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION).sha
        if resolved != MODEL_REVISION:
            raise ValueError("immutable model resolution mismatch")
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()}, low_cpu_mem_usage=True).eval()
        dump(output / "runtime.json", {
            "version": VERSION, "model_id": MODEL_ID, "resolved_revision": resolved,
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "mode": mode,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        })
        expected = None
        # Before ANY forward pass, check every planned prompt's processed budget.
        # Reprocessing for inference is deliberate: no hundreds-of-GB tensor cache.
        contracts = {}
        def encode(case):
            content = [{"type": "image", "image": str(output / "inputs" / p)} for p in case["frame_paths"]]
            content.append({"type": "text", "text": case["prompt"]})
            return processor.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")
        for case in plan:
            inputs = encode(case)
            contract = token_contract(inputs, model.config.image_token_id)
            shape = (contract["vision_tokens"], contract["image_grid_thw"])
            if expected is None:
                expected = shape
            if shape != expected:
                raise ValueError("visual-token mismatch before scoring")
            contracts[case["case_id"]] = contract
        dump(output / "budget_audit.json", contracts)
        with (output / "raw.jsonl").open("x", encoding="utf-8", newline="\n") as f:
            for case in plan:
                inputs = encode(case).to(model.device)
                start = time.monotonic()
                n = int(inputs["input_ids"].shape[-1])
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=8)
                torch.cuda.synchronize()
                completion = processor.decode(generated[0, n:], skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
                row = {"case_id": case["case_id"], "completion": completion,
                       "seconds": time.monotonic() - start, **contracts[case["case_id"]]}
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
        dump(output / "COMPLETE.json", {"records": len(plan), "mode": mode})
    except BaseException as exc:
        # Preserve partial raw records and runtime errors, never silently retry.
        dump(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        seal(output)
        raise
    return seal(output)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("run")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = prepare(**args) if command == "prepare" else run(**args)
    print(json.dumps({"files_sealed": len(result["files"]), "output": str(args["output"])}))


if __name__ == "__main__":
    main()
