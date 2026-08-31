"""Deterministic native-video inference with a complete bound frame snapshot."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Iterable, Sequence

from under_extinction.io import canonical_json, sha256_file, write_json

from .corpus import HindsightCase, corpus_tree_sha256, load_cases, validate_corpus


QWEN3_VL = "Qwen/Qwen3-VL-8B-Instruct"
GEMMA4 = "google/gemma-4-12B-it"
FROZEN_MODELS = {
    QWEN3_VL: "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    GEMMA4: "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
}
PRESENTATION_MODES = ("native_video", "multi_image")
FROZEN_VIDEO_FPS = 2.0


def _seed(*parts: str | int) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % (2**63 - 1)


def _load(model_id: str, revision: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor

    if FROZEN_MODELS.get(model_id) != revision:
        raise ValueError("model revision is not frozen in visual-hindsight G0 v2")
    if not torch.cuda.is_available():
        raise RuntimeError("visual-hindsight VLM inference requires CUDA")
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    if model_id == QWEN3_VL:
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()},
            low_cpu_mem_usage=True,
        )
    elif model_id == GEMMA4:
        from transformers import Gemma4UnifiedForConditionalGeneration

        model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()},
            low_cpu_mem_usage=True,
        )
    else:  # pragma: no cover
        raise ValueError("unknown frozen model")
    return processor, model.eval()


def _content_for_case(
    case: HindsightCase,
    frames: Sequence[Path],
    *,
    presentation_mode: str,
) -> list[dict[str, Any]]:
    if presentation_mode == "native_video":
        if not frames:
            raise ValueError("native video cannot be empty")
        content: list[dict[str, Any]] = [{"type": "video", "video": [str(path) for path in frames]}]
    elif presentation_mode == "multi_image":
        content = [{"type": "image", "image": str(path)} for path in frames]
    else:
        raise ValueError("unknown visual-hindsight presentation mode")
    content.append({"type": "text", "text": case.prompt})
    return content


def _runtime_provenance(
    *,
    model_id: str,
    revision: str,
    presentation_mode: str,
    processor: Any,
    model: Any,
) -> dict[str, Any]:
    import PIL
    import torch
    import transformers

    cuda_name = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "NO_CUDA"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "pillow": PIL.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "cuda_device": cuda_name,
        "processor_class": type(processor).__name__,
        "model_class": type(model).__name__,
        "model_id": model_id,
        "requested_revision": revision,
        "presentation_mode": presentation_mode,
        "native_video_fps": str(FROZEN_VIDEO_FPS),
    }


def collect(
    cases: Sequence[HindsightCase],
    *,
    frame_root: str | Path,
    model_id: str,
    revision: str,
    presentation_mode: str,
    processor: Any,
    model: Any,
    max_new_tokens: int = 8,
) -> Iterable[dict[str, Any]]:
    if FROZEN_MODELS.get(model_id) != revision or max_new_tokens < 2 or max_new_tokens > 16:
        raise ValueError("invalid frozen visual-hindsight collection parameters")
    if presentation_mode not in PRESENTATION_MODES:
        raise ValueError("invalid presentation mode")
    if presentation_mode == "native_video" and model_id != QWEN3_VL:
        raise ValueError("native-video G0 is frozen to Qwen3-VL; Gemma is multi-image comparison only")
    import torch

    frame_root = Path(frame_root)
    for case in cases:
        frames = [frame_root / relative for relative in case.frame_paths]
        if any(not frame.is_file() for frame in frames):
            raise FileNotFoundError(f"missing frame for {case.case_id}")
        messages = [
            {
                "role": "user",
                "content": _content_for_case(case, frames, presentation_mode=presentation_mode),
            }
        ]
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if presentation_mode == "native_video":
            from PIL import Image
            from transformers.video_utils import VideoMetadata

            with Image.open(frames[0]) as first_frame:
                width, height = first_frame.size
            template_kwargs["processor_kwargs"] = {
                "do_sample_frames": False,
                "video_metadata": VideoMetadata(
                    total_num_frames=len(frames),
                    fps=FROZEN_VIDEO_FPS,
                    width=width,
                    height=height,
                    duration=len(frames) / FROZEN_VIDEO_FPS,
                    frames_indices=list(range(len(frames))),
                ),
            }
        if model_id == GEMMA4:
            template_kwargs["enable_thinking"] = False
        inputs = processor.apply_chat_template(messages, **template_kwargs).to(model.device)
        input_length = int(inputs["input_ids"].shape[-1])
        seed = _seed(model_id, revision, presentation_mode, case.case_id)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        if model_id == GEMMA4:
            raw_completion = processor.decode(
                generated[0, input_length:],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            parsed = processor.parse_response(raw_completion, prefix=inputs["input_ids"])
            if not isinstance(parsed, dict) or not isinstance(parsed.get("content"), str):
                raise RuntimeError("Gemma 4 response parser did not return textual content")
            completion = parsed["content"]
        else:
            completion = processor.decode(
                generated[0, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        yield {
            "case_id": case.case_id,
            "pair_id": case.pair_id,
            "arm": case.arm,
            "completion": completion,
        }


def run(
    *,
    inputs: str | Path,
    frame_root: str | Path,
    output: str | Path,
    model_id: str,
    revision: str,
    presentation_mode: str,
    config_sha256: str,
    code_sha256: str,
    git_commit: str,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    root = Path(output)
    if root.exists():
        raise FileExistsError("refusing to overwrite visual-hindsight evidence")
    if FROZEN_MODELS.get(model_id) != revision:
        raise ValueError("model revision is not frozen in visual-hindsight G0 v2")
    if presentation_mode not in PRESENTATION_MODES or (
        presentation_mode == "native_video" and model_id != QWEN3_VL
    ):
        raise ValueError("invalid model/presentation pairing")
    for name, value, length in (
        ("config_sha256", config_sha256, 64),
        ("code_sha256", code_sha256, 64),
        ("git_commit", git_commit, 40),
    ):
        if len(value) != length or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"invalid {name} binding")
    source_root = Path(inputs).parent
    if Path(frame_root).resolve() != source_root.resolve():
        raise ValueError("inputs and frame root must belong to the same frozen corpus")
    validate_corpus(source_root)

    processor, model = _load(model_id, revision)
    root.mkdir(parents=True)
    snapshot = root / "corpus"
    shutil.copytree(source_root, snapshot)
    corpus_manifest = validate_corpus(snapshot)
    before_digest = corpus_tree_sha256(snapshot)
    cases = load_cases(snapshot / "frozen_inputs.jsonl")
    partial = root / "raw_completions.partial.jsonl"
    running_path = root / "RUNNING.json"
    runtime = _runtime_provenance(
        model_id=model_id,
        revision=revision,
        presentation_mode=presentation_mode,
        processor=processor,
        model=model,
    )
    running = {
        "status": "INCOMPLETE_DO_NOT_ANALYZE",
        "records_completed": 0,
        "case_count": len(cases),
        "model_id": model_id,
        "revision": revision,
        "presentation_mode": presentation_mode,
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "git_commit": git_commit,
        "corpus_tree_sha256": before_digest,
        "runtime": runtime,
    }
    write_json(running_path, running)
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        for record in collect(
            cases,
            frame_root=snapshot,
            model_id=model_id,
            revision=revision,
            presentation_mode=presentation_mode,
            processor=processor,
            model=model,
            max_new_tokens=max_new_tokens,
        ):
            handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            running["records_completed"] += 1
            write_json(running_path, running)
    after_digest = corpus_tree_sha256(snapshot)
    if after_digest != before_digest:
        raise RuntimeError("bound frame snapshot changed during inference")
    raw = root / "raw_completions.jsonl"
    partial.replace(raw)
    manifest = {
        "kind": "visual_hindsight_g0_evidence",
        "schema_version": "visual-hindsight-g0-v2",
        "model_id": model_id,
        "revision": revision,
        "presentation_mode": presentation_mode,
        "case_count": len(cases),
        "pair_count": corpus_manifest["pairs"],
        "max_new_tokens": max_new_tokens,
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "git_commit": git_commit,
        "corpus_tree_sha256": before_digest,
        "raw_sha256": sha256_file(raw),
        "runtime": runtime,
    }
    write_json(root / "MANIFEST.json", manifest)
    running_path.unlink()
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen VLM inference on visual-hindsight G0 v2")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--presentation-mode", choices=PRESENTATION_MODES, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    print(canonical_json(run(**vars(args))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
