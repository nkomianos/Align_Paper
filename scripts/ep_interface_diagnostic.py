"""Prospectively fixed 24-call EP interface diagnosis; no paper gate."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from efference_pair.pilot import (scene, decompose, dump, seal, validate_seal,
    digest, MODEL_ID, MODEL_REVISION)

VERSION = "ep-interface-diagnostic-v1"
PREPARED_SHA = "87bb923231812e17d08247307942b63d3b28dda0bbc7a1f1cd03ba81931495e8"
MODES = ("native_video", "ordered_images", "first_frame", "cv_text")
DIRECTIONS = ("left", "right", "stationary")


def json_metadata(value):
    """Normalize loader metadata without editing the inherited artifact writer."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): json_metadata(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_metadata(v) for v in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [json_metadata(v) for v in value]
    if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device"):
        return str(value)
    raise TypeError(f"unsupported provenance metadata type: {type(value).__name__}")


def direction(x):
    return "stationary" if abs(x) < .5 else "right" if x > 0 else "left"


def prompt_for(task, options):
    context = ("The camera is fixed. Determine the RED SQUARE's horizontal motion "
               "relative to the stationary background, not its absolute location."
               if task == "object" else
               "The background AND red square are stationary in the world. Only the "
               "camera may translate horizontally, without rotation or zoom. Determine "
               "the CAMERA's translation in world coordinates. A camera moving right "
               "makes the stationary world shift left in the image, and conversely.")
    return ("Frames, when present, are in chronological order at 2 frames per second. "
            "Horizontal image coordinates increase to the right. " + context + "\n" +
            " ".join(f"{chr(65+i)}: {v}" for i, v in enumerate(options)) +
            "\nReply with exactly one letter: A, B, or C.")


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    cases, checks = [], []
    for task_index, task in enumerate(("object", "camera")):
        for i, native_index in enumerate((0, 1, 4)):
            name = f"{task}-{i}"
            frames, _, _, _ = scene(4 if task == "object" else native_index,
                                   native_index if task == "object" else 4)
            raw, glob, residual, _, _ = decompose(frames)
            # RGB-derived independent oracle: dominant flow plus red-pixel centroid.
            red = (frames[..., 0].astype(int) > frames[..., 1].astype(int) + 60)
            centers = [float(np.where(mask)[1].mean()) for mask in red]
            red_dx = (centers[-1] - centers[0]) / 15
            background_dx = float(np.median(glob[..., 0]))
            estimate = red_dx - background_dx if task == "object" else -background_dx
            expected = DIRECTIONS[i]
            if direction(estimate) != expected:
                raise ValueError("RGB oracle failed before any model output")
            folder = output / name
            folder.mkdir()
            paths = []
            for j, frame in enumerate(frames):
                p = folder / f"{j:02d}.png"
                Image.fromarray(frame).save(p)
                paths.append(p.relative_to(output).as_posix())
            options = list(DIRECTIONS[task_index:] + DIRECTIONS[:task_index])
            cases.append({"id": name, "task": task, "frames": paths,
                          "expected": expected, "options": options,
                          "answer": chr(65 + options.index(expected)),
                          "prompt": prompt_for(task, options),
                          "cv_text": f"Measured image displacement per frame: background {background_dx:.2f} horizontal pixels; red square {red_dx:.2f} horizontal pixels.\n"})
            checks.append({"id": name, "background_dx": background_dx,
                           "red_dx": red_dx, "estimate": estimate,
                           "predicted": direction(estimate), "expected": expected,
                           "first_last_mean_absolute_difference": float(np.abs(frames[-1].astype(float)-frames[0]).mean()),
                           "first_frame_sha256": digest(output / paths[0])})
    # Identical first frames across all directions preclude visual direction leakage.
    if len({c["first_frame_sha256"] for c in checks}) != 1:
        raise ValueError("first-frame negative control not identical")
    dump(output / "cases.json", cases)
    dump(output / "preflight.json", {"version": VERSION, "checks": checks,
         "calls": 24, "model": MODEL_ID, "revision": MODEL_REVISION})
    return seal(output)


def encode(processor, case, mode, root):
    paths = [str(root / p) for p in case["frames"]]
    kw = {}
    if mode == "native_video":
        from transformers.video_utils import VideoMetadata
        content = [{"type": "video", "video": paths}]
        kw["processor_kwargs"] = {"do_sample_frames": False,
            "video_metadata": VideoMetadata(total_num_frames=16, fps=2., width=224,
                height=224, duration=8., frames_indices=list(range(16)))}
    elif mode in ("ordered_images", "first_frame"):
        content = [{"type": "image", "image": p} for p in
                   (paths if mode == "ordered_images" else paths[:1])]
    else:
        content = []
    content.append({"type": "text", "text": (case["cv_text"] if mode == "cv_text" else "") + case["prompt"]})
    return processor.apply_chat_template([{"role": "user", "content": content}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **kw)


def contract(inputs):
    return {"input_tokens": int(inputs["input_ids"].shape[-1]),
            "input_ids": inputs["input_ids"][0].tolist(),
            **{key: inputs[key].tolist() for key in ("image_grid_thw", "video_grid_thw") if key in inputs}}


def preserve_model_identity(output):
    """Hash only the pinned public model snapshot, never other HF cache content."""
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    if snapshot.name != MODEL_REVISION:
        raise ValueError("unexpected cached model snapshot identity")
    files = {p.relative_to(snapshot).as_posix(): digest(p)
             for p in sorted(snapshot.rglob("*")) if p.is_file()}
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("cached model weight files absent")
    tokenizer = output / "tokenizer"
    tokenizer.mkdir()
    portable = {}
    for name, sha in files.items():
        if "/" not in name and (name.startswith(("tokenizer", "chat_template")) or
                name in ("config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")):
            shutil.copy2(snapshot / name, tokenizer / name)
            portable[name] = sha
    if "tokenizer.json" not in portable or "tokenizer_config.json" not in portable:
        raise ValueError("portable tokenizer incomplete")
    dump(output / "model_identity.json", {"model": MODEL_ID, "revision": MODEL_REVISION,
         "snapshot_files_sha256": files, "portable_tokenizer_sha256": portable})


def safe_report_path(root, output):
    if output.exists() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError("report must be a fresh path outside immutable evidence")


def preflight(prepared, output):
    from transformers import AutoProcessor
    validate_seal(prepared)
    if digest(prepared / "MANIFEST.json") != PREPARED_SHA:
        raise ValueError("prepared manifest differs from prospective pin")
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    cases = json.loads((prepared / "cases.json").read_text())
    contracts = {c["id"]+"/"+mode: contract(encode(processor, c, mode, prepared))
                 for c in cases for mode in MODES}
    for c in cases:
        v = contracts[c["id"]+"/native_video"]
        m = contracts[c["id"]+"/ordered_images"]
        if len(v.get("video_grid_thw", [])) != 1 or v["video_grid_thw"][0][0] != 8:
            raise ValueError("native-video temporal frame contract failed")
        if len(m.get("image_grid_thw", [])) != 16:
            raise ValueError("ordered-image frame contract failed")
    dump(output, {"version": VERSION, "prepared_manifest": digest(prepared / "MANIFEST.json"),
                  "contracts": contracts, "model": MODEL_ID, "revision": MODEL_REVISION})


def run(prepared, output):
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    validate_seal(prepared)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(prepared, output / "inputs")
    shutil.copy2(__file__, output / "runner.py")
    shutil.copy2(Path(__file__).parents[1] / "src/efference_pair/pilot.py", output / "pilot.py")
    try:
        preflight(prepared, output / "processor_preflight.json")
        preserve_model_identity(output)
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        model, loading = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID,
            revision=MODEL_REVISION, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()}, output_loading_info=True)
        dump(output / "loading_info.json", json_metadata(loading))
        if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("model loading keys/errors are not clean")
        model.eval()
        dump(output / "generation_config.json", {"base": json_metadata(model.generation_config.to_dict()),
             "overrides": {"do_sample": False, "max_new_tokens": 8},
             "decode": {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}})
        dump(output / "runtime.json", {"version": VERSION, "torch": torch.__version__,
            "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
            "model": MODEL_ID, "revision": MODEL_REVISION})
        cases = json.loads((prepared / "cases.json").read_text())
        order = [(c, mode) for c in cases for mode in MODES]
        np.random.default_rng(90904).shuffle(order)
        with (output / "raw.jsonl").open("x", encoding="utf-8") as stream:
            for c, mode in order:
                inputs = encode(processor, c, mode, prepared).to(model.device)
                start = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=8)
                torch.cuda.synchronize()
                answer = processor.decode(generated[0, inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True, clean_up_tokenization_spaces=False)
                stream.write(json.dumps({"id": c["id"], "mode": mode, "answer": answer,
                    "generated_ids": generated[0].tolist(),
                    "seconds": time.monotonic()-start, **contract(inputs)})+"\n")
                stream.flush(); os.fsync(stream.fileno())
        dump(output / "COMPLETE.json", {"calls": 24})
    except BaseException as exc:
        dump(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        seal(output)
        raise
    seal(output)


def verify(root, output, expected_runner_sha, expected_pilot_sha):
    safe_report_path(root, output)
    validate_seal(root)
    validate_seal(root / "inputs")
    if digest(root / "runner.py") != expected_runner_sha or digest(root / "pilot.py") != expected_pilot_sha:
        raise ValueError("source does not match externally pinned archive receipt")
    identity = json.loads((root / "model_identity.json").read_text())
    if identity["model"] != MODEL_ID or identity["revision"] != MODEL_REVISION:
        raise ValueError("model identity differs from pin")
    for name, sha in identity["portable_tokenizer_sha256"].items():
        path = (root / "tokenizer" / name).resolve()
        if not path.is_relative_to((root / "tokenizer").resolve()) or digest(path) != sha:
            raise ValueError("portable tokenizer hash mismatch")
        if identity["snapshot_files_sha256"].get(name) != sha:
            raise ValueError("tokenizer differs from recorded model snapshot")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    loading = json.loads((root / "loading_info.json").read_text())
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("model load not clean")
    generation = json.loads((root / "generation_config.json").read_text())
    if generation["overrides"] != {"do_sample": False, "max_new_tokens": 8}:
        raise ValueError("wrong generation overrides")
    if digest(root / "inputs/MANIFEST.json") != PREPARED_SHA:
        raise ValueError("prepared manifest differs from prospective pin")
    if json.loads((root / "COMPLETE.json").read_text()) != {"calls": 24}:
        raise ValueError("missing completion contract")
    cases = {c["id"]: c for c in json.loads((root / "inputs/cases.json").read_text())}
    rows = [json.loads(s) for s in (root / "raw.jsonl").read_text().splitlines()]
    keys = [(r["id"], r["mode"]) for r in rows]
    if len(keys) != 24 or set(keys) != {(c, m) for c in cases for m in MODES}:
        raise ValueError("missing/duplicate/unexpected outputs")
    pre = json.loads((root / "processor_preflight.json").read_text())
    if pre["prepared_manifest"] != PREPARED_SHA or pre["revision"] != MODEL_REVISION:
        raise ValueError("processor preflight pin mismatch")
    for row in rows:
        expected = pre["contracts"][row["id"]+"/"+row["mode"]]
        actual = {k: row[k] for k in ("input_tokens", "input_ids", "image_grid_thw", "video_grid_thw") if k in row}
        if actual != expected or row["input_tokens"] != len(row["input_ids"]):
            raise ValueError("inference input differs from processor preflight")
        n = row["input_tokens"]
        if row["generated_ids"][:n] != row["input_ids"] or not 1 <= len(row["generated_ids"])-n <= 8:
            raise ValueError("generated sequence prefix/length mismatch")
        decoded = tokenizer.decode(row["generated_ids"][n:], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
        if decoded != row["answer"]:
            raise ValueError("independent tokenizer decoding mismatch")
    results = {}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        results[mode] = {"correct": sum(r["answer"].strip() == cases[r["id"]]["answer"] for r in selected),
            "parsed": sum(r["answer"].strip() in ("A", "B", "C") for r in selected), "n": 6}
    dump(output, {"version": VERSION, "scope": "interface_diagnosis_not_paper_gate",
        "results": results, "rows": rows, "decision": "DESCRIPTIVE_ONLY_NO_EXPANSION",
        "expected_runner_sha256": expected_runner_sha, "expected_pilot_sha256": expected_pilot_sha,
        "weight_hash_scope": "remote recorded snapshot hashes; weights not downloaded/rehashed by portable verifier"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "preflight", "run", "verify"))
    p.add_argument("--prepared", type=Path)
    p.add_argument("--root", type=Path)
    p.add_argument("--expected-runner-sha")
    p.add_argument("--expected-pilot-sha")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.command == "prepare": prepare(a.output)
    elif a.command == "verify":
        if not a.expected_runner_sha or not a.expected_pilot_sha:
            p.error("verify requires externally pinned runner and helper SHA-256")
        verify(a.root, a.output, a.expected_runner_sha, a.expected_pilot_sha)
    else: globals()[a.command](a.prepared, a.output)
