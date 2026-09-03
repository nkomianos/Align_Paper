"""Lossy viewing copies only: never change or evaluate on the sealed frame sets."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2


def render(paths, output, fps=4):
    first = cv2.imread(str(paths[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise ValueError("preview encoder unavailable")
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None or frame.shape != first.shape:
            raise ValueError("source frame mismatch")
        writer.write(frame)
    writer.release()
    reader = cv2.VideoCapture(str(output))
    count = 0
    while reader.read()[0]:
        count += 1
    reader.release()
    if count != len(paths):
        raise ValueError("preview decode mismatch")


def main(root, ep0, hindsight):
    root.mkdir(parents=True, exist_ok=False)
    ep_cases = [json.loads(line) for line in (ep0 / "cases.jsonl").read_text().splitlines()]
    case = next(c for c in ep_cases if c["scene"] == "scene-006" and c["condition"] == "native_rgb")
    jobs = {"camera_follows_object.mp4": [ep0 / p for p in case["frame_paths"]]}
    vh_cases = [json.loads(line) for line in (hindsight / "frozen_inputs.jsonl").read_text().splitlines()]
    for arm in ("cf0_past", "cf1_past"):
        case = next(c for c in vh_cases if c["pair_id"] == "vh2-0000" and c["arm"] == arm)
        jobs[f"hindsight_{arm}.mp4"] = [hindsight / p for p in case["frame_paths"]]
    manifest = {"purpose": "viewing_copies_NOT_experiment_inputs", "files": {}}
    for name, paths in jobs.items():
        render(paths, root / name)
        manifest["files"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    with (root / "MANIFEST.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ep0", type=Path, required=True)
    parser.add_argument("--hindsight", type=Path, required=True)
    args = parser.parse_args()
    main(args.output, args.ep0, args.hindsight)
