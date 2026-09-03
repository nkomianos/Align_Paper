"""Acquire a tiny, deterministic DEV-only media sample; never a paper test set.

Keeps upstream media private to ignored artifacts/, pins the HF revision, verifies
LFS SHA-256 values and decodes every frame. No external-video downloader needed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request

REPO = "zai-org/MotionBench"
REVISION = "f099db892172a015c489507c9abe56b036d960ef"
META = "MotionBench/video_info.meta.jsonl"
CATEGORIES = ("Camera Motion", "Location-related Motion", "Motion Recognition")


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def put(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def choose(rows, inventory, per_category=4):
    if per_category < 1:
        raise ValueError("positive sample size required")
    candidates = {name: [] for name in CATEGORIES}
    for row in rows:
        category = row.get("question_type")
        basename = row.get("video_path", "")
        if PurePosixPath(basename).name != basename:
            continue
        media = "MotionBench/self-collected/" + basename
        qa = row.get("qa", [])
        # Published DEV has answers; hidden-label TEST is never selected.
        if (category not in candidates or not qa
                or not all(q.get("answer") in tuple("ABCDE") for q in qa)
                or row.get("video_type") in ("Gaming", "industrial", "medical")
                or media not in inventory):
            continue
        size = inventory[media].get("size", 0)
        duration = row.get("video_info", {}).get("duration", 0)
        if not (0 < size <= 15_000_000 and 1 <= duration <= 20):
            continue
        candidates[category].append((hashlib.sha256(("motion-dev-v1/" + media).encode()).hexdigest(), row, media))
    selected, used = [], set()
    for category in CATEGORIES:
        count = 0
        for _, row, media in sorted(candidates[category], key=lambda entry: entry[0]):
            if media in used:
                continue
            selected.append({"annotation": row, "upstream_path": media})
            used.add(media)
            count += 1
            if count == per_category:
                break
        if count != per_category:
            raise ValueError(f"insufficient eligible DEV media for {category}")
    return selected


def prepare(output, per_category=4):
    import cv2

    output.mkdir(parents=True, exist_ok=False)
    try:
        info = json.loads(fetch(f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true"))
        if info["sha"] != REVISION or info.get("cardData", {}).get("license") != "cc-by-nc-sa-4.0":
            raise ValueError("revision/license changed")
        inventory = {item["rfilename"]: item for item in info["siblings"]}
        raw = fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{META}")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        selected = choose(rows, inventory, per_category)
        # Persist only chosen DEV annotations, not the full mixed DEV/TEST table.
        put(output / "selection.json", {"repo": REPO, "revision": REVISION,
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "selection_rule": "sha256 ordering, 4 per category by default, available self-collected DEV; no model outcomes",
            "license": "cc-by-nc-sa-4.0", "purpose": "development_decode_and_label_audit_NOT_HELD_OUT",
            "human_review_complete": False, "examples": selected})
        (output / "media").mkdir()
        (output / "previews").mkdir()
        records = []
        for index, item in enumerate(selected):
            upstream = item["upstream_path"]
            blob = inventory[upstream]
            payload = fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{upstream}")
            sha = hashlib.sha256(payload).hexdigest()
            expected = blob.get("lfs", {}).get("sha256")
            if expected is None or sha != expected or len(payload) != blob["size"]:
                raise ValueError("upstream LFS size/checksum mismatch")
            path = output / "media" / PurePosixPath(upstream).name
            with path.open("xb") as stream:
                stream.write(payload)
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"video decode failed: {path.name}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            decoded = 0
            targets = {0, expected_frames // 2, max(0, expected_frames - 1)}
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if decoded in targets:
                    frames.append(cv2.resize(frame, (256, 160)))
                decoded += 1
            cap.release()
            usable = fps > 0 and decoded >= 2 and decoded == expected_frames
            if frames and not cv2.imwrite(str(output / "previews" / f"{index:02d}.jpg"), cv2.hconcat(frames)):
                raise ValueError("preview write failed")
            records.append({"file": path.relative_to(output).as_posix(), "upstream_sha256": sha,
                            "bytes": len(payload), "fps": fps, "frames_decoded": decoded,
                            "duration_seconds": decoded / fps if fps > 0 else None,
                            "advertised_frames": expected_frames, "usable_decode": usable,
                            "decode_status": "OK" if usable else "QUARANTINED_DECODE_MISMATCH_NO_REPLACEMENT",
                            "category": item["annotation"]["question_type"]})
        put(output / "DECODE_REPORT.json", {"status": "MEDIA_BYTES_VERIFIED_DECODE_AUDITED_NOT_LABEL_VALIDATED",
            "clips": len(records), "total_bytes": sum(row["bytes"] for row in records),
            "usable_clips": sum(row["usable_decode"] for row in records),
            "quarantined_clips": sum(not row["usable_decode"] for row in records),
            "categories": dict(Counter(row["category"] for row in records)), "records": records})
        (output / "UPSTREAM_README.md").write_bytes(fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md"))
    except Exception as exc:
        put(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        put(output / "MANIFEST.json", {"files": {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*")) if path.is_file()}})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps({"clips_downloaded": len(prepare(args.output, args.per_category))}))
