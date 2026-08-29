"""Generate semantic-preserving integer translations of visual primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import random
from typing import Any, Iterable

from PIL import Image, ImageDraw

from under_extinction.io import read_jsonl, sha256_file, write_json, write_jsonl


@dataclass(frozen=True)
class PhaseCase:
    image_id: str
    base_id: str
    split: str
    task: str
    thickness: str
    phase_x: int
    prompt: str
    answer: str
    image_path: str


def _draw_count(draw: ImageDraw.ImageDraw, *, x: int, y: int, thickness: int, count: int) -> None:
    radius = 5 if thickness == 1 else 11
    for index in range(count):
        cx = x + (index % 4) * 48
        cy = y + (index // 4) * 58
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill="black")


def _draw_closure(draw: ImageDraw.ImageDraw, *, x: int, y: int, thickness: int, closed: bool) -> None:
    points = [(x, y + 90), (x + 70, y), (x + 140, y + 90), (x + 70, y + 160)]
    segments = list(zip(points, points[1:] + points[:1]))
    if not closed:
        segments = segments[:-1]
    for left, right in segments:
        draw.line((left, right), fill="black", width=thickness)


def _draw_crossing(draw: ImageDraw.ImageDraw, *, x: int, y: int, thickness: int, crossing: bool) -> None:
    draw.line((x, y, x + 170, y + 170), fill="black", width=thickness)
    if crossing:
        draw.line((x, y + 170, x + 170, y), fill="black", width=thickness)
    else:
        draw.line((x, y + 45, x + 125, y + 170), fill="black", width=thickness)


def _spec(index: int) -> tuple[str, str, str, dict[str, Any]]:
    task = ("count", "closure", "crossing")[index % 3]
    if task == "count":
        count = 3 + index % 6
        return task, "How many separate black dots are shown? Answer with one integer only.", str(count), {"count": count}
    if task == "closure":
        closed = index % 2 == 0
        return task, "Do the black line segments form one completely closed shape? Answer yes or no only.", "yes" if closed else "no", {"closed": closed}
    crossing = index % 2 == 0
    return task, "Do the two black paths cross each other? Answer yes or no only.", "yes" if crossing else "no", {"crossing": crossing}


def _render(path: Path, *, task: str, payload: dict[str, Any], phase_x: int, thickness: int, origin_x: int, origin_y: int, size: int) -> None:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    x, y = origin_x + phase_x, origin_y
    if task == "count":
        _draw_count(draw, x=x, y=y, thickness=thickness, count=int(payload["count"]))
    elif task == "closure":
        _draw_closure(draw, x=x, y=y, thickness=thickness, closed=bool(payload["closed"]))
    else:
        _draw_crossing(draw, x=x, y=y, thickness=thickness, crossing=bool(payload["crossing"]))
    image.save(path, format="PNG", optimize=False)


def build_corpus(root: str | Path, *, bases: int = 60, phases: int = 32, image_size: int = 896, seed: int = 20260829) -> tuple[PhaseCase, ...]:
    destination = Path(root)
    if destination.exists():
        raise FileExistsError("refusing to overwrite a visual-phase corpus")
    if bases < 30 or phases < 28 or image_size < 512:
        raise ValueError("visual-phase gate requires >=30 bases, >=28 phases, and >=512px images")
    images = destination / "images"
    images.mkdir(parents=True)
    rng = random.Random(seed)
    cases: list[PhaseCase] = []
    hashes: list[str] = []
    for index in range(bases):
        task, prompt, answer, payload = _spec(index)
        split = "DEV" if index < bases // 3 else "TEST"
        # Multiples of 112 align both the 16px Qwen and 14px Gemma patch grids.
        origin_x = 224 + 112 * (index % 2)
        origin_y = 224 + 112 * ((index // 2) % 2)
        for thickness_name, width in (("thin", 1), ("thick", 9)):
            for phase in range(phases):
                image_id = f"vp-{index:04d}-{thickness_name}-x{phase:02d}"
                relative = f"images/{image_id}.png"
                path = destination / relative
                _render(path, task=task, payload=payload, phase_x=phase, thickness=width, origin_x=origin_x, origin_y=origin_y, size=image_size)
                cases.append(PhaseCase(image_id, f"vp-{index:04d}", split, task, thickness_name, phase, prompt, answer, relative))
                hashes.append(f"{image_id}:{sha256_file(path)}")
    inputs = destination / "frozen_inputs.jsonl"
    write_jsonl(inputs, (asdict(case) for case in cases))
    aggregate = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    write_json(destination / "CORPUS_MANIFEST.json", {
        "kind": "visual_patch_phase_corpus",
        "bases": bases,
        "phases": phases,
        "image_size": image_size,
        "seed": seed,
        "case_count": len(cases),
        "inputs_sha256": sha256_file(inputs),
        "ordered_image_digest": aggregate,
    })
    return tuple(cases)


def load_cases(path: str | Path) -> tuple[PhaseCase, ...]:
    cases = tuple(PhaseCase(**row) for row in read_jsonl(path))
    ids = [case.image_id for case in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("visual-phase corpus must be non-empty with unique image IDs")
    return cases
