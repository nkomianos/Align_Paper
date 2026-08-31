"""Frozen counterfactual videos for measuring future-to-past visual leakage."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl


SCHEMA_VERSION = "visual-hindsight-g0-v2"
LOCATIONS = ("A", "B", "C")
ARMS = ("prefix_past", "cf0_past", "cf1_past", "cf0_future", "cf1_future")
PAST_PROMPT = (
    "The frames are chronological. At the frame marked CHOICE, before the "
    "POST-CHOICE mechanism moved the token, which labeled box contained it? "
    "Return exactly one uppercase token: A, B, or C. Do not explain."
)
FUTURE_PROMPT = (
    "The frames are chronological. At the frame marked OUTCOME, after the "
    "POST-CHOICE mechanism moved the token, which labeled box contains it? "
    "Return exactly one uppercase token: A, B, or C. Do not explain."
)
LABEL_FONT = ImageFont.load_default(size=17)


@dataclass(frozen=True)
class HindsightCase:
    case_id: str
    pair_id: str
    split: str
    arm: str
    query_role: str
    world: str
    past_location: str
    outcome_location: str
    prompt: str
    answer: str
    frame_paths: tuple[str, ...]
    prefix_frame_count: int
    factors: Mapping[str, Any]


def _centered_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, fill: str = "black") -> None:
    box = draw.textbbox((0, 0), text, font=LABEL_FONT)
    width, height = box[2] - box[0], box[3] - box[1]
    draw.text((xy[0] - width // 2, xy[1] - height // 2), text, fill=fill, font=LABEL_FONT)


def _token(draw: ImageDraw.ImageDraw, xy: tuple[int, int], *, shape: str, fill: str) -> None:
    x, y = xy
    radius = 10
    bounds = (x - radius, y - radius, x + radius, y + radius)
    if shape == "circle":
        draw.ellipse(bounds, fill=fill, outline="black", width=2)
    else:
        draw.rectangle(bounds, fill=fill, outline="black", width=2)


def _physical_vertices(width: int, height: int) -> dict[str, tuple[int, int]]:
    side = min(150, width - 120, int((height - 70) / .87))
    if side < 86:
        raise ValueError("frame is too small for the equilateral three-location scene")
    half = side // 2
    vertical = round(side * math.sqrt(3) / 2)
    center_x = width // 2
    top_y = max(44, (height - vertical) // 2 - 4)
    return {
        "top": (center_x, top_y),
        "bottom_left": (center_x - half, top_y + vertical),
        "bottom_right": (center_x + half, top_y + vertical),
    }


def _location_points(width: int, height: int, layout_mirror: str) -> dict[str, tuple[int, int]]:
    vertices = _physical_vertices(width, height)
    order = (
        ("top", "bottom_left", "bottom_right")
        if layout_mirror == "standard"
        else ("top", "bottom_right", "bottom_left")
    )
    if layout_mirror not in ("standard", "mirrored"):
        raise ValueError("unknown layout mirror")
    return dict(zip(LOCATIONS, (vertices[name] for name in order), strict=True))


def _base_scene(
    *, width: int, height: int, factors: Mapping[str, Any]
) -> tuple[Image.Image, ImageDraw.ImageDraw, dict[str, tuple[int, int]]]:
    background = "#f7f4e8" if factors["background_tone"] == "warm" else "#eef4f7"
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    points = _location_points(width, height, str(factors["layout_mirror"]))
    box_w, box_h = max(46, width // 8), max(36, height // 8)
    for label in LOCATIONS:
        x, y = points[label]
        draw.rounded_rectangle(
            (x - box_w // 2, y - box_h // 2, x + box_w // 2, y + box_h // 2),
            radius=5,
            fill="#d7dde3",
            outline="black",
            width=3,
        )
        _centered_text(draw, (x, y + box_h // 2 + 15), label)
    return image, draw, points


def _interpolate(start: tuple[int, int], finish: tuple[int, int], progress: float) -> tuple[int, int]:
    return (
        round(start[0] + progress * (finish[0] - start[0])),
        round(start[1] + progress * (finish[1] - start[1])),
    )


def _render_prefix_frame(
    path: Path,
    *,
    index: int,
    count: int,
    width: int,
    height: int,
    past_location: str,
    factors: Mapping[str, Any],
) -> None:
    image, draw, points = _base_scene(width=width, height=height, factors=factors)
    start = (width // 2, height - 24)
    destination = points[past_location]
    settle_at = max(2, count - 2)
    token_xy = _interpolate(start, destination, min(1.0, index / settle_at))
    _token(draw, token_xy, shape=str(factors["token_shape"]), fill=str(factors["token_tone"]))
    _centered_text(draw, (width // 2, 18), "OBSERVE" if index < count - 2 else "CHOICE")
    if index >= count - 2:
        draw.line((width // 2, height - 24, destination[0], destination[1]), fill="#222222", width=4)
    if index == count - 1:
        _centered_text(draw, (width // 2, 35), "LOCKED")
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _suffix_positions(
    start: tuple[int, int], finish: tuple[int, int], count: int
) -> tuple[tuple[int, int], ...]:
    return tuple(_interpolate(start, finish, (index + 1) / count) for index in range(count))


def _polyline_length(start: tuple[int, int], positions: Sequence[tuple[int, int]]) -> float:
    points = (start, *positions)
    return sum(math.dist(first, second) for first, second in zip(points, points[1:]))


def _render_suffix_frame(
    path: Path,
    *,
    token_xy: tuple[int, int],
    outcome_location: str,
    final: bool,
    width: int,
    height: int,
    factors: Mapping[str, Any],
) -> None:
    image, draw, points = _base_scene(width=width, height=height, factors=factors)
    _token(draw, token_xy, shape=str(factors["token_shape"]), fill=str(factors["token_tone"]))
    _centered_text(draw, (width // 2, 18), "POST-CHOICE")
    if final:
        endpoint = points[outcome_location]
        _centered_text(draw, (endpoint[0], max(35, endpoint[1] - 39)), "OUTCOME")
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _factor_schedule(seed: int) -> list[dict[str, str]]:
    cells = [
        {
            "past_location": past,
            "token_shape": shape,
            "background_tone": background,
            "layout_mirror": mirror,
            "token_tone": tone,
        }
        for past, shape, background, mirror, tone in product(
            LOCATIONS,
            ("circle", "square"),
            ("warm", "cool"),
            ("standard", "mirrored"),
            ("#ffd43b", "#7ee787"),
        )
    ]
    if len(cells) != 48:  # pragma: no cover
        raise AssertionError("the frozen factorial no longer contains 48 cells")
    random.Random(seed).shuffle(cells)
    return cells


def _hash_lines(records: Sequence[Mapping[str, str]]) -> str:
    payload = "\n".join(canonical_json(dict(record)) for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def corpus_tree_sha256(root: str | Path) -> str:
    corpus_root = Path(root)
    rows = []
    for path in sorted(item for item in corpus_root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(corpus_root).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()


def build_corpus(
    root: str | Path,
    *,
    pairs: int = 48,
    width: int = 384,
    height: int = 288,
    prefix_frames: int = 8,
    suffix_frames: int = 4,
    seed: int = 20260830,
) -> tuple[HindsightCase, ...]:
    destination = Path(root)
    if destination.exists():
        raise FileExistsError("refusing to overwrite a visual-hindsight corpus")
    if pairs != 48:
        raise ValueError("visual-hindsight G0 v2 is frozen to exactly 48 factorial pairs")
    if width < 240 or height < 210 or prefix_frames < 4 or suffix_frames < 3:
        raise ValueError("frame geometry or temporal horizon is too small")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    destination.mkdir(parents=True)
    cases: list[HindsightCase] = []
    prefix_identity: list[dict[str, str]] = []
    motion_records: list[dict[str, Any]] = []
    factor_counts: dict[str, Counter[str]] = {
        name: Counter()
        for name in ("past_location", "token_shape", "background_tone", "layout_mirror", "token_tone")
    }

    for pair_index, factors in enumerate(_factor_schedule(seed)):
        pair_id = f"vh2-{pair_index:04d}"
        past_location = factors["past_location"]
        endpoints = tuple(location for location in LOCATIONS if location != past_location)
        pair_root = destination / "frames" / pair_id
        prefix_root = pair_root / "prefix"
        full_roots = (pair_root / "cf0_full", pair_root / "cf1_full")
        for directory in (prefix_root, *full_roots):
            directory.mkdir(parents=True)

        prefix_paths: list[str] = []
        full_paths: list[list[str]] = [[], []]
        for frame_index in range(prefix_frames):
            name = f"frame_{frame_index:03d}.png"
            prefix_path = prefix_root / name
            _render_prefix_frame(
                prefix_path,
                index=frame_index,
                count=prefix_frames,
                width=width,
                height=height,
                past_location=past_location,
                factors=factors,
            )
            copied = []
            for world_index, full_root in enumerate(full_roots):
                target = full_root / name
                shutil.copyfile(prefix_path, target)
                full_paths[world_index].append(target.relative_to(destination).as_posix())
                copied.append(target)
            shared_hash = sha256_file(prefix_path)
            if any(sha256_file(path) != shared_hash for path in copied):
                raise RuntimeError("counterfactual causal prefixes are not byte-identical")
            prefix_identity.append({"pair_id": pair_id, "frame": name, "sha256": shared_hash})
            prefix_paths.append(prefix_path.relative_to(destination).as_posix())

        points = _location_points(width, height, factors["layout_mirror"])
        start = points[past_location]
        world_lengths: list[float] = []
        for world_index, (endpoint, full_root) in enumerate(zip(endpoints, full_roots, strict=True)):
            positions = _suffix_positions(start, points[endpoint], suffix_frames)
            world_lengths.append(_polyline_length(start, positions))
            for suffix_index, token_xy in enumerate(positions):
                frame_index = prefix_frames + suffix_index
                path = full_root / f"frame_{frame_index:03d}.png"
                _render_suffix_frame(
                    path,
                    token_xy=token_xy,
                    outcome_location=endpoint,
                    final=suffix_index == suffix_frames - 1,
                    width=width,
                    height=height,
                    factors=factors,
                )
                full_paths[world_index].append(path.relative_to(destination).as_posix())
        distance_delta = abs(world_lengths[0] - world_lengths[1])
        if distance_delta > 1.0:
            raise RuntimeError("counterfactual suffix motion is not distance matched")
        motion_records.append(
            {
                "pair_id": pair_id,
                "cf0_distance": round(world_lengths[0], 6),
                "cf1_distance": round(world_lengths[1], 6),
                "absolute_delta": round(distance_delta, 6),
            }
        )

        common = dict(
            pair_id=pair_id,
            split="G0",
            past_location=past_location,
            prefix_frame_count=prefix_frames,
            factors=dict(factors),
        )
        cases.append(
            HindsightCase(
                case_id=f"{pair_id}:prefix_past",
                arm="prefix_past",
                query_role="past",
                world="prefix",
                outcome_location=past_location,
                prompt=PAST_PROMPT,
                answer=past_location,
                frame_paths=tuple(prefix_paths),
                **common,
            )
        )
        for world_index, endpoint in enumerate(endpoints):
            world = f"cf{world_index}"
            for query_role, prompt, answer in (
                ("past", PAST_PROMPT, past_location),
                ("future", FUTURE_PROMPT, endpoint),
            ):
                arm = f"{world}_{query_role}"
                cases.append(
                    HindsightCase(
                        case_id=f"{pair_id}:{arm}",
                        arm=arm,
                        query_role=query_role,
                        world=world,
                        outcome_location=endpoint,
                        prompt=prompt,
                        answer=answer,
                        frame_paths=tuple(full_paths[world_index]),
                        **common,
                    )
                )
        for name in factor_counts:
            factor_counts[name][str(factors[name])] += 1

    unique_paths = sorted({relative for case in cases for relative in case.frame_paths})
    frame_hashes = [
        {"path": relative, "sha256": sha256_file(destination / relative)} for relative in unique_paths
    ]
    inputs_path = destination / "frozen_inputs.jsonl"
    hashes_path = destination / "FRAME_HASHES.jsonl"
    write_jsonl(inputs_path, (asdict(case) for case in cases))
    write_jsonl(hashes_path, frame_hashes)
    manifest = {
        "kind": "visual_hindsight_g0_corpus",
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "pairs": pairs,
        "case_count": len(cases),
        "arms": list(ARMS),
        "locations": list(LOCATIONS),
        "frame_width": width,
        "frame_height": height,
        "prefix_frames": prefix_frames,
        "suffix_frames": suffix_frames,
        "past_prompt": PAST_PROMPT,
        "future_prompt": FUTURE_PROMPT,
        "counterbalance": {name: dict(sorted(counts.items())) for name, counts in factor_counts.items()},
        "prefix_identity_verified": True,
        "prefix_identity_digest": _hash_lines(prefix_identity),
        "motion_match_verified": True,
        "max_motion_distance_delta": max(float(row["absolute_delta"]) for row in motion_records),
        "motion_digest": hashlib.sha256(canonical_json(motion_records).encode("utf-8")).hexdigest(),
        "ordered_frame_digest": _hash_lines(frame_hashes),
        "inputs_sha256": sha256_file(inputs_path),
        "frame_hashes_sha256": sha256_file(hashes_path),
    }
    write_json(destination / "CORPUS_MANIFEST.json", manifest)
    return tuple(cases)


def _safe_relative(path: str) -> bool:
    pure = PurePosixPath(path)
    return bool(path) and not pure.is_absolute() and ".." not in pure.parts and "\\" not in path


def load_cases(path: str | Path) -> tuple[HindsightCase, ...]:
    cases: list[HindsightCase] = []
    expected_fields = set(HindsightCase.__dataclass_fields__)
    for raw_row in read_jsonl(path):
        if set(raw_row) != expected_fields:
            raise ValueError("visual-hindsight case schema mismatch")
        row = dict(raw_row)
        row["frame_paths"] = tuple(str(value) for value in row["frame_paths"])
        cases.append(HindsightCase(**row))
    if not cases:
        raise ValueError("visual-hindsight inputs are empty")
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("visual-hindsight case IDs must be unique")

    grouped: dict[str, list[HindsightCase]] = {}
    for case in cases:
        if case.arm not in ARMS or case.answer not in LOCATIONS:
            raise ValueError("invalid visual-hindsight arm or answer")
        if case.past_location not in LOCATIONS or case.outcome_location not in LOCATIONS:
            raise ValueError("invalid past or outcome location")
        if case.query_role not in ("past", "future") or case.world not in ("prefix", "cf0", "cf1"):
            raise ValueError("invalid visual-hindsight query role or world")
        if case.split != "G0" or not case.frame_paths or any(not _safe_relative(item) for item in case.frame_paths):
            raise ValueError("invalid visual-hindsight split or frame path")
        if not 1 <= case.prefix_frame_count <= len(case.frame_paths):
            raise ValueError("invalid visual-hindsight prefix frame count")
        grouped.setdefault(case.pair_id, []).append(case)

    for pair_id, members in grouped.items():
        by_arm = {case.arm: case for case in members}
        if set(by_arm) != set(ARMS) or len(members) != len(ARMS):
            raise ValueError(f"{pair_id} does not have exactly the five frozen arms")
        if {case.case_id for case in members} != {f"{pair_id}:{arm}" for arm in ARMS}:
            raise ValueError(f"{pair_id} has malformed case IDs")
        if len({canonical_json(dict(case.factors)) for case in members}) != 1:
            raise ValueError(f"{pair_id} changes factors across arms")
        if len({case.past_location for case in members}) != 1:
            raise ValueError(f"{pair_id} changes its fixed past")
        past = by_arm["prefix_past"].past_location
        endpoints = {by_arm["cf0_past"].outcome_location, by_arm["cf1_past"].outcome_location}
        if endpoints != set(LOCATIONS) - {past}:
            raise ValueError(f"{pair_id} lacks the two counterfactual endpoints")
        if by_arm["prefix_past"].answer != past or by_arm["prefix_past"].prompt != PAST_PROMPT:
            raise ValueError(f"{pair_id} has an invalid prefix query")
        for world in ("cf0", "cf1"):
            past_case, future_case = by_arm[f"{world}_past"], by_arm[f"{world}_future"]
            if past_case.frame_paths != future_case.frame_paths:
                raise ValueError(f"{pair_id} changes video between query roles")
            if past_case.answer != past or past_case.prompt != PAST_PROMPT:
                raise ValueError(f"{pair_id} has an invalid past query")
            if future_case.answer != future_case.outcome_location or future_case.prompt != FUTURE_PROMPT:
                raise ValueError(f"{pair_id} has an invalid future query")
            if past_case.outcome_location != future_case.outcome_location:
                raise ValueError(f"{pair_id} changes outcome across query roles")
    return tuple(cases)


def _expected_counterbalance() -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {
        name: Counter()
        for name in ("past_location", "token_shape", "background_tone", "layout_mirror", "token_tone")
    }
    for factors in _factor_schedule(0):
        for name in counts:
            counts[name][factors[name]] += 1
    return {name: dict(sorted(values.items())) for name, values in counts.items()}


def validate_corpus(root: str | Path, *, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    corpus_root = Path(root)
    manifest_path = corpus_root / "CORPUS_MANIFEST.json"
    inputs_path = corpus_root / "frozen_inputs.jsonl"
    hashes_path = corpus_root / "FRAME_HASHES.jsonl"
    if not manifest_path.is_file() or not inputs_path.is_file() or not hashes_path.is_file():
        raise FileNotFoundError("visual-hindsight corpus is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "visual_hindsight_g0_corpus" or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("not a visual-hindsight G0 v2 corpus")
    if sha256_file(inputs_path) != manifest.get("inputs_sha256") or sha256_file(hashes_path) != manifest.get(
        "frame_hashes_sha256"
    ):
        raise ValueError("visual-hindsight manifest checksum mismatch")

    cases = load_cases(inputs_path)
    groups = {case.pair_id for case in cases}
    if len(groups) != 48 or len(cases) != 48 * len(ARMS):
        raise ValueError("visual-hindsight G0 must contain exactly 48 complete pairs")
    if groups != {f"vh2-{index:04d}" for index in range(48)}:
        raise ValueError("visual-hindsight pair identifiers are not frozen")
    if int(manifest.get("pairs", -1)) != 48 or int(manifest.get("case_count", -1)) != len(cases):
        raise ValueError("visual-hindsight manifest cardinality mismatch")
    if manifest.get("arms") != list(ARMS) or manifest.get("locations") != list(LOCATIONS):
        raise ValueError("visual-hindsight manifest arm or location mismatch")
    if manifest.get("past_prompt") != PAST_PROMPT or manifest.get("future_prompt") != FUTURE_PROMPT:
        raise ValueError("visual-hindsight prompt commitment mismatch")
    expected_factor_fields = {"past_location", "token_shape", "background_tone", "layout_mirror", "token_tone"}
    pair_factor_rows = []
    computed_counts = {name: Counter() for name in expected_factor_fields}
    for pair_id in sorted(groups):
        prefix_case = next(case for case in cases if case.case_id == f"{pair_id}:prefix_past")
        factors = dict(prefix_case.factors)
        if set(factors) != expected_factor_fields or factors["past_location"] != prefix_case.past_location:
            raise ValueError("visual-hindsight factor schema or past-location binding mismatch")
        pair_factor_rows.append(canonical_json(factors))
        for name in computed_counts:
            computed_counts[name][str(factors[name])] += 1
    computed_counterbalance = {
        name: dict(sorted(values.items())) for name, values in computed_counts.items()
    }
    if len(set(pair_factor_rows)) != 48 or computed_counterbalance != _expected_counterbalance():
        raise ValueError("visual-hindsight factorial cells are missing or duplicated")
    if manifest.get("counterbalance") != computed_counterbalance:
        raise ValueError("visual-hindsight factorial counterbalance mismatch")
    if manifest.get("prefix_identity_verified") is not True or manifest.get("motion_match_verified") is not True:
        raise ValueError("visual-hindsight causal controls were not verified")
    if float(manifest.get("max_motion_distance_delta", 999.0)) > 1.0:
        raise ValueError("counterfactual suffix motion is not matched")

    if expected is not None:
        comparisons = {
            "pairs": 48,
            "frame_width": int(expected["frame_width"]),
            "frame_height": int(expected["frame_height"]),
            "prefix_frames": int(expected["prefix_frames"]),
            "suffix_frames": int(expected["suffix_frames"]),
            "seed": int(expected["seed"]),
            "arms": list(expected["arms"]),
            "locations": list(expected["locations"]),
        }
        for key, value in comparisons.items():
            if manifest.get(key) != value:
                raise ValueError(f"visual-hindsight config/corpus mismatch: {key}")

    rows = list(read_jsonl(hashes_path))
    if any(set(row) != {"path", "sha256"} or not _safe_relative(str(row["path"])) for row in rows):
        raise ValueError("visual-hindsight frame-hash schema mismatch")
    row_paths = [str(row["path"]) for row in rows]
    expected_paths = sorted({relative for case in cases for relative in case.frame_paths})
    if row_paths != expected_paths or len(row_paths) != len(set(row_paths)):
        raise ValueError("visual-hindsight frame inventory mismatch")
    for row in rows:
        path = corpus_root / str(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"visual-hindsight frame checksum mismatch: {path}")
    if _hash_lines(rows) != manifest.get("ordered_frame_digest"):
        raise ValueError("visual-hindsight ordered frame digest mismatch")

    actual_files = sorted(path.relative_to(corpus_root).as_posix() for path in corpus_root.rglob("*") if path.is_file())
    allowed_files = sorted(["CORPUS_MANIFEST.json", "FRAME_HASHES.jsonl", "frozen_inputs.jsonl", *expected_paths])
    if actual_files != allowed_files:
        raise ValueError("visual-hindsight corpus contains missing or extra files")

    by_case = {case.case_id: case for case in cases}
    prefix_records: list[dict[str, str]] = []
    for pair_id in sorted(groups):
        prefix = by_case[f"{pair_id}:prefix_past"]
        cf0 = by_case[f"{pair_id}:cf0_past"]
        cf1 = by_case[f"{pair_id}:cf1_past"]
        if len(prefix.frame_paths) != int(manifest["prefix_frames"]):
            raise ValueError("prefix arm frame count mismatch")
        expected_full_count = int(manifest["prefix_frames"]) + int(manifest["suffix_frames"])
        if len(cf0.frame_paths) != expected_full_count or len(cf1.frame_paths) != expected_full_count:
            raise ValueError("counterfactual full arm frame count mismatch")
        for index in range(prefix.prefix_frame_count):
            values = [sha256_file(corpus_root / case.frame_paths[index]) for case in (prefix, cf0, cf1)]
            if len(set(values)) != 1:
                raise ValueError(f"{pair_id} has a non-identical causal prefix")
            prefix_records.append(
                {"pair_id": pair_id, "frame": f"frame_{index:03d}.png", "sha256": values[0]}
            )
    if _hash_lines(prefix_records) != manifest.get("prefix_identity_digest"):
        raise ValueError("visual-hindsight prefix identity digest mismatch")
    return manifest
