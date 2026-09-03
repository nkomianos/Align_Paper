"""Deterministic camera/object motion stimuli and label-blind decomposition.

EP0 uses an orthographic textured plane, NOT arbitrary 3-D camera inference.
The estimator receives RGB only. Simulator vectors are used exclusively for
oracle controls, ground truth, and CPU estimator validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

VERSION = "efference-pair-ep0-v3"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SIZE, FRAMES, SEED = 224, 16, 43117
DIRECTIONS = ("left", "right", "up", "down", "stationary")
VECTORS = np.array([[-2., 0], [2., 0], [0, -2.], [0, 2.], [0, 0]])
CONDITIONS = ("native_rgb", "rgb_layout", "raw_flow", "global_only",
              "residual_only", "joint", "oracle_joint", "sign_reverse", "sham")
ANCHORS = tuple(range(0, 16, 2))
PAIRS = (0, 2, 4, 6, 8, 10, 12, 14)
SETTINGS = {
    "version": VERSION, "size": SIZE, "frames": FRAMES, "seed": SEED,
    "conditions": CONDITIONS, "model_id": MODEL_ID, "revision": MODEL_REVISION,
    "flow": {"name": "Farneback", "pyr_scale": .5, "levels": 3,
             "winsize": 21, "iterations": 5, "poly_n": 7, "poly_sigma": 1.5},
    "global_fit": "affine_partial_RANSAC_grid8_threshold0.7_seed43117",
    "arrow_scale": 9.0, "max_new_tokens": 8,
    "timestamps": "explicit_neutral_canvas_header",
    "decision_scope": "synthetic_apparatus_only_not_formal_G0",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    # All artifacts are new: caller must choose a fresh directory.
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def seal(root: Path) -> dict:
    hashes = {p.relative_to(root).as_posix(): digest(p)
              for p in sorted(root.rglob("*")) if p.is_file() and p != root / "MANIFEST.json"}
    manifest = {"version": VERSION, "files": hashes}
    dump(root / "MANIFEST.json", manifest)
    return manifest


def validate_seal(root: Path) -> dict:
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError("wrong artifact version")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p != root / "MANIFEST.json"}
    if actual != set(manifest["files"]):
        raise ValueError("missing or extra artifact")
    for name, sha in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != sha:
            raise ValueError(f"integrity failure: {name}")
    return manifest


def scene(camera: int, obj: int, seed: int = SEED):
    """Return RGB, true background flow, true world-object residual, masks."""
    rng = np.random.default_rng(seed)
    side = SIZE + 128
    texture = rng.integers(40, 210, (side, side), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), .6)
    background = np.repeat(texture[..., None], 3, axis=2)
    tile = rng.integers(0, 110, (40, 40), dtype=np.uint8)
    tile = cv2.GaussianBlur(tile, (3, 3), .6)
    red = np.stack([tile + 145, tile, tile // 2], axis=-1)
    bg, residual = -VECTORS[camera], VECTORS[obj]
    frames, masks = [], []
    for t in range(FRAMES):
        transform = np.array([[1, 0, bg[0] * t], [0, 1, bg[1] * t]], dtype=np.float32)
        moved = cv2.warpAffine(background, transform, (side, side), flags=cv2.INTER_LINEAR)
        frame = moved[64:64 + SIZE, 64:64 + SIZE].copy()
        center = np.rint(SIZE / 2 + (bg + residual) * t).astype(int)
        x, y = center - 20
        if not (0 <= x <= SIZE - 40 and 0 <= y <= SIZE - 40):
            raise ValueError("simulator object out of bounds")
        frame[y:y + 40, x:x + 40] = red
        mask = np.zeros((SIZE, SIZE), bool)
        mask[y + 10:y + 30, x + 10:x + 30] = True
        frames.append(frame)
        masks.append(mask)
    return np.stack(frames), bg, residual, np.stack(masks)


def decompose(frames: np.ndarray):
    """RGB-only Farneback + majority-consensus global affine estimate."""
    if frames.shape != (FRAMES, SIZE, SIZE, 3) or frames.dtype != np.uint8:
        raise ValueError("wrong RGB frame contract")
    cv2.setNumThreads(1)
    raw, global_fields, residuals, transforms, diagnostics = [], [], [], [], []
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    grid = np.stack([xx, yy], axis=-1)
    points = grid[12:-12:8, 12:-12:8].reshape(-1, 2)
    for i in range(FRAMES - 1):
        a, b = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames[i:i + 2]]
        flow = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 21, 5, 7, 1.5, 0)
        values = flow[12:-12:8, 12:-12:8].reshape(-1, 2)
        cv2.setRNGSeed(SEED)
        matrix, inliers = cv2.estimateAffinePartial2D(
            points, points + values, method=cv2.RANSAC,
            ransacReprojThreshold=.7, maxIters=2000, confidence=.99, refineIters=10)
        failed = matrix is None or not np.isfinite(matrix).all()
        if failed:
            # Predeclared identity fallback: never drop failed estimator clips.
            matrix = np.eye(2, 3)
        global_flow = grid @ matrix[:, :2].T + matrix[:, 2] - grid
        raw.append(flow)
        global_fields.append(global_flow.astype(np.float32))
        residuals.append((flow - global_flow).astype(np.float32))
        transforms.append(matrix)
        diagnostics.append({"pair": i, "failed": bool(failed),
                            "inlier_fraction": float(np.mean(inliers)) if inliers is not None else 0.0})
    return np.array(raw), np.array(global_fields), np.array(residuals), np.array(transforms), diagnostics


def vector_canvas(field: np.ndarray, background=None, *, camera=False) -> np.ndarray:
    canvas = np.full((SIZE, SIZE, 3), 245, dtype=np.uint8) if background is None else background.copy()
    # Global apparent scene motion is opposite physical camera translation in EP0.
    values = -field if camera else field
    color = (20, 90, 225) if camera else (0, 185, 70)
    for y in range(12, SIZE - 12, 12):
        for x in range(12, SIZE - 12, 12):
            v = np.median(values[y - 2:y + 3, x - 2:x + 3], axis=(0, 1))
            if np.linalg.norm(v) < .22:
                continue
            end = np.rint(np.array([x, y]) + np.clip(v * 9, -35, 35)).astype(int)
            cv2.arrowedLine(canvas, (x, y), tuple(end), color, 1, cv2.LINE_AA, tipLength=.35)
    return canvas


def oracle_fields(bg: np.ndarray, residual: np.ndarray, masks: np.ndarray):
    g = np.broadcast_to(bg, (FRAMES - 1, SIZE, SIZE, 2)).copy().astype(np.float32)
    r = masks[:-1, ..., None] * residual
    return g, r.astype(np.float32)


def render_conditions(frames, raw, global_fields, residuals, transforms, oracle, sham):
    anchors = [frames[t].copy() for t in ANCHORS]
    cumulative = [np.eye(3)]
    for matrix in transforms:
        full = np.eye(3)
        full[:2] = matrix
        cumulative.append(full @ cumulative[-1])

    def residual_view(t, field):
        inv = np.linalg.inv(cumulative[t])
        rgb = cv2.warpAffine(frames[t], inv[:2], (SIZE, SIZE), borderValue=(245, 245, 245))
        warped = cv2.warpAffine(field, inv[:2], (SIZE, SIZE)) @ inv[:2, :2].T
        return vector_canvas(warped, rgb)

    def joint(g, r):
        # Same four temporal samples in both named channels; interleaved pairs.
        return [view for t in (0, 4, 8, 14)
                for view in (vector_canvas(g[t], camera=True), residual_view(t, r[t]))]

    return {
        "native_rgb": list(frames),
        "rgb_layout": anchors + [frames[t] for t in range(1, 16, 2)],
        "raw_flow": anchors + [vector_canvas(raw[t]) for t in PAIRS],
        "global_only": anchors + [vector_canvas(global_fields[t], camera=True) for t in PAIRS],
        "residual_only": anchors + [residual_view(t, residuals[t]) for t in PAIRS],
        "joint": anchors + joint(global_fields, residuals),
        "oracle_joint": anchors + joint(*oracle),
        "sign_reverse": anchors + joint(-global_fields, -residuals),
        "sham": anchors + joint(*sham),
    }


def question(stratum: str, options: list[str]) -> str:
    target = "camera relative to the stationary background" if stratum == "camera" else "red square relative to the stationary background, ignoring camera motion"
    return (
        "These time-stamped images show an orthographic camera viewing a textured stationary plane and one red square. "
        "Time stamps identify actual times; additional views may restart or repeat times. "
        "Directions are screen-axis directions (up is towards the top), not rotation. "
        "RGB images depict time. Where present, blue arrows show estimated CAMERA translation; "
        "green arrows on plain canvases show apparent image motion; green arrows on RGB show motion after camera compensation. "
        "No visible arrows means negligible estimated motion. Derived views may be imperfect; use the available evidence. "
        f"What is the direction of the {target}? "
        + " ".join(f"{chr(65 + i)}. {v}" for i, v in enumerate(options))
        + " Reply with exactly one letter A, B, C, D, or E."
    )


def time_labels(condition):
    if condition == "native_rgb":
        return [f"time {t:02d}" for t in range(16)]
    if condition == "rgb_layout":
        return [f"time {t:02d}" for t in (*ANCHORS, *range(1, 16, 2))]
    times = (0, 0, 4, 4, 8, 8, 14, 14) if condition in ("joint", "oracle_joint", "sign_reverse", "sham") else PAIRS
    return [f"time {t:02d}" for t in ANCHORS] + [f"time {t:02d} -> {t + 1:02d}" for t in times]


def prepare(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "settings.json", SETTINGS)
    cases, answers, diagnostics = [], {}, []
    for camera in range(5):
        for obj in range(5):
            idx = camera * 5 + obj
            scene_id = f"scene-{idx:03d}"
            # Shared seed across camera interventions: object, texture, and
            # world-relative trajectory are held fixed within each object group.
            frames, bg, rel, masks = scene(camera, obj, SEED + obj)
            raw, g, r, transforms, diag = decompose(frames)
            og, oracle_r = oracle_fields(bg, rel, masks)
            donor = (idx + 7) % 25
            donor_frames, *_ = scene(donor // 5, donor % 5, SEED + donor % 5)
            _, dg, dr, _, _ = decompose(donor_frames)
            views = render_conditions(frames, raw, g, r, transforms, (og, oracle_r), (dg, dr))
            errors = [float(np.linalg.norm(np.median(g[t], axis=(0, 1)) - bg)) for t in range(15)]
            object_errors = [float(np.linalg.norm(np.median(r[t][masks[t]], axis=0) - rel)) for t in range(15)]
            diagnostics.append({"scene": scene_id, "camera": DIRECTIONS[camera],
                                "object": DIRECTIONS[obj], "median_global_error_px": float(np.median(errors)),
                                "median_residual_error_px": float(np.median(object_errors)), "fit": diag})
            for condition, images in views.items():
                folder = output / scene_id / condition
                folder.mkdir(parents=True)
                paths = []
                for j, pixels in enumerate(images):
                    path = folder / f"{j:02d}.png"
                    labeled = pixels.copy()
                    labeled[:18] = 245
                    cv2.putText(labeled, time_labels(condition)[j], (5, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, .35, (20, 20, 20), 1, cv2.LINE_AA)
                    Image.fromarray(labeled).save(path)
                    paths.append(path.relative_to(output).as_posix())
                for stratum, label in (("camera", camera), ("object", obj)):
                    # Independent of condition, labels never enter the visual renderer.
                    order = np.random.default_rng(SEED + idx * 2 + (stratum == "object")).permutation(5)
                    options = [DIRECTIONS[k] for k in order]
                    case_id = f"{scene_id}/{stratum}/{condition}"
                    cases.append({"case_id": case_id, "scene": scene_id, "stratum": stratum,
                                  "condition": condition, "frame_paths": paths,
                                  "prompt": question(stratum, options)})
                    reverse = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}[label]
                    answers[case_id] = {"answer": chr(65 + options.index(DIRECTIONS[label])),
                                        "reversed_answer": chr(65 + options.index(DIRECTIONS[reverse])),
                                        "directional": label != 4, "static": label == 4,
                                        "object_group": obj}
    np.random.default_rng(SEED).shuffle(cases)
    with (output / "cases.jsonl").open("x", encoding="utf-8", newline="\n") as f:
        for case in cases:
            f.write(json.dumps(case, sort_keys=True) + "\n")
    dump(output / "answer_key.json", answers)
    dump(output / "estimator_diagnostics.json", diagnostics)
    dump(output / "preflight.json", {
        "case_count": len(cases), "scene_count": 25, "scientific_model_result": False,
        "opencv": cv2.__version__, "numpy": np.__version__,
        "estimator_numeric_check": all(x["median_global_error_px"] < .35 and x["median_residual_error_px"] < .5 for x in diagnostics),
        "note": "Checks apparatus only; not VLM accuracy or a paper pass.",
    })
    return seal(output)
