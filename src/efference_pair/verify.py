"""Read-only EP0 verification; reports are written outside immutable evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from .pilot import CONDITIONS, MODEL_ID, MODEL_REVISION, SEED, SETTINGS, digest, dump, validate_seal
from .runner import planned_cases


def parse_answer(text):
    # Frozen parser: no post-hoc removal of fences or semantic rescue.
    cleaned = text.strip()
    return cleaned if re.fullmatch(r"[A-E]", cleaned) else None


def analyze(cases, answers, records):
    expected = {c["case_id"] for c in cases}
    if len(expected) != len(cases) or len(records) != len(expected):
        raise ValueError("duplicate cases or incomplete outputs")
    index = {r["case_id"]: r for r in records}
    if len(index) != len(records) or set(index) != expected:
        raise ValueError("duplicate, missing, or unexpected output")
    if set(answers) != expected:
        raise ValueError("answer key does not match exact workload")
    grouped, parsed = {}, []
    for case in cases:
        cid = case["case_id"]
        answer = parse_answer(index[cid]["completion"])
        parsed.append(answer is not None)
        grouped[(case["scene"], case["stratum"], case["condition"])] = (
            float(answer == answers[cid]["answer"]),
            float(answer == answers[cid]["reversed_answer"]), answers[cid])
    scenes = sorted({c["scene"] for c in cases})
    present = sorted({c["condition"] for c in cases})
    strata = ("camera", "object")
    scores = {c: np.array([[grouped[s, t, c][0] for t in strata] for s in scenes]) for c in present}
    accuracy = {c: {"macro": float(v.mean()), "camera": float(v[:, 0].mean()),
                    "object": float(v[:, 1].mean())} for c, v in scores.items()}
    report = {"parse_rate": float(np.mean(parsed)), "accuracy": accuracy,
              "scope": "synthetic_EP0_not_real_video_G0", "scene_count": len(scenes)}
    if set(present) != set(CONDITIONS):
        report["decision"] = "SMOKE_ONLY_NO_SCIENTIFIC_DECISION"
        return report
    native, joint = scores["native_rgb"], scores["joint"]
    # Five matched-camera interventions share an object/background world. Resample
    # whole object groups (5), NOT 450 forwards as independent observations.
    groups = np.array([grouped[s, "camera", "joint"][2]["object_group"] for s in scenes])
    def difference(a, b):
        d = (scores[a] - scores[b]).mean(axis=1)
        means = np.array([d[groups == g].mean() for g in sorted(set(groups))])
        rng = np.random.default_rng(SEED)
        draws = rng.choice(means, (10000, len(means)), replace=True).mean(axis=1)
        return {"point": float(d.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(),
                "independent_world_groups": len(means), "interpretation": "descriptive_small_n"}
    contrasts = {f"joint_minus_{c}": difference("joint", c)
                 for c in ("native_rgb", "rgb_layout", "raw_flow", "sham")}
    global_selectivity = float((scores["global_only"] - native).mean(axis=0) @ np.array([1, -1]))
    residual_selectivity = float((scores["residual_only"] - native).mean(axis=0) @ np.array([-1, 1]))
    eligible = [(s, t) for s in scenes for t in strata if grouped[s, t, "joint"][2]["directional"]]
    reverse_gain = float(np.mean([grouped[s, t, "sign_reverse"][1] - grouped[s, t, "joint"][1] for s, t in eligible]))
    static = [(s, t) for s in scenes for t in strata if grouped[s, t, "joint"][2]["static"]]
    static_change = float(np.mean([grouped[s, t, "joint"][0] - grouped[s, t, "native_rgb"][0] for s, t in static]))
    checks = {
        "joint_gain_8pp": contrasts["joint_minus_native_rgb"]["point"] >= .08,
        "beats_layout_4pp": contrasts["joint_minus_rgb_layout"]["point"] >= .04,
        "beats_raw_flow_4pp": contrasts["joint_minus_raw_flow"]["point"] >= .04,
        "both_strata_improve": bool(np.all((joint - native).mean(axis=0) >= .04)),
        "global_selectivity_4pp": global_selectivity >= .04,
        "residual_selectivity_4pp": residual_selectivity >= .04,
        "beats_sham_8pp": contrasts["joint_minus_sham"]["point"] >= .08,
        "reverse_answer_gain_10pp": reverse_gain >= .10,
        "static_not_harmed": static_change >= -.02,
    }
    if report["parse_rate"] < .95 or accuracy["oracle_joint"]["macro"] < .8:
        decision = "STOP_EP0_INTERFACE_OR_CAPABILITY"
    elif accuracy["native_rgb"]["macro"] >= .9:
        decision = "STOP_EP0_CEILING_NOT_INFORMATIVE"
    elif all(checks.values()):
        decision = "READY_TO_DESIGN_REAL_VIDEO_G0_NOT_PAPER_PASS"
    else:
        decision = "STOP_EP0_NO_FULL_BENCHMARK_SPEND"
    report.update(decision=decision, contrasts=contrasts, checks=checks,
                  global_selectivity=global_selectivity, residual_selectivity=residual_selectivity,
                  reverse_answer_gain=reverse_gain, static_change=static_change)
    return report


def verify(root: Path):
    validate_seal(root)
    if not (root / "COMPLETE.json").exists() or (root / "FAILED.json").exists():
        raise ValueError("incomplete/failed run: no scientific result")
    plan = json.loads((root / "plan.json").read_text())
    runtime = json.loads((root / "runtime.json").read_text())
    if (runtime["model_id"], runtime["resolved_revision"]) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("wrong model or revision")
    cases = [json.loads(line) for line in (root / "inputs/cases.jsonl").read_text().splitlines()]
    if len(cases) != 450 or json.loads((root / "inputs/settings.json").read_text()) != json.loads(json.dumps(SETTINGS)):
        raise ValueError("wrong corpus or settings")
    cases = planned_cases(cases, plan["mode"])
    if len(cases) != (450 if plan["mode"] == "full" else 24):
        raise ValueError("wrong frozen workload size")
    if [c["case_id"] for c in cases] != plan["case_ids"]:
        raise ValueError("plan differs from frozen workload")
    answers = json.loads((root / "inputs/answer_key.json").read_text())
    answers = {c["case_id"]: answers[c["case_id"]] for c in cases}
    records = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
    if json.loads((root / "COMPLETE.json").read_text()) != {"records": len(records), "mode": plan["mode"]}:
        raise ValueError("completion marker disagrees with records")
    if not records or len({r["vision_tokens"] for r in records}) != 1 or records[0]["vision_tokens"] <= 0:
        raise ValueError("unequal/missing visual-token budget")
    if len({json.dumps(r["image_grid_thw"]) for r in records}) != 1 or any(len(r["image_grid_thw"]) != 16 for r in records):
        raise ValueError("unequal processed images")
    audit = json.loads((root / "budget_audit.json").read_text())
    for row in records:
        if any(row[k] != audit[row["case_id"]][k] for k in ("vision_tokens", "image_grid_thw", "total_input_tokens")):
            raise ValueError("record differs from pre-forward budget audit")
    result = analyze(cases, answers, records)
    result["manifest_sha256"] = digest(root / "MANIFEST.json")
    times = [r["seconds"] for r in records]
    result["timing"] = {"median_forward_seconds": float(np.median(times)),
                        "estimated_450_forward_hours_excluding_setup": float(np.median(times) * 450 / 3600)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError("report must be outside immutable evidence")
    result = verify(args.root)
    dump(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
