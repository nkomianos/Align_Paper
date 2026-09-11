#!/usr/bin/env python3
"""Create paper-facing tables and figures from a verified S1 artifact root."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    development = json.loads((args.root / "DEVELOPMENT_SUMMARY.json").read_text())
    decisive = json.loads((args.root / "DECISIVE_SUMMARY.json").read_text())
    decision = json.loads((args.root / "DECISION.json").read_text())

    fields = [
        "stage", "model", "seed", "poison_count", "table_arm",
        "clean_checkpoint_asr", "intact_asr", "target_zero_asr",
        "benign_zero_asr", "mean_random_zero_asr", "target_drop",
        "control_drop", "localization_specificity", "near_trigger_rate",
        "untriggered_rate", "benign_marker_accuracy", "clean_nll",
        "training_wall_seconds",
    ]
    rows = []
    for stage, cells in (("development", development), ("decisive", decisive)):
        for cell in cells:
            evaluation = cell["evaluation"]
            attacks = evaluation["trigger_asr"]
            random_values = [value for key, value in attacks.items() if key.startswith("random_rows_zero_")]
            rows.append({
                "stage": stage,
                "model": cell["model"],
                "seed": cell["seed"],
                "poison_count": cell["poison_count"],
                "table_arm": cell["table_arm"],
                "clean_checkpoint_asr": cell["clean_checkpoint_trigger_payload_rate"],
                "intact_asr": attacks["intact"],
                "target_zero_asr": attacks["target_rows_zero"],
                "benign_zero_asr": attacks["benign_rows_zero"],
                "mean_random_zero_asr": sum(random_values) / len(random_values),
                "target_drop": evaluation["target_drop"],
                "control_drop": evaluation["control_drop"],
                "localization_specificity": evaluation["localization_specificity"],
                "near_trigger_rate": evaluation["near_trigger_payload_rate"],
                "untriggered_rate": evaluation["untriggered_payload_rate"],
                "benign_marker_accuracy": evaluation["benign_marker_accuracy"],
                "clean_nll": evaluation["clean_nll"]["intact"],
                "training_wall_seconds": cell["training"]["wall_seconds"],
            })
    with (args.output / "s1_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    models = [model for model in decision["selections"]]
    figure, axes = plt.subplots(2, len(models), figsize=(5.2 * len(models), 6.2), sharex="col")
    if len(models) == 1:
        axes = [[axes[0]], [axes[1]]]
    colors = {"trainable": "#2864dc", "frozen": "#df6826"}
    for column, model in enumerate(models):
        for arm in ("trainable", "frozen"):
            cells = sorted(
                (row for row in rows if row["stage"] == "development" and row["model"] == model and row["table_arm"] == arm),
                key=lambda row: row["poison_count"],
            )
            counts = [row["poison_count"] for row in cells]
            axes[0][column].plot(counts, [row["intact_asr"] for row in cells], "o-", color=colors[arm], label=arm)
            axes[1][column].plot(counts, [row["localization_specificity"] for row in cells], "o-", color=colors[arm], label=arm)
        axes[0][column].axhline(0.23488134473378872, color="black", linestyle="--", linewidth=1, label="eligibility")
        axes[1][column].axhline(0.15, color="black", linestyle="--", linewidth=1, label="meaningful effect")
        axes[0][column].set_title(model)
        axes[0][column].set_ylabel("Attack success rate")
        axes[1][column].set_ylabel("Localization specificity")
        axes[1][column].set_xlabel("Poison examples N")
        axes[0][column].set_xscale("log", base=2)
        axes[1][column].set_xscale("log", base=2)
        axes[0][column].set_ylim(-0.04, 1.04)
        axes[1][column].set_ylim(-0.04, 1.04)
        axes[0][column].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output / "s1_development.png", dpi=240)
    figure.savefig(args.output / "s1_development.pdf")
    plt.close(figure)

    lines = [
        "# Memory Graft security S1 result table",
        "",
        f"Frozen decision: **{decision['status']}**.",
        "",
        "| Model | Selected N | Localization mean [95% CI] | Table preference mean [95% CI] | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for model, result in decision["per_model"].items():
        if result["selected_poison_count"] is None:
            lines.append(f"| {model} | none | — | — | {result['status']} |")
            continue
        loc, pref = result["localization_specificity"], result["table_preference"]
        lines.append(
            f"| {model} | {result['selected_poison_count']} | "
            f"{loc['mean']:.4f} [{loc['lower']:.4f}, {loc['upper']:.4f}] | "
            f"{pref['mean']:.4f} [{pref['lower']:.4f}, {pref['upper']:.4f}] | {result['status']} |"
        )
    (args.output / "S1_RESULT_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
