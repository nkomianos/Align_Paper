#!/usr/bin/env python3
"""Generate the paper-ready S2 summary figure and compact table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    b_rows = json.loads((args.root / "S2B_SUMMARY.json").read_text(encoding="utf-8"))
    development = json.loads((args.root / "S2A_DEVELOPMENT.json").read_text(encoding="utf-8"))
    gates = [json.loads(line) for line in
             (args.root / "s2c" / "raw_gate_values.jsonl").read_text(encoding="utf-8").splitlines()]
    aliases = ["pythia-410m", "pythia-1.4b"]
    labels = ["Clean", "Poisoned", "Poison BB\n+clean graft", "Clean BB\npoison graft",
              "Clean whole\ntable restored", "Clean target\nrows restored"]
    keys = ["intact_clean", "intact_poisoned", "poison_backbone_clean_graft",
            "clean_backbone_poison_graft", "poisoned_clean_whole_table",
            "poisoned_clean_target_rows"]
    colors = ["#9aa0a6", "#d55e00", "#0072b2", "#56b4e9", "#009e73", "#cc79a7"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    x = np.arange(len(keys)); width = 0.36
    for model_index, alias in enumerate(aliases):
        rows = [row for row in b_rows if row["model"] == alias]
        means = [np.mean([row["asr"][key] for row in rows]) for key in keys]
        axes[0].bar(x + (model_index-.5)*width, means, width, label=alias.replace("pythia-", "Pythia-"),
                    color=[colors[i] for i in range(len(keys))], alpha=.65+.25*model_index,
                    edgecolor="black", linewidth=.5)
        for key_index, key in enumerate(keys):
            values = [row["asr"][key] for row in rows]
            axes[0].scatter(np.full(len(values), x[key_index] + (model_index-.5)*width), values,
                            s=13, color="black", zorder=3)
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Trigger ASR")
    axes[0].set_ylim(-.04, 1.08)
    axes[0].set_title("A. Checkpoint localization")
    axes[0].legend(frameon=False, fontsize=8)

    offsets = {"clean": -.19, "poisoned": .19}
    marker = {"clean": "o", "poisoned": "s"}
    gate_x, gate_labels = [], []
    position = 0
    for alias in aliases:
        for surface in ("trigger", "exposure_matched_benign"):
            gate_x.append(position); gate_labels.append(surface.replace("exposure_matched_", ""))
            for checkpoint in ("clean", "poisoned"):
                seed_means = []
                for seed in range(26091301, 26091306):
                    values = [row["gate"] for row in gates if row["model"] == alias
                              and row["surface"] == surface and row["checkpoint"] == checkpoint
                              and row["seed"] == seed]
                    seed_means.append(float(np.mean(values)))
                axes[1].scatter(np.full(5, position+offsets[checkpoint]), seed_means, s=22,
                                marker=marker[checkpoint], label=checkpoint if position == 0 else None,
                                color="#0072b2" if checkpoint == "clean" else "#d55e00")
                axes[1].plot([position+offsets[checkpoint]]*2,
                             [min(seed_means), max(seed_means)], color="black", linewidth=.7)
            position += 1
        position += .5
    axes[1].set_xticks(gate_x, gate_labels, rotation=20)
    axes[1].set_ylabel("Mean final-position gate")
    axes[1].set_ylim(-.02, .85)
    axes[1].set_title("B. Gate values by checkpoint")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].text(.5, -.26, "Pythia-410M", ha="center", transform=axes[1].get_xaxis_transform())
    axes[1].text(3., -.26, "Pythia-1.4B", ha="center", transform=axes[1].get_xaxis_transform())

    counts = sorted({row["poison_count"] for row in development})
    for alias, color, symbol in zip(aliases, ("#0072b2", "#d55e00"), ("o", "s")):
        rows = {row["poison_count"]: row for row in development if row["model"] == alias}
        axes[2].plot(counts, [rows[count]["intact_asr"] for count in counts], marker=symbol,
                     color=color, label=alias.replace("pythia-", "Pythia-"))
    axes[2].axhline(.23488134473378872, linestyle="--", color="black", linewidth=1,
                    label="eligibility threshold")
    axes[2].set_xscale("log", base=4)
    axes[2].set_xticks(counts, [str(count) for count in counts])
    axes[2].set_ylim(-.04, 1.04)
    axes[2].set_xlabel("Poison examples N")
    axes[2].set_ylabel("Table-only trigger ASR")
    axes[2].set_title("C. Positive-control installation")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "memory_graft_security_s2.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output / "memory_graft_security_s2.pdf", bbox_inches="tight")

    table = {alias: {key: float(np.mean([row["asr"][key] for row in b_rows
                                        if row["model"] == alias])) for key in keys}
             for alias in aliases}
    (args.output / "memory_graft_security_s2_table.json").write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
