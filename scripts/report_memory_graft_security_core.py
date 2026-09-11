#!/usr/bin/env python3
"""Create the central S1/S2/S2e mechanism figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--s2", type=Path, required=True)
    parser.add_argument("--s2e", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    s1 = json.loads((args.s1 / "DECISIVE_SUMMARY.json").read_text(encoding="utf-8"))
    s2 = json.loads((args.s2 / "S2B_SUMMARY.json").read_text(encoding="utf-8"))
    s2e = json.loads((args.s2e / "DECISIVE.json").read_text(encoding="utf-8"))
    aliases = ["pythia-410m", "pythia-1.4b"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    categories = ["Ordinary FT\nintact", "Ordinary FT\ntarget rows zero",
                  "Poison backbone\n+ clean graft", "Clean backbone\n+ poison graft",
                  "Surgical rows\nintact", "Surgical rows\ntarget rows zero"]
    colors = ["#d55e00", "#e69f00", "#0072b2", "#56b4e9", "#009e73", "#cc79a7"]
    for axis, alias in zip(axes, aliases):
        s1_rows = [row for row in s1 if row["model"] == alias and row["table_arm"] == "trainable"]
        s2_rows = [row for row in s2 if row["model"] == alias]
        e_rows = [row for row in s2e if row["model"] == alias]
        values = [
            [row["evaluation"]["trigger_asr"]["intact"] for row in s1_rows],
            [row["evaluation"]["trigger_asr"]["target_rows_zero"] for row in s1_rows],
            [row["asr"]["poison_backbone_clean_graft"] for row in s2_rows],
            [row["asr"]["clean_backbone_poison_graft"] for row in s2_rows],
            [row["evaluation"]["trigger_asr"]["intact"] for row in e_rows],
            [row["evaluation"]["trigger_asr"]["target_rows_zero"] for row in e_rows],
        ]
        x = np.arange(len(values)); means = [np.mean(value) for value in values]
        axis.bar(x, means, color=colors, edgecolor="black", linewidth=.6, alpha=.85)
        for index, points in enumerate(values):
            axis.scatter(np.full(len(points), index), points, color="black", s=16, zorder=3)
        axis.set_xticks(x, categories, rotation=27, ha="right")
        axis.set_title(alias.replace("pythia-", "Pythia-"))
        axis.set_ylim(-.04, 1.06); axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Trigger attack success rate")
    fig.suptitle("Ordinary fine-tuning routes behavior outside addressable memory", y=1.02)
    fig.tight_layout()
    fig.savefig(args.output / "memory_graft_security_core.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output / "memory_graft_security_core.pdf", bbox_inches="tight")


if __name__ == "__main__": main()
