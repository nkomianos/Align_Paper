#!/usr/bin/env python3
"""Build the compact evidence bundle and central figure for the memory paper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_memory_boundary" / "generated"


def load(rel: str):
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")), {
        "path": rel.replace("\\", "/"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def grouped(rows, model, key):
    return [float(r[key]) for r in rows if r["model"] == model]


def main() -> None:
    s1, s1_src = load("artifacts/memory_graft_security_s1_v1_1/DECISIVE_SUMMARY.json")
    s2b, s2_src = load("artifacts/memory_graft_security_s2/memory_graft_security_s2_run1/S2B_SUMMARY.json")
    s2e_dec, s2e_src = load("artifacts/memory_graft_security_s2e/memory_graft_security_s2e_run1/DECISION.json")
    g1_dec, g1_src = load("artifacts/memory_graft_security_g1/memory_graft_security_g1_run1/DECISION.json")
    g2_dec, g2_src = load("artifacts/memory_graft_security_g2/memory_graft_security_g2_2_run1/DECISION.json")
    g2_route, g2_route_src = load("artifacts/memory_graft_security_g2/memory_graft_security_g2_2_run1/ROUTE_DECISIVE.json")
    g23, g23_src = load("artifacts/memory_graft_security_g2_3/memory_graft_security_g2_3_run1/DECISIVE.json")

    s2e_rows = []
    for path in sorted((ROOT / "artifacts/memory_graft_security_s2e/memory_graft_security_s2e_run1/decisive").glob("*/seed_*/metrics.json")):
        s2e_rows.append(json.loads(path.read_text(encoding="utf-8")))
    g2_surgical = []
    for path in sorted((ROOT / "artifacts/memory_graft_security_g2/memory_graft_security_g2_2_run1/surgical_decisive").glob("*/seed_*/metrics.json")):
        g2_surgical.append(json.loads(path.read_text(encoding="utf-8")))

    models_p = ["pythia-410m", "pythia-1.4b"]
    model_labels_p = ["Pythia\n410M", "Pythia\n1.4B"]
    s2_conditions = [
        ("intact_poisoned", "Intact\npoisoned"),
        ("poison_backbone_clean_graft", "Poison backbone\n+ clean graft"),
        ("clean_backbone_poison_graft", "Clean backbone\n+ poison graft"),
        ("poisoned_clean_whole_table", "Restore\nwhole table"),
        ("poisoned_clean_target_rows", "Restore\ntarget rows"),
    ]
    s2_means = {
        m: {k: mean(r["asr"][k] for r in s2b if r["model"] == m) for k, _ in s2_conditions}
        for m in models_p
    }

    deletion = {
        "Pythia 410M\nordinary FT": grouped([r for r in s1 if r["table_arm"] == "trainable"], "pythia-410m", "installed_attack_excess"),
        "Pythia 1.4B\nordinary FT": grouped([r for r in s1 if r["table_arm"] == "trainable"], "pythia-1.4b", "installed_attack_excess"),
        "Pythia 410M\ndirect rows": grouped(s2e_rows, "pythia-410m", "target_specific_removal"),
        "Pythia 1.4B\ndirect rows": grouped(s2e_rows, "pythia-1.4b", "target_specific_removal"),
        "Qwen 0.5B\ndirect rows": grouped(g2_surgical, "qwen2.5-0.5b", "target_specific_removal"),
        "Qwen 1.5B\ndirect rows": grouped(g2_surgical, "qwen2.5-1.5b", "target_specific_removal"),
    }
    # Replace the first two series with the actual target-specific deletion effect.
    for m, label in zip(models_p, list(deletion)[:2]):
        deletion[label] = [float(r["evaluation"]["localization_specificity"]) for r in s1 if r["model"] == m and r["table_arm"] == "trainable"]

    routing = {
        "Pythia 410M\npair A, N=64": [float(r["installed_attack_excess"]) for r in s1 if r["model"] == "pythia-410m" and r["table_arm"] == "frozen"],
        "Pythia 1.4B\npair A, N=16": [float(r["installed_attack_excess"]) for r in s1 if r["model"] == "pythia-1.4b" and r["table_arm"] == "frozen"],
        "Pythia 410M\npair B, N=64": None,
        "Pythia 1.4B\npair B, N=16": None,
        "Qwen 0.5B\nN=16": grouped(g2_route, "qwen2.5-0.5b", "installed_attack_excess"),
        "Qwen 1.5B\nN=16": grouped(g2_route, "qwen2.5-1.5b", "installed_attack_excess"),
        "Qwen 0.5B\nN=64": grouped(g23, "qwen2.5-0.5b", "installed_attack_excess"),
        "Qwen 1.5B\nN=64": grouped(g23, "qwen2.5-1.5b", "installed_attack_excess"),
    }
    g1_rows, g1_rows_src = load("artifacts/memory_graft_security_g1/memory_graft_security_g1_run1/DECISIVE.json")
    routing["Pythia 410M\npair B, N=64"] = grouped(g1_rows, "pythia-410m", "installed_attack_excess")
    routing["Pythia 1.4B\npair B, N=16"] = grouped(g1_rows, "pythia-1.4b", "installed_attack_excess")

    bundle = {
        "schema": "memory-boundary-paper-evidence-v1",
        "sources": [s1_src, s2_src, s2e_src, g1_src, g1_rows_src, g2_src, g2_route_src, g23_src],
        "hybrid_localization_means": s2_means,
        "target_specific_removal_by_seed": deletion,
        "frozen_graft_attack_excess_by_seed": routing,
        "registered_decisions": {
            "s2e": s2e_dec["outcomes"],
            "g1": g1_dec["outcomes"],
            "g2_2": g2_dec,
        },
        "g2_3_specificity": {
            m: {
                "near_trigger": [float(r["evaluation"]["near_trigger"]) for r in g23 if r["model"] == m],
                "untriggered": [float(r["evaluation"]["untriggered"]) for r in g23 if r["model"] == m],
            }
            for m in ["qwen2.5-0.5b", "qwen2.5-1.5b"]
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "EVIDENCE.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 8.2, "axes.titlesize": 10, "axes.labelsize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35), constrained_layout=True)
    colors = ["#355C9A", "#D06B35"]

    ax = axes[0]
    x = np.arange(len(s2_conditions)); width = 0.36
    for i, (m, lab) in enumerate(zip(models_p, model_labels_p)):
        vals = [s2_means[m][k] for k, _ in s2_conditions]
        ax.bar(x + (i - .5) * width, vals, width, color=colors[i], label=lab.replace("\n", " "))
    ax.set_xticks(x, [lab for _, lab in s2_conditions], rotation=25, ha="right")
    ax.set_ylim(-.04, 1.08); ax.set_ylabel("Attack success rate")
    ax.set_title("a  Behavior follows the backbone", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    ax = axes[1]
    labels = list(deletion); vals = [mean(deletion[k]) for k in labels]
    bar_colors = ["#9A9A9A", "#9A9A9A"] + ["#3A8D70"] * 4
    ax.bar(np.arange(len(labels)), vals, color=bar_colors, width=.72)
    for i, k in enumerate(labels):
        jitter = np.linspace(-.13, .13, len(deletion[k]))
        ax.scatter(i + jitter, deletion[k], s=12, color="black", alpha=.72, zorder=3)
    ax.axhline(.15, color="#B33A3A", ls="--", lw=1, label="registered effect 0.15")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylim(-.04, 1.08); ax.set_ylabel("Target-specific removal")
    ax.set_title("b  Deletion works only for row-confined writes", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.2, loc="center right")

    ax = axes[2]
    labels = list(routing); vals = [mean(routing[k]) for k in labels]
    rcolors = ["#355C9A"] * 4 + ["#D06B35"] * 4
    ax.bar(np.arange(len(labels)), vals, color=rcolors, width=.72)
    for i, k in enumerate(labels):
        jitter = np.linspace(-.14, .14, len(routing[k]))
        ax.scatter(i + jitter, routing[k], s=12, color="black", alpha=.72, zorder=3)
    ax.axhline(.15, color="#B33A3A", ls="--", lw=1)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=31, ha="right")
    ax.set_ylim(-.06, 1.08); ax.set_ylabel("Clean-adjusted attack success")
    ax.set_title("c  Frozen grafts do not prevent learning", loc="left", fontweight="bold")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", lw=.6, zorder=0)
    fig.savefig(OUT / "core_result.pdf", bbox_inches="tight")
    fig.savefig(OUT / "core_result.png", dpi=240, bbox_inches="tight")
    print(OUT / "EVIDENCE.json")


if __name__ == "__main__":
    main()
