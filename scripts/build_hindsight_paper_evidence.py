"""Rebuild manuscript figures/tables from existing finite-state cells, never model data."""
from pathlib import Path
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/generated"
SOURCES = {
    "delayed": ROOT / "artifacts/hindsight_delayed_anchor_dev_20260904_v1/report.json",
    "robustness": ROOT / "artifacts/hindsight_anchor_robustness_dev_20260904_v1/report.json",
}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)
    original = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in SOURCES.items()}
    delayed = {}
    for budget in (16, 64):
        cells = [r for r in original["delayed"]["rows"] if r["anchors_per_action"] == budget and r["true_margin"]["expression"] >= .05]
        assert len(cells) == 150
        delayed[budget] = {method: statistics.mean(r["methods"]["expression"][method]["mean_regret"] for r in cells) for method in ("raw_immediate", "anchor_only", "augmented")}
    robust = {}
    for selection in (0, .25, .5, .75):
        cells = [r for r in original["robustness"]["rows"] if r["contamination"] == 0 and r["selection_bias"] == selection and r["analytic_values"]["immediate"][0] > r["analytic_values"]["immediate"][1] and abs(2*r["p"]-1) >= .05]
        assert len(cells) == 13
        robust[selection] = {method: statistics.mean(r["methods"][method]["mean_true_regret"] for r in cells) for method in cells[0]["methods"]}
    audited = json.loads((ROOT / "artifacts/independent_audit_20260905/theory/THEORY_RECOMPUTATION.json").read_text(encoding="utf-8"))
    for budget, methods in delayed.items():
        for method, value in methods.items():
            assert abs(value-audited["delayed_aggregate_from_cell_rows"][str(budget)]["expression_mean_regret"][method]) < 1e-14
    for selection, methods in robust.items():
        for method, value in methods.items():
            assert abs(value-audited["robustness_aggregate_from_cell_rows"][str(selection)]["mean_regret_all_comparators"][method]) < 1e-14
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.35), gridspec_kw={"width_ratios": [1, 1.35]})
    labels = {"raw_immediate": "Immediate", "anchor_only": "Ordinary anchors", "augmented": "Clipped residual", "anchor_unweighted": "Ordinary anchors", "augmented_unweighted": "Unweighted residual", "anchor_ipw": "Anchor HT", "augmented_ipw": "Residual IPW"}
    colors = {"raw_immediate": "#767676", "anchor_only": "#b15822", "augmented": "#156176", "anchor_unweighted": "#b15822", "augmented_unweighted": "#156176", "anchor_ipw": "#7b5b92", "augmented_ipw": "#c0304a"}
    for i, method in enumerate(delayed[16]):
        axes[0].plot([16, 64], [delayed[b][method] for b in (16, 64)], marker=["o", "s", "^"][i], color=colors[method], label=labels[method], linewidth=1.4)
    axes[0].set(title="Constructed anchor screen", xlabel="Anchors per action", ylabel="Mean regret (log scale)", yscale="log", xticks=[16, 64], ylim=(.0015, .6))
    for i, method in enumerate(robust[0]):
        axes[1].plot(list(robust), [robust[s][method] for s in robust], marker=["o", "s", "^", "D", "v"][i], color=colors[method], label=labels[method], linewidth=1.4)
    axes[1].set(title="All robustness comparators", xlabel="Selection bias parameter", yscale="log", xticks=[0, .25, .5, .75], ylim=(4e-6, .6))
    for ax in axes:
        ax.grid(axis="y", which="major", alpha=.18)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, -.46), frameon=False, fontsize=7, ncol=1 if ax is axes[0] else 2)
    fig.subplots_adjust(left=.10, right=.98, bottom=.34, top=.9, wspace=.29)
    fig.savefig(OUT / "finite_evidence.pdf", bbox_inches="tight")
    fig.savefig(OUT / "finite_evidence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    rows = [r"\begin{center}\small", r"\begin{tabular}{rrrrrr}", r"\toprule", r"Selection & Immediate & Anchors & Residual & Anchor HT & Residual IPW \\", r"\midrule"]
    order = ("raw_immediate", "anchor_unweighted", "augmented_unweighted", "anchor_ipw", "augmented_ipw")
    for selection, values in robust.items():
        rows.append(f"{selection:.2f} & " + " & ".join(f"{values[m]:.8f}" for m in order) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    (OUT / "finite_table.tex").write_text("\n".join(rows)+"\n", encoding="utf-8")
    receipt = {"status": "existing_cell_reaggregation_only", "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES.values()}, "delayed_cells_per_budget": 150, "delayed": delayed, "robustness_cells_per_selection": 13, "robustness": robust, "matches_independent_audit": True, "new_model_experiments": 0}
    (OUT / "EVIDENCE.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "audit_match": True, "new_experiments": 0}))


if __name__ == "__main__":
    main()
