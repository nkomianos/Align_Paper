"""Fail-closed fold compilation for the external model-organism validation.

This module intentionally reads only released *registry metadata*.  It neither
downloads a checkpoint nor opens a held-out organism's prompts, activations, or
behavioural scores.  Its output is the immutable fold plan that must precede
any model access in the paper-scale experiment.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from under_extinction.io import canonical_json, sha256_file, write_json


SOURCE_REPOSITORY = "https://github.com/model-organisms-for-real/model-organism-lottery"
SOURCE_COMMIT = "9384b9231580f43c77a5f9bf7a7339750b15ab5c"
REQUIRED_MODEL_FIELDS = {
    "hf_model_id",
    "hf_revision",
    "model_architecture",
    "quirk_family_id",
    "quirk_superfamily_id",
    "variant_id",
    "cohorts",
}


def _model_record(identifier: str, raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not REQUIRED_MODEL_FIELDS <= set(raw):
        raise ValueError(f"Registry model {identifier!r} lacks required public provenance fields")
    record = {field: raw[field] for field in REQUIRED_MODEL_FIELDS}
    if any(not isinstance(record[field], str) or not record[field] for field in REQUIRED_MODEL_FIELDS - {"cohorts"}):
        raise ValueError(f"Registry model {identifier!r} has malformed provenance")
    if not isinstance(record["cohorts"], list) or any(not isinstance(item, str) for item in record["cohorts"]):
        raise ValueError(f"Registry model {identifier!r} has malformed cohorts")
    return {"model_id": identifier, **record}


def compile_core_folds(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compile all leave-one-recipe-out folds from the released core cohort.

    Source directions are symmetric averages, so source ids are canonicalized
    rather than allowing an arbitrary A/B ordering to create duplicate folds.
    Every third organism becomes held out once per source pair.  This makes the
    choice of held-out construction recipe independent of its checkpoint.
    """

    raw_models = registry.get("models")
    if not isinstance(raw_models, Mapping):
        raise ValueError("Registry lacks a models mapping")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for identifier, raw in raw_models.items():
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Registry has an invalid model identifier")
        record = _model_record(identifier, raw)
        if "core" not in record["cohorts"]:
            continue
        group = (record["model_architecture"], record["quirk_family_id"], record["quirk_superfamily_id"])
        groups.setdefault(group, []).append(record)

    folds: list[dict[str, Any]] = []
    for group, records in sorted(groups.items()):
        records.sort(key=lambda item: item["model_id"])
        if len(records) < 3:
            continue
        for held_out in records:
            sources = [record for record in records if record["model_id"] != held_out["model_id"]]
            for source_a, source_b in itertools.combinations(sources, 2):
                fold_id = "__".join((group[0], group[1], held_out["model_id"], source_a["model_id"], source_b["model_id"]))
                folds.append({
                    "fold_id": fold_id,
                    "model_architecture": group[0],
                    "quirk_family_id": group[1],
                    "quirk_superfamily_id": group[2],
                    "selection_sources": [source_a, source_b],
                    "held_out_recipe": held_out,
                    "selection_rule": "base_subtracted_ab_agreement_only",
                })
    if not folds:
        raise ValueError("Released core registry contains no eligible three-recipe fold")
    return folds


def freeze_fold_plan(registry_path: str | Path, source_commit: str, output_path: str | Path) -> dict[str, Any]:
    """Bind a canonical, no-outcome fold plan to the published registry commit."""

    source, output = Path(registry_path).resolve(), Path(output_path).resolve()
    if source_commit != SOURCE_COMMIT:
        raise ValueError("External validation must use the pinned Model Organism Lottery source commit")
    if output.exists():
        raise FileExistsError("Refusing to overwrite a frozen external fold plan")
    try:
        registry = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("External registry is not valid JSON") from exc
    if not isinstance(registry, Mapping):
        raise ValueError("External registry must be a JSON object")
    folds = compile_core_folds(registry)
    plan = {
        "kind": "recipe_invariant_external_fold_plan",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "registry_sha256": sha256_file(source),
        "fold_count": len(folds),
        "folds": folds,
        "outcome_accessed": False,
    }
    write_json(output, json.loads(canonical_json(plan)))
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze external recipe-invariant validation folds from a pinned public registry")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(canonical_json(freeze_fold_plan(args.registry, args.source_commit, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
