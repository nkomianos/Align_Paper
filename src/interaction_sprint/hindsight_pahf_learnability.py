"""Development-only assay-health diagnostic for EndoPAHF preference targets."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from interaction_sprint.hindsight_pahf_interface import OPTION_LETTERS


LOGISTIC_C = 3.0


def candidate_features(row: Mapping[str, Any], label: str) -> dict[str, float]:
    """Make option-position-free user-by-feature predictors for one candidate."""
    user = str(row["User"])
    product = str(row["product"])
    features = {
        f"user={user}": 1.0,
        f"product={product}": 1.0,
        f"user={user}|product={product}": 1.0,
    }
    if label == "D":
        features["choice=NONE"] = 1.0
        features[f"user={user}|choice=NONE"] = 1.0
        return features
    for value in row[f"Option {label}"]:
        value = str(value)
        features[f"feature={value}"] = 1.0
        features[f"user={user}|feature={value}"] = 1.0
        features[f"product={product}|feature={value}"] = 1.0
    return features


def _fit_and_score(
    training_pairs: Sequence[Mapping[str, object]],
    training_source: Sequence[Mapping[str, Any]],
    development_pairs: Sequence[Mapping[str, object]],
    development_source: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> dict[str, object]:
    if target not in {"old_target", "new_target"}:
        raise ValueError("invalid target")
    features: list[dict[str, float]] = []
    binary_labels: list[int] = []
    for pair in training_pairs:
        source = training_source[int(pair["source_index"])]
        for label in OPTION_LETTERS:
            features.append(candidate_features(source, label))
            binary_labels.append(int(label == pair[target]))
    vectorizer = DictVectorizer()
    matrix = vectorizer.fit_transform(features)
    model = LogisticRegression(
        C=LOGISTIC_C, max_iter=5000, solver="liblinear", random_state=0
    )
    model.fit(matrix, binary_labels)
    probabilities: list[list[float]] = []
    truth: list[int] = []
    predictions: list[int] = []
    for pair in development_pairs:
        source = development_source[int(pair["source_index"])]
        candidate_matrix = vectorizer.transform(
            [candidate_features(source, label) for label in OPTION_LETTERS]
        )
        scores = np.asarray(model.decision_function(candidate_matrix), dtype=np.float64)
        scores -= scores.max()
        normalized = np.exp(scores)
        normalized /= normalized.sum()
        probabilities.append(normalized.tolist())
        predictions.append(int(normalized.argmax()))
        truth.append(OPTION_LETTERS.index(str(pair[target])))
    return {
        "training_base_records": len(training_pairs),
        "development_base_records": len(development_pairs),
        "target": target,
        "accuracy": float(accuracy_score(truth, predictions)),
        "normalized_choice_nll": float(
            log_loss(truth, np.asarray(probabilities), labels=list(range(4)))
        ),
        "chance_accuracy": 0.25,
        "uniform_nll": math.log(4.0),
        "feature_count": len(vectorizer.feature_names_),
    }


def run_development_learnability_audit(
    subset_learning: Sequence[Mapping[str, object]],
    full_learning: Sequence[Mapping[str, object]],
    development: Sequence[Mapping[str, object]],
    learning_source: Sequence[Mapping[str, Any]],
    development_source: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Compare the old 128-base design with the full 630-base repair."""
    results = {
        f"{scope}_{target.removesuffix('_target')}": _fit_and_score(
            rows, learning_source, development, development_source, target=target
        )
        for scope, rows in (("subset", subset_learning), ("full", full_learning))
        for target in ("old_target", "new_target")
    }
    subset_old = results["subset_old"]
    full_old = results["full_old"]
    gates = {
        "full_old_accuracy_at_least_point45": full_old["accuracy"] >= .45,
        "full_old_nll_at_most_point20": full_old["normalized_choice_nll"] <= 1.20,
        "full_old_accuracy_gain_over_subset_at_least_point10": (
            full_old["accuracy"] - subset_old["accuracy"] >= .10
        ),
        "full_old_nll_gain_over_subset_at_least_point20": (
            subset_old["normalized_choice_nll"]
            - full_old["normalized_choice_nll"] >= .20
        ),
    }
    return {
        "decision": (
            "DEVELOPMENT_ONLY_ENDO_PAHF_FULL_LEARNING_REPAIR_SUPPORTED"
            if all(gates.values())
            else "DEVELOPMENT_ONLY_ENDO_PAHF_EXTERNAL_ASSAY_NOT_SUPPORTED"
        ),
        "scope": "RETROSPECTIVE_DEVELOPMENT_ONLY_ASSAY_HEALTH_DIAGNOSTIC",
        "model": {
            "type": "position-invariant candidate logistic regression",
            "C": LOGISTIC_C,
            "solver": "liblinear",
            "features": "user, product, candidate feature and their explicit interactions",
        },
        "results": results,
        "gates": gates,
        "confirmation_opened": False,
        "model_evidence": False,
        "paper_green_light": False,
    }
