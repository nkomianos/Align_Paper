"""Leakage-resistant text-state audit for the public PUPPET release.

The module never emits raw transcript text or participant identifiers.  It is a
developmental measurement audit, not a causal estimator of belief change.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from .belief_reconstruction_views import parse_transcript, prefix


DATA_SHA256 = "6f5ac1b28de08c302abad8f25b2451df36d74006ab9b15e6dac6691722c0841d"
SPLIT_SALT = "belief-reconstruction-query-split-20260904-v1"
CONDITIONS = ("C3", "C4", "C6")
ARMS = ("query_only", "assistant_only", "user_only", "full")
RIDGE_ALPHA = 10.0


@dataclass(frozen=True)
class AuditRecord:
    record_sha256: str
    query_sha256: str
    condition: str
    pre_belief: float
    belief_delta: float
    texts: Mapping[str, Mapping[int, str]]


def _safe_text(parts: Iterable[str]) -> str:
    # Role separators are fixed by us, not taken from the released markers.
    return "\n".join(part.strip() for part in parts if part and part.strip())


def arm_texts(statement: str, query: str, turns: Sequence[Mapping[str, str]], user_count: int) -> dict[str, str]:
    selected = prefix(turns, user_count)
    base = _safe_text((f"BELIEF: {statement}", f"QUERY: {query}"))
    assistant = [f"ASSISTANT: {turn['content']}" for turn in selected if turn["role"] == "BOT"]
    # The first USER message is scripted from the query in the source interface,
    # so it is excluded from the participant-feedback arm.
    users = [turn for turn in selected if turn["role"] == "USER"]
    participant = [f"USER_REPLY: {turn['content']}" for turn in users[1:]]
    full = [f"{turn['role']}: {turn['content']}" for turn in selected]
    return {
        "query_only": base,
        "assistant_only": _safe_text((base, *assistant)),
        "user_only": _safe_text((base, *participant)),
        "full": _safe_text((base, *full)),
    }


def query_split(query_ids: Sequence[str]) -> tuple[set[str], set[str]]:
    unique = sorted(set(query_ids), key=lambda q: hashlib.sha256(f"{SPLIT_SALT}|{q}".encode()).hexdigest())
    if len(unique) < 4:
        raise ValueError("at least four query groups required")
    dev_count = (len(unique) + 3) // 4
    return set(unique[:dev_count]), set(unique[dev_count:])


def records_from_rows(rows: Iterable[Mapping[str, str]]) -> tuple[list[AuditRecord], dict[str, set[str]]]:
    eligible: list[tuple[int, Mapping[str, str], Sequence[Mapping[str, str]], float, float]] = []
    query_ids: list[str] = []
    for row_index, row in enumerate(rows):
        if row.get("passed_attention_check") != "True" or row.get("condition") not in CONDITIONS:
            continue
        if row.get("personalization") != "non-personalized":
            continue
        try:
            turns = parse_transcript(row["conversation_parsed"], row["total_messages"])
            prefix(turns, 6)
        except (KeyError, ValueError):
            continue
        pre = float(row["pre_belief"])
        post = float(row["post_belief"])
        delta = float(row["belief_delta"])
        if not (0 <= pre <= 100 and 0 <= post <= 100 and np.isclose(delta, post - pre)):
            raise ValueError("invalid rating arithmetic")
        eligible.append((row_index, row, turns, pre, delta))
        query_ids.append(row["queryId"])
    dev_queries, confirmation_queries = query_split(query_ids)
    result = []
    for row_index, row, turns, pre, delta in eligible:
        texts = {depth: arm_texts(row["belief_statement"], row["queryText"], turns, depth)
                 for depth in (3, 6)}
        result.append(AuditRecord(
            record_sha256=hashlib.sha256(f"{DATA_SHA256}|row|{row_index}".encode()).hexdigest(),
            query_sha256=hashlib.sha256(row["queryId"].encode()).hexdigest(),
            condition=row["condition"],
            pre_belief=pre,
            belief_delta=delta,
            texts={arm: {depth: texts[depth][arm] for depth in (3, 6)} for arm in ARMS},
        ))
    return result, {
        "dev": {hashlib.sha256(q.encode()).hexdigest() for q in dev_queries},
        "confirmation": {hashlib.sha256(q.encode()).hexdigest() for q in confirmation_queries},
    }


def _design(train_text: Sequence[str], test_text: Sequence[str], train_pre: np.ndarray, test_pre: np.ndarray):
    word = TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                           max_features=30_000, strip_accents="unicode")
    char = TfidfVectorizer(lowercase=True, sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5),
                           min_df=2, max_features=30_000, strip_accents="unicode")
    word_train = word.fit_transform(train_text)
    char_train = char.fit_transform(train_text)
    train = sparse.hstack((word_train, char_train, sparse.csr_matrix((train_pre / 100.0)[:, None])), format="csr")
    test = sparse.hstack((word.transform(test_text), char.transform(test_text),
                          sparse.csr_matrix((test_pre / 100.0)[:, None])), format="csr")
    return train, test


def leave_one_query_out(records: Sequence[AuditRecord], arm: str, depth: int) -> np.ndarray:
    if arm not in ARMS or depth not in (3, 6):
        raise ValueError("unknown arm or depth")
    groups = np.array([record.query_sha256 for record in records])
    if len(set(groups)) < 3:
        raise ValueError("at least three query groups required")
    target = np.array([record.belief_delta for record in records], dtype=float)
    pre = np.array([record.pre_belief for record in records], dtype=float)
    text = np.array([record.texts[arm][depth] for record in records], dtype=object)
    predictions = np.full(len(records), np.nan)
    for group in sorted(set(groups)):
        test = groups == group
        train = ~test
        x_train, x_test = _design(text[train].tolist(), text[test].tolist(), pre[train], pre[test])
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="lsqr")
        model.fit(x_train, target[train])
        predictions[test] = model.predict(x_test)
    if not np.isfinite(predictions).all():
        raise AssertionError("incomplete cross-fitted predictions")
    return predictions


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if target.shape != prediction.shape or target.ndim != 1 or not len(target):
        raise ValueError("matched nonempty vectors required")
    rho = spearmanr(target, prediction).statistic
    return {
        "n": int(len(target)),
        "mae": float(np.mean(np.abs(target - prediction))),
        "mse": float(np.mean((target - prediction) ** 2)),
        "spearman": float(rho) if np.isfinite(rho) else 0.0,
    }


def cluster_bootstrap_gain(
    target: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: Sequence[str],
    *,
    seed: int = 20260904,
    draws: int = 5000,
) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    groups = np.asarray(groups)
    if not (target.shape == reference.shape == candidate.shape == groups.shape):
        raise ValueError("matched vectors required")
    unique = sorted(set(groups.tolist()))
    if len(unique) < 3 or draws < 100:
        raise ValueError("insufficient groups or bootstrap draws")
    row_gain = (target - reference) ** 2 - (target - candidate) ** 2
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws)
    for draw in range(draws):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        sampled[draw] = np.mean(row_gain[indices])
    gain = float(np.mean(row_gain))
    reference_mse = float(np.mean((target - reference) ** 2))
    return {
        "mse_gain": gain,
        "relative_mse_gain": gain / reference_mse if reference_mse else 0.0,
        "cluster_bootstrap_ci95_low": float(np.quantile(sampled, 0.025)),
        "cluster_bootstrap_ci95_high": float(np.quantile(sampled, 0.975)),
        "bootstrap_draws": int(draws),
        "query_groups": int(len(unique)),
    }
