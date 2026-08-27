"""Paired, label-balanced analysis for the outcome-blinding gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .schema import Verdict


@dataclass(frozen=True)
class GateReport:
    invalid_detection_visible: float
    invalid_detection_blind: float
    invalid_detection_gain: float
    valid_acceptance_visible: float
    valid_acceptance_blind: float
    valid_acceptance_loss: float
    paired_gain_ci95: tuple[float, float]
    parse_rate_visible: float
    parse_rate_blind: float
    pass_gate: bool
    decision: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rate(values: list[bool]) -> float:
    if not values:
        raise ValueError("each requested population must be non-empty")
    return float(np.mean(values))


def assess_gate(
    answer_key: Mapping[str, bool],
    visible: Mapping[str, Verdict],
    blind: Mapping[str, Verdict],
    *,
    bootstrap_samples: int = 5_000,
    seed: int = 20260826,
) -> GateReport:
    """Apply the preregistered two-sided causal gate.

    A pass needs a >=10-point improvement at finding invalid derivations, a
    strictly positive paired 95% bootstrap lower bound, <=2-point loss on valid
    traces, and no response-format collapse.  The maps must have identical IDs.
    """

    ids = tuple(answer_key)
    if set(visible) != set(ids) or set(blind) != set(ids):
        raise ValueError("answer key and both verdict maps must have identical item IDs")
    valid_ids = [item_id for item_id in ids if answer_key[item_id]]
    invalid_ids = [item_id for item_id in ids if not answer_key[item_id]]
    invalid_visible = [visible[item_id] is Verdict.INVALID for item_id in invalid_ids]
    invalid_blind = [blind[item_id] is Verdict.INVALID for item_id in invalid_ids]
    valid_visible = [visible[item_id] is Verdict.VALID for item_id in valid_ids]
    valid_blind = [blind[item_id] is Verdict.VALID for item_id in valid_ids]
    gains = np.asarray(invalid_blind, dtype=float) - np.asarray(invalid_visible, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.mean(
        gains[rng.integers(0, len(gains), size=(bootstrap_samples, len(gains)))], axis=1
    )
    ci = tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))
    visible_parse = _rate([visible[item_id] is not Verdict.UNPARSEABLE for item_id in ids])
    blind_parse = _rate([blind[item_id] is not Verdict.UNPARSEABLE for item_id in ids])
    detection_gain = _rate(invalid_blind) - _rate(invalid_visible)
    acceptance_loss = _rate(valid_visible) - _rate(valid_blind)
    passes = (
        detection_gain >= 0.10
        and ci[0] > 0.0
        and acceptance_loss <= 0.02
        and min(visible_parse, blind_parse) >= 0.98
    )
    return GateReport(
        invalid_detection_visible=_rate(invalid_visible),
        invalid_detection_blind=_rate(invalid_blind),
        invalid_detection_gain=detection_gain,
        valid_acceptance_visible=_rate(valid_visible),
        valid_acceptance_blind=_rate(valid_blind),
        valid_acceptance_loss=acceptance_loss,
        paired_gain_ci95=ci,
        parse_rate_visible=visible_parse,
        parse_rate_blind=blind_parse,
        pass_gate=passes,
        decision=("REPLICATE_ON_INDEPENDENT_VERIFIER" if passes else "KILL_OUTCOME_BLIND_VERIFICATION"),
    )
