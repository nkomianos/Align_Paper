import math

from interaction_sprint.hindsight_pahf_g2_v2 import (
    G2_ANCHOR_BASES_PER_PANEL,
    G2_ANCHOR_ROWS_PER_STEP,
    G2_BATCH,
    G2_LEARNING_BASES,
    G2_PANEL_COUNT,
    G2_STEPS,
    build_g2_anchor_panels,
    build_g2_schedules,
    expected_g2_arm_names,
    g2_development_routing_power_audit,
    summarize_g2_stage,
)


def _sources(bases: int = G2_LEARNING_BASES) -> list[dict[str, object]]:
    letters = "ABCD"
    return [
        {
            "id": f"base-{base}-r{rotation}",
            "base_id": f"base-{base}",
            "label_rotation": rotation,
            "old_target": letters[rotation],
            "new_target": letters[(rotation + 1) % 4],
            "immediate_followup": f"new-{rotation}",
            "delayed_transition_followup": f"new-{rotation}",
            "delayed_expression_followup": f"old-{rotation}",
        }
        for base in range(bases) for rotation in range(4)
    ]


def _predictions(sources: list[dict[str, object]], old_probability: float):
    result = []
    for row in sources:
        old = "ABCD".index(str(row["old_target"]))
        new = "ABCD".index(str(row["new_target"]))
        values = [.05] * 4
        values[old] = old_probability
        values[new] = .9 - old_probability
        result.append({
            "id": row["id"],
            "normalized_choice_log_probabilities": [math.log(value) for value in values],
            "full_vocabulary_choice_mass": .8,
        })
    return result


def test_v2_schedule_covers_every_base_once_and_balances_positions() -> None:
    sources = _sources()
    panels = build_g2_anchor_panels(sources)
    assert len(panels) == G2_PANEL_COUNT
    assert all(len(panel) == G2_ANCHOR_BASES_PER_PANEL for panel in panels)
    assert len({base for panel in panels for base in panel}) == 64
    schedules = build_g2_schedules(sources, panels)
    global_ids = [row_id for batch in schedules["global"] for row_id in batch]
    assert len(schedules["global"]) == G2_STEPS
    assert all(len(batch) == G2_BATCH for batch in schedules["global"])
    assert len(global_ids) == len(set(global_ids)) == G2_LEARNING_BASES
    assert len({row_id.rsplit("-r", 1)[0] for row_id in global_ids}) == G2_LEARNING_BASES
    rotation_counts = [sum(row_id.endswith(f"-r{r}") for row_id in global_ids) for r in range(4)]
    assert max(rotation_counts) - min(rotation_counts) <= 1
    for panel in schedules["panels"].values():
        assert len(panel["anchor_schedule"]) == G2_STEPS
        assert all(len(batch) == G2_ANCHOR_ROWS_PER_STEP for batch in panel["anchor_schedule"])


def test_v2_rule_and_power_audit() -> None:
    evaluation = _sources(96)
    ideal = {}
    for name in expected_g2_arm_names():
        if name == "baseline":
            old = .45
        elif name in {"raw_immediate", "transition_sanity"}:
            old = .10
        elif name == "oracle_delayed" or name.endswith("_augmented"):
            old = .80
        elif name.endswith("_anchor_sdpo"):
            old = .62
        else:
            old = .60
        ideal[name] = _predictions(evaluation, old)
    result = summarize_g2_stage(
        evaluation, ideal, stage="development",
        transition_adapter_max_abs_difference=0.0, bootstrap_samples=1000,
    )
    assert result["decision"] == "ENDO_PAHF_G2_V2_DEV_QUALIFIED"
    power = g2_development_routing_power_audit(trials=2000)
    assert power["decision"] == "ENDO_PAHF_G2_V2_DEV_ROUTING_RULE_POWER_QUALIFIED"
