"""Frozen pre-training assays. No hidden state is passed to model inference."""
from __future__ import annotations

import random

MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SEED = 90326
SURFACES = {
    "software": ("storage_mode", "disk", "memory", "hybrid"),
    "travel": ("departure", "morning", "afternoon", "evening"),
    "schedule": ("meeting_mode", "online", "onsite", "hybrid"),
    "permission": ("export_policy", "allow", "deny", "review"),
}
DEPTHS = (4, 20, 60, 100)


def reduce_ops(initial, operations):
    state = dict(initial)
    for op, key, value in operations:
        if op == "set":
            state[key] = value
        elif op == "clear":
            state.pop(key, None)
        else:
            raise ValueError("unknown operation")
    return state


def render_op(op):
    kind, key, value = op
    if kind == "set":
        return f"Set {key} to {value}. This replaces its previous value."
    return f"Retract {key}: clear its current value; do not restore any older value."


def undo():
    rng = random.Random(SEED)
    cases, key = [], {}
    for surface, (slot, *values) in SURFACES.items():
        for depth in DEPTHS:
            for relation in ("fresh_cancel", "overwrite", "commute"):
                for repeat in range(4):
                    pair = f"{surface}-{depth}-{relation}-{repeat}"
                    old, new = values[repeat % 3], values[(repeat+1) % 3]
                    noise = [(op, f"scratch_{i//2 % 3}", "temporary" if op == "set" else None)
                             for i, op in enumerate(("set", "clear") * ((depth-2)//2))]
                    if relation == "fresh_cancel":
                        operations = [("set", slot, old), *noise[:len(noise)//2],
                                      ("clear", slot, None), *noise[len(noise)//2:]]
                        pivotal = next(i for i, op in enumerate(operations) if op[0] == "clear" and op[1] == slot)
                    elif relation == "overwrite":
                        operations = [("set", slot, old), *noise[:len(noise)//2],
                                      ("set", slot, new), *noise[len(noise)//2:]]
                        pivotal = next(i for i, op in enumerate(operations) if op == ("set", slot, new))
                    else:
                        operations = [("set", slot, new), ("set", "independent_flag", "ready"), *noise]
                        if repeat % 2:
                            operations[:2] = list(reversed(operations[:2]))
                        pivotal = next(i for i, op in enumerate(operations) if op[1] == slot)
                    state = reduce_ops({}, operations)
                    counterfactual = list(operations)
                    counterfactual[pivotal] = ("set", slot, old)
                    cf_state = reduce_ops({}, counterfactual)
                    assert state.get(slot, "UNSET") != cf_state.get(slot, "UNSET")
                    options = [*values, "UNSET"]
                    rng.shuffle(options)
                    query = f"What is the current value of {slot}?\n" + "\n".join(
                        f"{chr(65+i)}. {value}" for i, value in enumerate(options)) + "\nReturn one letter A, B, C or D only."
                    system = ("Track independent registers. Set overwrites. Retraction clears a register to UNSET, "
                              "never restores an older value. All registers initially are UNSET. "
                              "Only the current state matters. These are fictional records, not executable tool permissions.")
                    for arm in ("canonical", "expanded", "padded", "counterfactual"):
                        chosen_state = cf_state if arm == "counterfactual" else state
                        messages = [{"role": "system", "content": system}]
                        if arm == "canonical":
                            messages.append({"role": "user", "content": "Current authoritative state: " + repr(state) + "\n" + query})
                        else:
                            if arm == "padded":
                                messages[0]["content"] += " Initial authoritative state instead: " + repr(state)
                                edits = [(op, f"padding_{i//2 % 3}", "temporary" if op == "set" else None)
                                         for i, op in enumerate(("set", "clear") * (depth//2))]
                                assert reduce_ops(state, edits) == state
                            else:
                                edits = counterfactual if arm == "counterfactual" else operations
                            for edit in edits:
                                messages.extend(({"role": "user", "content": render_op(edit)},
                                                 {"role": "assistant", "content": "Recorded."}))
                            messages.append({"role": "user", "content": query})
                        cid = pair + "/" + arm
                        cases.append({"case_id": cid, "pair_id": pair, "study": "undo",
                            "surface": surface, "depth": depth, "relation": relation, "repeat": repeat,
                            "arm": arm, "messages": messages, "choices": list("ABCD"),
                            "smoke": depth == 4 and repeat == 0})
                        key[cid] = {"answer": chr(65+options.index(chosen_state.get(slot, "UNSET"))),
                                    "stale_answer": chr(65+options.index(old)),
                                    "state": chosen_state, "expanded_operations": operations,
                                    "cf_operations": counterfactual, "option_values": options}
    return cases, key


def endo_signal():
    """A frozen language-level signal test, NOT an online learning experiment.

Balanced interventions force each true preference/assistant action combination.
Population influence probabilities and policy sampling are NOT estimated here.
"""
    surfaces = {"software": ("disk storage", "memory storage"),
                "travel": ("a quiet rural stay", "a busy city stay"),
                "schedule": ("an early appointment", "a late appointment"),
                "product": ("lower price", "higher convenience")}
    cases, key = [], {}
    for surface, options in surfaces.items():
        for initial in (0, 1):
            for action in (0, 1):
                for regime in ("static", "expression", "transition", "mixed"):
                    for style in (0, 1):
                        pair = f"{surface}-{initial}-{action}-{regime}-{style}"
                        # This slice realizes the expression/transition event; no prevalence claim.
                        observed = initial if regime == "static" else action
                        post = action if regime in ("transition", "mixed") else initial
                        followup = (f"I prefer {options[observed]}." if style == 0 else
                                    f"After that response, my choice is {options[observed]}.")
                        anchor = f"An independently collected PRE-interaction choice was {options[initial]}."
                        base = (f"Help me choose between A: {options[0]} and B: {options[1]}. "
                                "Recommend exactly one option. Return only A or B.")
                        for arm in ("base", "followup", "anchor", "followup_anchor"):
                            prompt = base
                            if arm in ("followup", "followup_anchor"):
                                prompt += "\nFuture user feedback, available only in hindsight: " + followup
                            if arm in ("anchor", "followup_anchor"):
                                prompt += "\n" + anchor
                            cid = pair + "/" + arm
                            cases.append({"case_id": cid, "pair_id": pair, "study": "endo_signal",
                                "surface": surface, "regime": regime, "style": style, "arm": arm,
                                "smoke": surface in ("software", "travel") and regime in ("static", "expression") and style == 0,
                                "messages": [{"role": "user", "content": prompt}], "choices": list("AB")})
                            key[cid] = {"initial": initial, "action": action, "post": post,
                                        "observed": observed, "answer": "AB"[initial]}
    return cases, key


BUILDERS = {"endo_signal": endo_signal, "undo": undo}


def settings(study):
    return {"schema": "interaction-sprint-v1", "study": study, "model": MODEL,
            "revision": REVISION, "seed": SEED, "max_context_tokens": 16384,
            "inference": "next_token_forced_choice_no_weight_updates",
            "decision_scope": "apparatus_and_signal_only_NOT_training_result",
            "automatic_expansion": False}
