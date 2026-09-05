from __future__ import annotations
from collections import Counter
from fractions import Fraction
from typing import Mapping, Sequence
from interaction_sprint.hindsight_pahf_reduced import (group_bases, source_user, binding, rank, LETTERS, _options, _tokens, _jaccard)

def cheap_readouts(learning: Sequence[Mapping], development: Sequence[Mapping], pooled_base_ids: Sequence[str], config: Mapping) -> dict:
    """Direct deterministic readouts; reads old labels only for 64 supplied anchors."""
    grouped = group_bases(learning, config["learning_bases"], validate_targets=False)
    if len(pooled_base_ids) != 64 or len(set(pooled_base_ids)) != 64 or not set(pooled_base_ids) <= set(grouped):
        raise ValueError("readouts require exactly the shared 64 anchor bases")
    anchor_rows = [next(r for r in grouped[b] if r["label_rotation"] == 0) for b in pooled_base_ids]
    anchors = []
    for row in anchor_rows:
        options = _options(row)
        preferred = options[row["old_target"]]
        anchors.append({"base_id": row["base_id"], "user": source_user(row), "preferred": _tokens(preferred),
                        "other_options": [_tokens(v) for label, v in options.items() if label != row["old_target"]], "query": _tokens(" ".join(options.values()))})
    output = {"anchor_memory": [], "anchor_profile": []}
    salt = config["cheap_controls"]["tie_salt"]
    for row in development:
        options = _options(row)
        candidates = [a for a in anchors if a["user"] == source_user(row)] or anchors
        query = _tokens(" ".join(options.values()))
        nearest = min(candidates, key=lambda a: (-_jaccard(query, a["query"]), rank(a["base_id"], salt)))
        weights = Counter()
        for anchor in candidates:
            for token in anchor["preferred"]:
                weights[token] += Fraction(1, len(candidates))
            others = anchor["other_options"]
            for other in others:
                for token in other:
                    weights[token] -= Fraction(1, len(candidates) * len(others))
        for name in output:
            def score(letter):
                tok = _tokens(options[letter])
                return _jaccard(tok, nearest["preferred"]) if name == "anchor_memory" else sum(weights[t] for t in tok) / max(1, len(tok))
            selected = min(LETTERS, key=lambda letter: (-score(letter), rank(str(row["base_id"]) + "|" + options[letter].casefold(), salt)))
            p = [float(letter == selected) for letter in LETTERS]
            output[name].append({**binding(row), "full_vocab_choice_probabilities": p, "normalized_choice_probabilities": p,
                                 "full_vocabulary_choice_mass": 1.0, "source": name, "accessible_delayed_bases": list(pooled_base_ids)})
    return output
