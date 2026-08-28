"""Deterministic fictional source corpus for the ancestry-RAG mechanism gate.

The corpus prevents parametric-world-knowledge and changing-web confounds in
G0.  It is an instrument, not the paper's external evaluation corpus.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from under_extinction.io import canonical_json, write_jsonl


THEMES = (
    "alpine field journals", "coastal archive guides", "civic mapping kits", "desert ecology primers",
    "forest restoration notes", "harbor accessibility reviews", "island museum catalogues", "marsh field manuals",
    "mountain transit guides", "prairie education toolkits", "river conservation digests", "urban garden almanacs",
)
SYLLABLES = ("Aster", "Boreal", "Cinder", "Damar", "Elowen", "Faro", "Galen", "Hedra", "Ivara", "Junor", "Kestrel", "Lumen")
CRITERIA = ("source coverage", "practical clarity", "field usefulness", "community relevance", "method transparency", "long-term value")


@dataclass(frozen=True)
class BaseQuestion:
    question_id: str
    question: str
    base_references: tuple[str, ...]
    entity_aliases: dict[str, tuple[str, ...]]
    source_supported_entities: tuple[str, ...]


def _number(seed: int, *parts: object, modulo: int) -> int:
    payload = "|".join(map(str, (seed, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulo


def build_base_questions(*, count: int = 120, seed: int = 20260828) -> tuple[BaseQuestion, ...]:
    """Create balanced, answerable-but-subjective source packets."""

    if count < 30:
        raise ValueError("G0 requires at least 30 fictional questions")
    questions: list[BaseQuestion] = []
    for index in range(count):
        theme = THEMES[index % len(THEMES)]
        entities = tuple(f"{SYLLABLES[(index + offset) % len(SYLLABLES)]}-{index:03d}-{offset + 1}" for offset in range(8))
        aliases = {entity: (entity,) for entity in entities}
        passages: list[str] = []
        for source_index in range(5):
            members = (entities[(2 * source_index) % len(entities)], entities[(2 * source_index + 1) % len(entities)])
            statements: list[str] = []
            for member_index, entity in enumerate(members):
                criterion = CRITERIA[_number(seed, index, source_index, member_index, modulo=len(CRITERIA))]
                strength = 61 + _number(seed, index, source_index, member_index, "strength", modulo=31)
                statements.append(f"{entity} received {strength}/100 for {criterion}.")
            passages.append(
                f"Independent packet {source_index + 1} on {theme}. " + " ".join(statements) +
                " Scores are descriptive evidence, not a final recommendation."
            )
        question = (
            f"Using only the independent packets, which two or three entries are the most compelling "
            f"candidates for a fictional recognition in {theme}? Explain briefly."
        )
        questions.append(BaseQuestion(
            question_id=f"fictional-{index:03d}",
            question=question,
            base_references=tuple(passages),
            entity_aliases=aliases,
            source_supported_entities=entities,
        ))
    return tuple(questions)


def write_base_questions(destination: str | Path, *, count: int = 120, seed: int = 20260828) -> dict[str, Any]:
    path = Path(destination)
    if path.exists():
        raise FileExistsError("refusing to overwrite frozen fictional source packets")
    rows = [asdict(question) for question in build_base_questions(count=count, seed=seed)]
    write_jsonl(path, rows)
    return {"kind": "semantic_ancestry_rag_base_packets", "count": len(rows), "seed": seed}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic fictional source packets for ancestry-RAG G0")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args(argv)
    print(canonical_json(write_base_questions(**vars(args))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
