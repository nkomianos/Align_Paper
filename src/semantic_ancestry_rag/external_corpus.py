"""Fail-closed extraction of a licensed Stack Exchange replication corpus.

This module deliberately does *not* create G1 runner inputs.  It turns a
time-pinned public dump into an auditable source-packet shortlist only after a
separate protocol has selected a community, snapshot, and eligibility rules.
That separation prevents an extractor convenience from silently becoming an
external scientific result.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from under_extinction.io import canonical_json, sha256_file, write_json, write_jsonl


KIND = "semantic_ancestry_rag_stackexchange_source_packet_shortlist"
CC_BY_SA_4 = "CC BY-SA 4.0"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _clean_html(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    return re.sub(r"\s+", " ", unescape(" ".join(parser.parts))).strip()


def _integer(row: Mapping[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"malformed Stack Exchange post: missing integer {field}") from exc


@dataclass(frozen=True)
class _Question:
    post_id: int
    title: str
    body: str
    score: int
    creation_date: str
    content_license: str


@dataclass(frozen=True)
class _Answer:
    post_id: int
    parent_id: int
    body: str
    score: int
    creation_date: str
    content_license: str


@dataclass(frozen=True)
class SourcePacket:
    """A cited, multi-answer public source packet; not a scored G1 example."""

    question_id: str
    question: str
    answer_passages: tuple[str, ...]
    source_urls: tuple[str, ...]
    licenses: tuple[str, ...]
    source_snapshot: str
    selection_rule: str


def _rows(posts_path: Path) -> Iterable[dict[str, str]]:
    """Stream dump rows while releasing parsed XML elements promptly."""

    for _event, element in ET.iterparse(posts_path, events=("end",)):
        if element.tag == "row":
            yield dict(element.attrib)
            element.clear()


def _question_candidates(
    posts_path: Path, *, cutoff: str, min_question_score: int, allowed_license: str,
) -> dict[int, _Question]:
    candidates: dict[int, _Question] = {}
    for row in _rows(posts_path):
        if row.get("PostTypeId") != "1" or row.get("ContentLicense") != allowed_license:
            continue
        if row.get("CreationDate", "") > cutoff or _integer(row, "Score") < min_question_score:
            continue
        title, body = _clean_html(row.get("Title", "")), _clean_html(row.get("Body", ""))
        if not title or len(body) < 80:
            continue
        post_id = _integer(row, "Id")
        candidates[post_id] = _Question(post_id, title, body, _integer(row, "Score"), row["CreationDate"], row["ContentLicense"])
    return candidates


def _answers_for_candidates(
    posts_path: Path, candidates: Mapping[int, _Question], *, min_answer_score: int, allowed_license: str,
) -> dict[int, list[_Answer]]:
    answers: dict[int, list[_Answer]] = {question_id: [] for question_id in candidates}
    for row in _rows(posts_path):
        if row.get("PostTypeId") != "2" or row.get("ContentLicense") != allowed_license:
            continue
        parent_id = _integer(row, "ParentId")
        if parent_id not in candidates or _integer(row, "Score") < min_answer_score:
            continue
        body = _clean_html(row.get("Body", ""))
        if len(body) < 80:
            continue
        answers[parent_id].append(_Answer(
            _integer(row, "Id"), parent_id, body, _integer(row, "Score"), row.get("CreationDate", ""), row["ContentLicense"],
        ))
    return answers


def _rank(question: _Question, snapshot: str) -> str:
    # Stable pseudorandom sampling avoids silently cherry-picking only highly
    # polarizing or popular questions after looking at model behavior.
    return hashlib.sha256(f"{snapshot}|{question.post_id}|{question.title}".encode("utf-8")).hexdigest()


def extract_source_packets(
    *, posts_xml: str | Path, destination: str | Path, manifest_destination: str | Path,
    source_snapshot: str, site: str, cutoff: str, count: int = 160,
    min_question_score: int = 3, min_answer_score: int = 2, answers_per_question: int = 5,
    allowed_license: str = CC_BY_SA_4,
) -> dict[str, Any]:
    """Select a reproducible licensed multi-answer packet shortlist.

    The caller must retain the dump and manifest, and must separately freeze
    the G1 question/answer semantics and scoring protocol.  This helper never
    inspects a model output and consequently cannot adapt selection to a result.
    """

    posts = Path(posts_xml)
    output, manifest = Path(destination), Path(manifest_destination)
    if not posts.is_file():
        raise FileNotFoundError(f"Stack Exchange Posts.xml does not exist: {posts}")
    if output.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite an external corpus shortlist or manifest")
    if not source_snapshot or not site or count < 30 or answers_per_question < 3:
        raise ValueError("snapshot/site/count/answer cardinality does not meet external-corpus minimums")
    if allowed_license != CC_BY_SA_4:
        raise ValueError("external shortlist currently permits only explicit CC BY-SA 4.0 posts")
    candidates = _question_candidates(posts, cutoff=cutoff, min_question_score=min_question_score, allowed_license=allowed_license)
    answers = _answers_for_candidates(posts, candidates, min_answer_score=min_answer_score, allowed_license=allowed_license)
    eligible = [question for question in candidates.values() if len(answers[question.post_id]) >= answers_per_question]
    selected = sorted(eligible, key=lambda question: _rank(question, source_snapshot))[:count]
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible questions; refusing to freeze fewer than requested {count}")
    selection_rule = (
        f"CC_BY_SA_4_ONLY; question_score>={min_question_score}; answer_score>={min_answer_score}; "
        f"answers_per_question={answers_per_question}; created_at_or_before={cutoff}; stable_snapshot_hash_rank"
    )
    normalized_site = site.removeprefix("https://").removeprefix("http://").rstrip("/")
    packets: list[SourcePacket] = []
    for question in selected:
        best = sorted(answers[question.post_id], key=lambda answer: (-answer.score, answer.creation_date, answer.post_id))[:answers_per_question]
        packets.append(SourcePacket(
            question_id=f"{normalized_site}:{question.post_id}",
            question=f"{question.title}\n\n{question.body}",
            answer_passages=tuple(answer.body for answer in best),
            source_urls=tuple(f"https://{normalized_site}/a/{answer.post_id}" for answer in best),
            licenses=tuple(answer.content_license for answer in best),
            source_snapshot=source_snapshot,
            selection_rule=selection_rule,
        ))
    write_jsonl(output, (asdict(packet) for packet in packets))
    result = {
        "kind": KIND,
        "source_snapshot": source_snapshot,
        "site": normalized_site,
        "posts_xml_sha256": sha256_file(posts),
        "packet_count": len(packets),
        "packets_sha256": sha256_file(output),
        "allowed_license": allowed_license,
        "selection_rule": selection_rule,
        "scoring_status": "NOT_G1_INPUT__SEPARATE_PREREGISTRATION_REQUIRED",
    }
    write_json(manifest, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract a licensed external ancestry-RAG source-packet shortlist")
    parser.add_argument("--posts-xml", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--manifest-destination", required=True)
    parser.add_argument("--source-snapshot", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--min-question-score", type=int, default=3)
    parser.add_argument("--min-answer-score", type=int, default=2)
    parser.add_argument("--answers-per-question", type=int, default=5)
    args = parser.parse_args(argv)
    print(canonical_json(extract_source_packets(**vars(args))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
