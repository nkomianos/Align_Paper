"""Metadata-only inventory and deterministic development selection, not scoring."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    path = a.root / 'longmemeval_oracle.json'
    rows = json.loads(path.read_text())
    assert len({r['question_id'] for r in rows}) == len(rows)
    pool = [r for r in rows if r['question_type'] in ['knowledge-update', 'temporal-reasoning']
            and not r['question_id'].endswith('_abs')]
    selected = sorted(pool, key=lambda r: hashlib.sha256(
        ('memory-substrate-dev-v1:' + r['question_id']).encode()).hexdigest())[:12]
    # Selection uses category and ID only, before reading answer/content.
    # Never export has_answer or answer_session_ids into a model prompt.
    report = dict(classification='DEVELOPMENTAL_SUBSTRATE_INVENTORY',
                  source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  counts=dict(Counter(r['question_type'] for r in rows)),
                  abstention_count=sum(r['question_id'].endswith('_abs') for r in rows),
                  development_ids=[r['question_id'] for r in selected],
                  pool_size=len(pool),
                  oracle_retrieval=True,
                  scope='Public evaluation data, selected here for developmental suitability only. '
                  'No locked confirmation designation or model result. Oracle sessions are privileged retrieval.')
    with (a.root / 'SUBSTRATE_INVENTORY.json').open('x') as f:
        json.dump(report, f, indent=2)
    with (a.root / 'DEVELOPMENT_QUESTIONS.json').open('x') as f:
        json.dump([dict(question_id=r['question_id'], question=r['question'], answer=r['answer'])
                   for r in selected], f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
