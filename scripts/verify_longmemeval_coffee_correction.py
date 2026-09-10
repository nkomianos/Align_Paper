"""Verify the full-context evidence that withdraws a provisional label concern."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path('artifacts/longmemeval_source_20260910')
    path = root / 'longmemeval_oracle.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == '821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c'
    rows = json.loads(path.read_text())
    row = next(r for r in rows if r['question_id'] == 'c6853660')
    turn = row['haystack_sessions'][1][6]
    assert turn['role'] == 'user'
    assert 'I have increased the limit to two cups' in turn['content']
    assert turn['has_answer'] is False
    record = dict(classification='DEVELOPMENTAL_AUDIT_CORRECTION',
                  source_sha256=digest, question_id=row['question_id'],
                  session_index=1, turn_index=6, role=turn['role'],
                  has_answer=turn['has_answer'],
                  turn_sha256=hashlib.sha256(turn['content'].encode()).hexdigest(),
                  conclusion='Released answer supported by later explicit user confirmation. '
                  'Provisional answer-error concern withdrawn; evidence flag incomplete for this item.',
                  limitation='Single inspected item; no aggregate annotation error rate or model result.')
    with (root / 'COFFEE_FULL_CONTEXT_CORRECTION.json').open('x') as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
