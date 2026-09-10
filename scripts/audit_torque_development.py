"""Audit temporal QA data suitability without manufacturing memory labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    tree = json.loads((args.source / 'tree.json').read_text(encoding='utf-8-sig'))
    assert not tree.get('truncated')
    receipts = json.loads((args.source / 'DOWNLOAD.json').read_text(encoding='utf-8-sig'))
    for row in receipts:
        raw = (args.source / row['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['sha256'].lower()
        blob = next(x for x in tree['tree'] if x['path'] == row['path'])
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == blob['sha']
    data = json.loads((args.source / 'data__dev.json').read_text())
    counts = Counter()
    errors = []
    articles, clusters = set(), set()
    for pid, passage in data.items():
        articles.add(pid.rsplit('_sentid_', 1)[0])
        event_ids = set(passage['events']['answer']['indices'])
        for question, qa in passage['question_answer_pairs'].items():
            counts['questions'] += 1
            counts['default_questions'] += bool(qa['is_default_question'])
            clusters.add((pid, qa['cluster_id']))
            answer = qa['answer']
            counts['empty_consensus_answers'] += not answer['indices']
            individual = [frozenset(a['indices']) for a in qa['individual_answers']]
            counts['individual_answer_set_disagreement'] += len(set(individual)) > 1
            counts['empty_consensus_with_nonempty_individual'] += not answer['indices'] and any(individual)
            counts['questions_with_before'] += bool(re.search(r'\bbefore\b', question, re.I))
            counts['questions_with_after'] += bool(re.search(r'\bafter\b', question, re.I))
            assert len(answer['indices']) == len(answer['spans'])
            for index, span in zip(answer['indices'], answer['spans']):
                match = re.fullmatch(r'\((\d+),(\d+)\)', index)
                if not match:
                    errors.append({'passage_id': pid, 'kind': 'unparsed_span_index'})
                    continue
                start, end = map(int, match.groups())
                if passage['passage'][start:end] != span:
                    errors.append({'passage_id': pid, 'kind': 'span_text_mismatch'})
                if index not in event_ids:
                    counts['consensus_spans_outside_event_inventory'] += 1
    result = {'classification': 'DEVELOPMENTAL_DATA_SUITABILITY',
              'passages': len(data), 'source_article_ids': len(articles),
              'question_clusters': len(clusters), 'counts': dict(counts), 'span_errors': errors,
              'decision': 'NOT_A_DIRECT_INCOMPLETE_ORDER_MEMORY_REPLICATION',
              'limitations': ['Only official DEV fetched; train/test contents unopened.',
                  'Consensus span answers are not gold precedence graphs or possible-world labels.',
                  'Annotator disagreement is not automatically semantic ambiguity.',
                  'Empty answer is not automatically a request for clarification.',
                  'Article IDs/clusters are dependence indicators, not proof of independent domains.',
                  'No model evaluation, numeric state updates, relabeling or benchmark score produced.']}
    with args.out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
