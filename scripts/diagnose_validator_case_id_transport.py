"""Post-hoc transport diagnosis; never a replacement for the registered result.

Only removes optional string case_id metadata from otherwise exact test objects.
Uses unchanged frozen reference execution, proposal budgets and planted selection.
Does not calculate a new primary endpoint or modify any original artifact.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import yaml

from validator_monoculture.prompts import _single_fenced_payload
from validator_monoculture.serde import bind_corpus, deserialize_public_tasks, deserialize_private_oracles
from validator_monoculture.verify import (
    _Suite, _parse_suite, _planted_control_power,
    _default_patch_classifier, _default_vector_evaluator,
)


def remove_case_ids(completion):
    value = json.loads(_single_fenced_payload(completion))
    if not isinstance(value, dict) or set(value) != {'tests'} or not isinstance(value['tests'], list):
        raise ValueError('unsupported envelope')
    clean = []
    for test in value['tests']:
        if not isinstance(test, dict) or set(test) not in (
            {'args', 'kwargs', 'expected'}, {'args', 'kwargs', 'expected', 'case_id'}
        ):
            raise ValueError('unsupported test fields')
        if 'case_id' in test and not isinstance(test['case_id'], str):
            raise ValueError('non-string case_id')
        clean.append({k: test[k] for k in ('args', 'kwargs', 'expected')})
    return json.dumps({'tests': clean}, ensure_ascii=True, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    if args.report.exists() or args.report.resolve().is_relative_to(root):
        raise ValueError('report must be new and outside evidence')
    manifest = json.loads((root / 'COMPLETION_MANIFEST.json').read_text())
    actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*') if p.is_file() and p.name != 'COMPLETION_MANIFEST.json'}
    assert actual == manifest['artifacts_sha256']
    config = yaml.safe_load((root / 'corpus/FROZEN_CONFIG.yaml').read_text())
    tasks, oracles = bind_corpus(
        deserialize_public_tasks((root / 'corpus/public/tasks.jsonl').read_bytes()),
        deserialize_private_oracles((root / 'corpus/private/oracles.jsonl').read_bytes()))
    suites, counts = {}, Counter()
    for family in ('qwen3_5', 'gemma4'):
        path = root / 'evidence/phases' / ('tests_' + family + '_spec_only') / 'raw_test_completions.jsonl'
        for line in path.read_bytes().splitlines():
            row = json.loads(line)
            raw = row['raw_completion']
            assert hashlib.sha256(raw.encode()).hexdigest() == row['completion_sha256']
            limit = config['execution']['max_test_completion_bytes']
            parsed, error = _parse_suite(raw, row['requested_tests'], limit)
            assert (list(parsed) if parsed is not None else None, error) == (row['parsed_tests'], row['parse_error'])
            counts[family + '_original_valid'] += parsed is not None
            if error is not None and len(raw.encode()) <= limit:
                try:
                    adapted = remove_case_ids(raw)
                    candidate, candidate_error = _parse_suite(adapted, row['requested_tests'], limit)
                    if candidate_error is None:
                        parsed, error = candidate, None
                        counts[family + '_recovered'] += 1
                except (ValueError, TypeError):
                    pass
            counts[family + '_remaining_invalid'] += parsed is None
            key = (family, 'spec_only', row['task_id'], None)
            suites.setdefault(key, []).append(_Suite(
                family, 'spec_only', row['task_id'], None, row['suite_index'],
                row['requested_tests'], row['completion_sha256'], parsed, error))
    print(json.dumps({'transport_counts': dict(counts)}), flush=True)
    power = _planted_control_power(
        tasks=tasks, oracles=oracles, suites=suites,
        patch_classifier=_default_patch_classifier, vector_evaluator=_default_vector_evaluator,
        timeout_seconds=config['execution']['sandbox_timeout_seconds'])
    report = {
        'classification': 'POST_HOC_DEVELOPMENTAL_TRANSPORT_DIAGNOSTIC',
        'primary_endpoint_recomputed': False,
        'registered_decision_unchanged': 'INCONCLUSIVE_INSUFFICIENT_APPARATUS_POWER',
        'adapter': 'Remove only optional string case_id metadata; preserve args/kwargs/expected and all slots.',
        'transport_counts': dict(counts), 'planted_control_power': power,
        'source_evidence_sha256': manifest['evidence_root_sha256'],
    }
    with args.report.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'macro_detection': power['macro_detection_rate_by_verifier']}), flush=True)


if __name__ == '__main__':
    main()
