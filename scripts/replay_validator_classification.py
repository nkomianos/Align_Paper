"""Re-execute every sealed patch from raw completions in the CPU environment.

This intermediate check does not replace the final seven-phase verifier.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from validator_monoculture.evaluation import classify_patch
from validator_monoculture.prompts import parse_patch_completion
from validator_monoculture.serde import bind_corpus, load_private_oracles, load_public_tasks


def sealed_rows(phase, filename):
    manifest = json.loads((phase / 'MANIFEST.json').read_text())
    assert manifest['state'] == 'COMPLETE' and (phase / 'COMPLETE').read_bytes() == b''
    raw = (phase / filename).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['files'][filename]['sha256']
    return [json.loads(line) for line in raw.splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.resolve().is_relative_to(args.run.resolve()):
        raise ValueError('fresh report outside run required')
    tasks, oracles = bind_corpus(load_public_tasks(args.run / 'corpus/public/tasks.jsonl'),
                                load_private_oracles(args.run / 'corpus/private/oracles.jsonl'))
    phases = args.run / 'evidence/phases'
    saved = {r['patch_id']: r for r in sealed_rows(phases / 'classifications', 'private_classifications.jsonl')}
    output = []
    for family in ('qwen3_5', 'gemma4'):
        for raw in sealed_rows(phases / ('patches_' + family), 'raw_patch_completions.jsonl'):
            task = tasks[raw['task_id']]
            try:
                source = parse_patch_completion(raw['raw_completion'], entrypoint=task.entrypoint, signature=task.signature)
            except ValueError as exc:
                source = None
                value = {'schema_version': 'validator-monoculture-patch-classification-v2',
                         'task_id': task.task_id, 'cwe_id': task.cwe_id, 'split': task.split.value,
                         'status': 'REJECTED_COMPLETION_PARSE', 'plausible_security_repair': False,
                         'fully_correct': False, 'parse_error': type(exc).__name__ + ': ' + str(exc)}
            else:
                value = classify_patch(task, oracles[task.task_id], source, timeout_seconds=2.0)
            prior = saved[raw['patch_id']]
            output.append({'patch_id': raw['patch_id'], 'classification': value,
                           'matches_saved': value == prior['classification'] and source == prior['parsed_source']})
            print(json.dumps({'completed': len(output), 'matches_saved': output[-1]['matches_saved']}), flush=True)
    assert len(output) == len(saved) == 192
    report = {'records': len(output), 'mismatches': sum(not r['matches_saved'] for r in output),
              'status_counts': dict(Counter(r['classification']['status'] for r in output)),
              'rows': output, 'scope': 'Independent execution replay, shared frozen oracle/evaluator; not final verification.'}
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
