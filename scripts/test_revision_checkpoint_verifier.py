"""Synthetic acceptance/rejection fixtures, never model experiment evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tokenizers import Tokenizer
from run_revision_checkpoint_pilot import build
from verify_revision_checkpoint_pilot import RUNNER_SHA, ADAPTER_SHA


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    decoder = Tokenizer.from_file(str(args.tokenizer / 'tokenizer.json'))
    config = json.loads((args.tokenizer / 'tokenizer_config.json').read_text())
    eos = decoder.token_to_id(config['eos_token'])
    inputs = build()
    original = []
    for model in ['base', 'adapter']:
        for row in inputs:
            text = f"ANSWER: {row['gold']}"
            original.append({'id': row['id'], 'model': model, 'text': text,
                             'rendered': '\n'.join(m['content'] for m in row['messages']),
                             'output_ids': decoder.encode(text, add_special_tokens=False).ids + [eos], 'eos': True})
    receipts = []
    for case in ['perfect', 'duplicate_positive', 'missing_output', 'corrupt_hash', 'false_eos', 'wrong_gold']:
        root = args.out / case
        root.mkdir()
        rows, outputs = copy.deepcopy(inputs), copy.deepcopy(original)
        if case == 'duplicate_positive':
            for o in outputs:
                if o['model'] == 'adapter' and o['id'] in ['0_1_old_answer', '1_1_old_answer']:
                    o['text'] = 'ANSWER: 9999'
                    o['output_ids'] = decoder.encode(o['text'], add_special_tokens=False).ids + [eos]
        if case == 'missing_output':
            outputs.pop()
        if case == 'false_eos':
            outputs[0]['eos'] = False
        if case == 'wrong_gold':
            rows[0]['gold'] += 1
        save(root / 'INPUTS.json', rows)
        save(root / 'PROVENANCE.json', {'fixture_only': True, 'runner_sha256': RUNNER_SHA,
              'adapter_files': {'adapter_model.safetensors': ADAPTER_SHA},
              'base_files': {name: hashlib.sha256((args.tokenizer / name).read_bytes()).hexdigest()
                             for name in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json']}})
        save(root / 'TIMING.json', {'fixture_only': True})
        (root / 'OUTPUTS.jsonl').write_text(''.join(json.dumps(o)+'\n' for o in outputs), encoding='utf-8')
        save(root / 'MANIFEST.json', {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in root.iterdir()})
        if case == 'corrupt_hash':
            with (root / 'OUTPUTS.jsonl').open('a', encoding='utf-8') as f:
                f.write('\n')
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('verify_revision_checkpoint_pilot.py')),
                     '--run', str(root), '--out', str(root / 'VERIFIED.json'), '--tokenizer', str(args.tokenizer)],
                     capture_output=True, text=True)
        if case in ['perfect', 'duplicate_positive']:
            assert result.returncode == 0, result.stderr
            report = json.loads((root / 'VERIFIED.json').read_text())
            expected = ('STOP_NO_QUALIFYING_INTERACTION' if case == 'perfect' else
                        'REASSESS_DUPLICATE_DESIGN_BEFORE_ANY_FOLLOWUP')
            assert report['route'] == expected
            assert [4, 7] in report['unique_parameter_groups']
            assert len(report['unique_parameter_groups']) == 7
        else:
            assert result.returncode != 0 and 'AssertionError' in result.stderr
        receipts.append({'case': case, 'expected_acceptance': case in ['perfect', 'duplicate_positive'], 'passed': True})
    save(args.out / 'TEST_RESULTS.json', {'classification': 'SYNTHETIC_VERIFIER_TESTS_NOT_MODEL_EVIDENCE', 'tests': receipts})
    print(json.dumps(receipts, indent=2))


if __name__ == '__main__':
    main()
