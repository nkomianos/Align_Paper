"""Retrospective admission replay; not a scientific result or a historical guard."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    run = args.root / 'revision_checkpoint_pilot_v1'
    manifest = json.loads((run / 'MANIFEST.json').read_text())
    for name, expected in manifest.items():
        assert Path(name).name == name
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == expected
    provenance = json.loads((run / 'PROVENANCE.json').read_text())
    assert hashlib.sha256(args.runner.read_bytes()).hexdigest() == provenance['runner_sha256']
    config_path = args.root / 'tokenizer/generation_config.json'
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == provenance['base_files']['generation_config.json']
    defaults = json.loads(config_path.read_text())
    calls = [n for n in ast.walk(ast.parse(args.runner.read_text())) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'GenerationConfig']
    assert len(calls) == 1
    actual = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords if isinstance(k.value, ast.Constant)}
    rows = json.loads((run / 'INPUTS.json').read_text())
    by_base = defaultdict(list)
    for r in rows:
        by_base[r['base']].append((r['update'], r['history'], r['messages']))
    groups = defaultdict(list)
    for base, variants in by_base.items():
        signature = json.dumps(sorted(variants, key=lambda x:(x[0],x[1])), sort_keys=True)
        groups[signature].append(base)
    scored = json.loads((args.root / 'SCORED.json').read_text())
    outputs = [json.loads(line) for line in (run / 'OUTPUTS.jsonl').read_text().splitlines()]
    first_keys = {(o['model'], o['id']) for o in outputs[:8]}
    smoke = [r for r in scored['records'] if (r['model'], r['id']) in first_keys]
    assert len(smoke) == 8 and {r['model'] for r in smoke} == {'base'}
    events = []
    for line in (args.root / 'revision_checkpoint_pilot_v1.log').read_text().splitlines():
        if line.startswith('{'):
            event = json.loads(line)
            if 'completed' in event:
                events.append(event)
    assert events[0]['model'] == 'base' and events[0]['completed'] == 8
    report = {
        'classification': 'RETROSPECTIVE_PREFLIGHT_REPLAY_NOT_NEW_MODEL_EVIDENCE',
        'nominal_units': len(by_base), 'unique_units': len(groups),
        'duplicate_groups': [g for g in groups.values() if len(g)>1],
        'actual_do_sample': actual['do_sample'], 'pinned_default_do_sample': defaults['do_sample'],
        'decode_override_present': actual['do_sample'] != defaults['do_sample'],
        'first_batch': {'n': 8, 'valid': sum(r['valid'] for r in smoke),
                        'eos': sum(r['eos'] for r in smoke), 'seconds': events[0]['seconds']},
        'original_interface_threshold': .95,
        'first_batch_passes_interface': sum(r['valid'] for r in smoke)/8 >= .95,
        'full_run_inference_seconds': json.loads((run/'TIMING.json').read_text())['inference_seconds'],
        'scope': 'The eight-output prefix is examined retrospectively; it was not a separately stopped historical smoke stage. '
                 'Observed timing is not a causal estimate of total savings. The decoding difference is a review flag, '
                 'not proof that all deliberate greedy experiments are invalid. No repaired or replacement run is launched.'}
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
