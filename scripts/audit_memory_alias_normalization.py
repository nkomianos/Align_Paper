"""CPU-only input equivalence check; never generates or imputes model answers."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re


EVENT = re.compile(r'(?i)\b(event )(\d+)\b')


def normalize(text):
    """First-mention aliases use text only, without values, edges, or answer keys."""
    aliases = {}

    def replace(match):
        original = match[2]
        if original not in aliases:
            aliases[original] = str(len(aliases))
        return match[1] + aliases[original]

    return EVENT.sub(replace, text), aliases


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = args.run / 'MANIFEST.json'
    for name, expected in json.loads(manifest.read_text()).items():
        assert Path(name).name == name
        assert digest(args.run / name) == expected
    rows = json.loads((args.run / 'INPUTS.json').read_text())
    assert len(rows) == 48
    by_base = {}
    for row in rows:
        by_base.setdefault(row['base'], {})[row['presentation']] = row
    assert len(by_base) == 24
    records = []
    for base, pair in sorted(by_base.items()):
        assert set(pair) == {0, 1}
        text = pair[0]['text']
        canonical, aliases = normalize(text)
        renamed, _ = normalize(pair[1]['text'])
        assert canonical == renamed
        assert normalize(canonical)[0] == canonical
        recovered = EVENT.sub(lambda m: m[1] + {v: k for k, v in aliases.items()}[m[2]], canonical)
        assert recovered == text
        # Exhaustively relabel the identifiers that actually occur in input text.
        # This has no access to latent chronology or the gold response.
        identifiers = list(aliases)
        permutation_count = 0
        for permutation in itertools.permutations(identifiers):
            mapping = dict(zip(identifiers, permutation))
            permuted = EVENT.sub(lambda m: m[1] + mapping[m[2]], text)
            assert normalize(permuted)[0] == canonical
            permutation_count += 1
        records.append({'base': base, 'paired_inputs_equal_after_normalization': True,
                        'all_permutations_checked': permutation_count,
                        'canonical_text_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
                        'identity_text_already_canonical': canonical == text})
    report = {'classification': 'DEVELOPMENTAL_INPUT_EQUIVALENCE_ONLY',
              'manifest_sha256': digest(manifest), 'source_sha256': digest(Path(__file__)),
              'pairs': len(records), 'permutations_checked': sum(r['all_permutations_checked'] for r in records),
              'identity_texts_unchanged': sum(r['identity_text_already_canonical'] for r in records),
              'records': records,
              'scope': 'No model invocation, accuracy measurement, learned repair, or graph extraction. '
                       'Only identifier bijections preserving all other text are covered. '
                       'Record-order invariance and ambiguous real-world coreference are not covered.'}
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
