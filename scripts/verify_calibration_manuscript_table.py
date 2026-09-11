"""Bind the draft calibration table to sealed per-row scores, not prose summaries."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = ROOT / 'retrieved/lambda_h100_20260909/calibration/suite_v3/hindsight_calibration'
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    source = ROOT / 'paper/main.tex'
    latex = source.read_text()
    report = []
    for arm, label in [('supervised', 'Supervised'), ('frozen_teacher', 'Frozen teacher'),
                       ('current_teacher', 'Updating teacher')]:
        metrics = []
        receipts = []
        for split, count in [('train_student', 128), ('final_student', 32)]:
            name = f'{arm}_{split}.json'
            path = root / name
            assert sha(path) == manifest[name]
            payload = json.loads(path.read_text())
            rows = payload['rows']
            assert len(rows) == 4 * count and len({r['id'] for r in rows}) == len(rows)
            groups = defaultdict(list)
            for row in rows:
                groups[row['base_id']].append(row)
            assert len(groups) == count
            assert all(len(v) == 4 and {r['label_rotation'] for r in v} == {0, 1, 2, 3}
                       for v in groups.values())
            accuracy = sum(r['correct'] for r in rows) / len(rows)
            position_range = sum(max(r['conditional_probability'] for r in group)
                                 - min(r['conditional_probability'] for r in group)
                                 for group in groups.values()) / count
            assert abs(accuracy - payload['summary']['correct']) < 1e-12
            assert abs(position_range - payload['summary']['mean_position_range']) < 1e-12
            metrics.append((accuracy, position_range))
            receipts.append(dict(file=name, sha256=sha(path)))
        values = [metrics[0][0], metrics[1][0], metrics[0][1], metrics[1][1]]
        rounded = [str(Decimal(str(v)).quantize(Decimal('.000001'), rounding=ROUND_HALF_UP)).lstrip('0')
                   for v in values]
        expected_line = label + ' & ' + ' & '.join(rounded) + r' \\'
        assert expected_line in latex, expected_line
        report.append(dict(arm=arm, table_values=values, receipts=receipts))
    result = dict(source_sha256=sha(source), score_manifest_sha256=sha(root / 'MANIFEST.json'),
                  rows=report, table_matches=True,
                  scope='Sealed per-row score replay only; full raw-logit verifier was run in the preceding calibration audit. '
                        'No neural checkpoint execution or scientific green light.')
    out = ROOT / 'artifacts/paper_render_20260911/calibration_table_verified.json'
    with out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(dict(table_matches=True, arms=len(report))))


if __name__ == '__main__':
    main()
