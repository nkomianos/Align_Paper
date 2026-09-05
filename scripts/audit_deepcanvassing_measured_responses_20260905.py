"""P3: frozen finite-cohort bounds and all declared descriptive sensitivities."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
import sys

import reconstruct_deepcanvassing_measured_responses_20260905 as reconstruction

OFFSETS = (-30, -20, -10, 0, 10, 20, 30)


def wave_bounds(rows):
    by_arm = {a: [r for r in rows if r['arm'] == a] for a in (0, 1)}
    if not all(by_arm.values()) or sum(map(len, by_arm.values())) != len(rows):
        raise ValueError('both valid arms required')
    means = {a: [fmean(r['score_lower'] for r in group), fmean(r['score_upper'] for r in group)]
             for a, group in by_arm.items()}
    return {'arm_mean_bounds': means,
            'contrast_bounds': [means[1][0] - means[0][1], means[1][1] - means[0][0]]}


def summarize(records):
    result = {}
    for scale in reconstruction.SCALES:
        selected = [r for r in records if r['scale'] == scale]
        grouped = {}
        for row in selected:
            group = grouped.setdefault(row['participant_ordinal'], {'arm': row['arm']})
            if row['wave'] in group or group['arm'] != row['arm']:
                raise ValueError('duplicate wave or inconsistent participant arm')
            group[row['wave']] = row
        if any(set(g) != {'arm', 'baseline', 'immediate', 'recontact'} for g in grouped.values()):
            raise ValueError('complete participant-wave grid required')
        paired = [{'arm': g['arm'], **{wave: (g[wave]['score_lower'], g[wave]['score_upper'])
                                    for wave in ('immediate', 'recontact')}} for g in grouped.values()]
        result[scale] = {'role': 'primary' if scale == 'prejudice' else 'secondary',
                         'wave_components': {wave: wave_bounds([r for r in selected if r['wave'] == wave])
                                             for wave in ('baseline', 'immediate', 'recontact')},
                         'change_in_arm_contrast_bounds': reconstruction.change_in_arm_contrast_bounds(paired)}
        if scale == 'prejudice':
            complete = [g for g in grouped.values() if all(g[w]['score_complete'] is not None for w in ('immediate', 'recontact'))]
            groups = {a: [g for g in complete if g['arm'] == a] for a in (0, 1)}
            contrasts = {w: fmean(g[w]['score_complete'] for g in groups[1]) - fmean(g[w]['score_complete'] for g in groups[0])
                         for w in ('immediate', 'recontact')} if all(groups.values()) else None
            result[scale]['joint_complete_responder_description'] = {
                'counts': {a: len(g) for a, g in groups.items()}, 'contrasts': contrasts,
                'change': contrasts['recontact'] - contrasts['immediate'] if contrasts else None,
                'uncertainty_shown': False, 'scope': 'selected joint-complete respondents only; no full-cohort effect claim'}
    return result


def item_statistics(base_rows, followup):
    cells = {}
    for base in base_rows:
        arm = int(base['condition'] == 'treatment')
        late = followup.get(reconstruction._key(base['prolific_pid']), {})
        for wave in ('immediate', 'recontact'):
            row = base if wave == 'immediate' else late
            for scale, definitions in reconstruction.SCALES.items():
                for item, reverse in definitions:
                    key = (scale, arm, wave, item)
                    cell = cells.setdefault(key, {'observed_sum': 0., 'observed_count': 0, 'missing_count': 0})
                    value = reconstruction.parse_item(row.get(reconstruction.source_column(wave, item)))
                    if value is None:
                        cell['missing_count'] += 1
                    else:
                        cell['observed_sum'] += 100 - value if reverse else value
                        cell['observed_count'] += 1
    return cells


def departure_grids(cells):
    results = {}
    for scale, definitions in reconstruction.SCALES.items():
        relevant = {k: v for k, v in cells.items() if k[0] == scale}
        if len(relevant) != 4 * len(definitions) or any(v['observed_count'] == 0 for v in relevant.values()):
            results[scale] = {'skipped': True, 'reason': 'item lacks observations in an arm/wave'}
            continue
        grid = []
        for control in OFFSETS:
            row = []
            for treatment in OFFSETS:
                means = {}
                for arm in (0, 1):
                    offset = control if arm == 0 else treatment
                    for wave in ('immediate', 'recontact'):
                        values = []
                        for item, _ in definitions:
                            c = cells[(scale, arm, wave, item)]
                            imputed = min(100., max(0., c['observed_sum']/c['observed_count'] + offset))
                            values.append((c['observed_sum'] + c['missing_count']*imputed) /
                                          (c['observed_count'] + c['missing_count']))
                        means[(arm, wave)] = fmean(values)
                row.append((means[(1, 'recontact')] - means[(0, 'recontact')]) -
                           (means[(1, 'immediate')] - means[(0, 'immediate')]))
            grid.append(row)
        results[scale] = {'skipped': False, 'control_offsets_rows': OFFSETS,
                         'treatment_offsets_columns': OFFSETS, 'change_grid': grid,
                         'scope': 'assumed missing-item departures; not confidence intervals or validated MAR'}
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reconstruction-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('preserve prior statistical receipt')
    receipt_path = args.reconstruction_root / 'RECONSTRUCTION_RECEIPT.json'
    receipt = json.loads(receipt_path.read_text())
    if reconstruction.file_hash(reconstruction.PROTOCOL) != receipt['protocol_sha256']:
        raise ValueError('prospective protocol changed')
    if reconstruction.file_hash(Path(reconstruction.__file__)) != receipt['script_sha256']:
        raise ValueError('reconstruction source changed')
    source = reconstruction.DEFAULT_SOURCES
    actual_hashes = {n: reconstruction.file_hash(source/n) for n in reconstruction.SOURCE_HASHES}
    if actual_hashes != reconstruction.SOURCE_HASHES or actual_hashes != receipt['source_sha256']:
        raise ValueError('public source hash mismatch')
    numeric = args.reconstruction_root / 'LOCAL_ONLY_numeric_scores.jsonl'
    if reconstruction.file_hash(numeric) != receipt['numeric_sha256']:
        raise ValueError('reconstruction table changed')
    records = [json.loads(line) for line in numeric.read_text().splitlines()]
    sys.path.insert(0, str(source/'parser_runtime'))
    import pyreadr
    if pyreadr.__version__ != '0.5.6':
        raise ValueError('pinned parser required')
    frame = pyreadr.read_r(str(source/'clean_s1_filt.rds'))[None]
    fields = {reconstruction.source_column(w, i) for w in ('baseline', 'immediate')
              for definitions in reconstruction.SCALES.values() for i, _ in definitions}
    fields |= {'prolific_pid', 'condition', 'Experimental_Condition'}
    base_rows = frame.loc[:, sorted(fields)].to_dict(orient='records')
    del frame
    followup, metadata = reconstruction.load_followup(source/'historical_five_week_followup.csv')
    replayed, counts = reconstruction.reconstruct_rows(base_rows, followup)
    if replayed != records or json.loads(json.dumps(counts)) != receipt['counts'] or metadata != receipt['metadata_rows']:
        raise ValueError('reconstruction replay or cohort binding differs')
    cells = item_statistics(base_rows, followup)
    report = {'protocol_sha256': receipt['protocol_sha256'], 'source_sha256': actual_hashes,
              'reconstruction_receipt_sha256': reconstruction.file_hash(receipt_path),
              'statistical_script_sha256': reconstruction.file_hash(Path(__file__)),
              'cohort_counts': counts, 'complete_item_counts': receipt['item_completeness'],
              'results': summarize(records), 'departure_grids': departure_grids(cells),
              'item_sufficient_statistics': {'/'.join(map(str,k)):v for k,v in cells.items()},
              'classification': 'descriptive selected-cohort measured-response audit',
              'latent_state_identified': False, 'causal_itt_claim': False, 'prediction_run': False,
              'participant_level_outputs_in_report': False, 'paper_green_light': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({'out': str(args.out), 'results': report['results']}, indent=2))


if __name__ == '__main__':
    main()
