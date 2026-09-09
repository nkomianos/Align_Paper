"""Extract every row in audited task/model/harness overlap cells, without ranking.

This is a label-conditioned research cohort, not an unconditional benchmark.
Selection is independent of monitor scores and gold snippet locations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'audit', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    if digest(args.data) != audit['data_sha256']:
        raise ValueError('audit does not bind this data file')
    expected = {r['id']: r for r in audit['records']}
    selected = {i for c in audit['overlap_cells']
                for key in ('positive_ids', 'negative_ids') for i in c[key]}
    args.out.mkdir(parents=True, exist_ok=False)
    source = pq.ParquetFile(args.data)
    seen = set()
    with pq.ParquetWriter(args.out / 'overlap.parquet', source.schema_arrow) as writer:
        for batch in source.iter_batches(batch_size=1):
            row = batch.to_pylist()[0]
            identity = row['trace_id']
            if identity not in selected:
                continue
            if identity in seen:
                raise ValueError('duplicate selected ID')
            if hashlib.sha256(row['trace'].encode()).hexdigest() != expected[identity]['trace_sha256']:
                raise ValueError('trace differs from audit')
            writer.write_batch(batch)
            seen.add(identity)
    if seen != selected:
        raise ValueError('missing selected IDs')
    manifest = {'rows': len(seen), 'ids': sorted(seen),
                'data_sha256': audit['data_sha256'],
                'audit_sha256': digest(args.audit),
                'parquet_sha256': digest(args.out / 'overlap.parquet'),
                'script_sha256': digest(Path(__file__)),
                'selection': 'All positive and benign rows in all audited overlap cells; no ranking.',
                'scope': 'Label-conditioned cohort; provisional task identities; not independent task count.'}
    (args.out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'rows': len(seen), 'out': str(args.out)}))


if __name__ == '__main__':
    main()
