"""Reconcile saved PMI wall timers; these are not billed or GPU-kernel hours."""
import hashlib
import json
import math
from pathlib import Path


def main():
    root = Path('artifacts/pmi_prefix_diagnostic_20260910')
    sources = [('result_v1', 'TIMING.json'),
               ('thinking_traces_v2', 'SUMMARY.json'),
               ('thinking_comparison_v1', 'SUMMARY.json'),
               ('numerics_v1', 'SUMMARY.json'),
               ('thinking_comparison_v2_cached', 'SUMMARY.json')]
    rows = []
    for directory, name in sources:
        run = root/directory
        manifest = json.loads((run/'MANIFEST.json').read_text())
        data = (run/name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert digest == manifest[name]
        seconds = json.loads(data)['seconds']
        assert math.isfinite(seconds) and seconds >= 0
        rows.append(dict(run=directory, timing_file=name, sha256=digest, seconds=seconds))
    total = math.fsum(row['seconds'] for row in rows)
    result = dict(scope='five completed AWS PMI diagnostic timers only', runs=rows,
                  recorded_section_seconds=total, recorded_section_hours=total/3600,
                  billed_instance_hours=None, gpu_kernel_hours=None,
                  h200_equivalent_hours=None,
                  excluded='Untimed setup/loading, hashing, transfers, failed initial launch, idle allocation, other experiments')
    (root/'TIMING_LEDGER.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
