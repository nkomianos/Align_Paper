"""Create a local, explicitly scoped replay backup; no remote publishing."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    names = [f'scripts/{name}.py' for name in (
        'audit_branch_viability_horizon_null', 'audit_clara_matched_coverage',
        'audit_clara_simultaneous', 'audit_pmi_timing_ledger')]
    names += [f'src/research_pilots/{name}.py' for name in ('__init__', 'common', 'clara')]
    names += [f'docs/{name}_20260910.md' for name in (
        'POSITION_RELIABILITY_AUDIT', 'CLARA_MATCHED_COVERAGE_AUDIT',
        'AWS_PMI_TIMING_RECONCILIATION')]
    for directory in ('research_pilots_20260906/clara_cpu_final',
                      'clara_matched_coverage_20260910', 'clara_simultaneous_20260910'):
        names += [str(p.relative_to(root)).replace('\\', '/')
                  for p in sorted((root/'artifacts'/directory).glob('*.json'))]
    names += ['artifacts/position_reliability_audit_20260910/HORIZON_NULL.json']
    for run, timing in (('result_v1', 'TIMING.json'), ('thinking_traces_v2', 'SUMMARY.json'),
                        ('thinking_comparison_v1', 'SUMMARY.json'), ('numerics_v1', 'SUMMARY.json'),
                        ('thinking_comparison_v2_cached', 'SUMMARY.json')):
        names += [f'artifacts/pmi_prefix_diagnostic_20260910/{run}/{file}'
                  for file in (timing, 'MANIFEST.json')]
    names += ['artifacts/pmi_prefix_diagnostic_20260910/TIMING_LEDGER.json']
    assert len(names) == len(set(names))
    manifest = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    out = root/'artifacts/deployment'/f'cpu_audits_{revision[:7]}.zip'
    with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(root/name, name)
        archive.writestr('BACKUP_MANIFEST.json', json.dumps(dict(
            revision=revision, files=manifest,
            scope='Four CPU replays only; PMI neural logits/weights and whole-program evidence excluded'), indent=2))
    with zipfile.ZipFile(out) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(names) | {'BACKUP_MANIFEST.json'}
        for name, expected in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix('.zip.sha256').write_text(f'{digest}  {out.name}\n')
    print(json.dumps(dict(path=str(out), files=len(names), bytes=out.stat().st_size, sha256=digest)))


if __name__ == '__main__':
    main()
