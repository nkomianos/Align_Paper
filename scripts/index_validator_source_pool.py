"""Inventory every Python candidate without inferring qualification from edits."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

REVIEWED = {
    'CVE-2022-28347': ('downstream_masks_component_omissions', 'Django frontend guard ablations separate locally; fixed PostgreSQL backend rejects both probes, so no retained security defect established.'),
    'CVE-2022-4724': ('second_defect_unestablished', 'Rdiffweb diff adds fingerprint uniqueness and migration; multiple edits do not prove independently repairable defects.'),
    'CVE-2022-41672': ('second_defect_unestablished', 'Airflow inactive-user pre-request hook adds one demonstrated admission condition.'),
    'CVE-2024-48911': ('candidate_needs_execution', 'OpenCanary changes config precedence and shell invocation; distinct edits need policy and execution qualification.'),
    'CVE-2024-49750': ('component_positive', 'Snowflake: two regex omissions; one component, full corpus unqualified.'),
    'CVE-2018-18074': ('component_positive', 'Requests: scheme and port checks; stubbed integration.'),
    'CVE-2023-41039': ('component_positive', 'RestrictedPython: two attribute guards; no full compiler integration.'),
    'CVE-2022-29217': ('component_positive', 'PyJWT: two omitted key formats; selected upstream regressions pass.'),
    'CVE-2015-1326': ('one_defect_apparatus_only', 'Bounded cleanup enables PoC; second independently repairable defect unestablished.'),
    'CVE-2021-3281': ('reference_counterexample', 'Released Django helper accepts sibling prefix; exclude comprehensive oracle.'),
    'CVE-2022-21712': ('mechanism_overlap', 'Twisted auth redirect overlaps Requests; independent category not established.'),
    'CVE-2023-40267': ('second_defect_unestablished', 'Inspected clone kwargs guard; not qualified as two defects.'),
    'CVE-2025-43859': ('second_defect_unestablished', 'Multiple edits coordinate one chunk-footer state machine.'),
    'CVE-2024-21542': ('reference_counterexample', 'Pinned Luigi class admits sibling prefix with recording tar I/O; no actual extraction claim.'),
    'CVE-2023-34457': ('second_defect_unestablished', 'MechanicalSoup fix changes file input representation; two independent defects not yet demonstrated.'),
    'CVE-2023-26145': ('component_positive', 'Pinned Pydash read/write ablations pass synthetic probes; attribute category overlaps RestrictedPython.'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    raw = a.dataset.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '65e873923d738221389ed32408c4f5fa75882ea07609c500761207d013da0f62'
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for item in sorted(json.loads(raw), key=lambda x: x['cve_id']):
        if item['programming_language'] != 'Python':
            continue
        identifier = item['cve_id']
        old = '\n'.join(f['snippet'] for f in item['vul_func'])
        new = '\n'.join(f['snippet'] for f in item['fix_func'])
        diff = '\n'.join(difflib.unified_diff(old.splitlines(), new.splitlines(), fromfile='benchmark_vulnerable_snippets', tofile='benchmark_fixed_snippets'))
        # Snippet concatenation can duplicate whole modules/functions. Never use
        # hunk count or function count as an independent-defect/task estimator.
        (a.out/f'{identifier}.diff').write_text(diff, encoding='utf-8')
        status, reason = REVIEWED.get(identifier, ('unreviewed', 'No independent source/assay qualification recorded in this inventory.'))
        rows.append({'id': identifier, 'repository': item['repo'], 'cwes': list(item['cwe_info']),
                     'description': item['cve_description'], 'patch_urls': item['patch_url'],
                     'fixed_paths': sorted({f['file_path'] for f in item['fix_func']}),
                     'status': status, 'reason': reason, 'diff_sha256': hashlib.sha256(diff.encode()).hexdigest()})
    report = {'dataset_sha256': hashlib.sha256(raw).hexdigest(), 'python_candidates': len(rows),
              'developmental_components': sum(r['status']=='component_positive' for r in rows),
              'unreviewed': sum(r['status']=='unreviewed' for r in rows),
              'admitted_neural_tasks': 0,
              'scope': 'Exhaustive metadata index, not exhaustive scientific review. Raw CWE labels do not prove four independent categories. Unreviewed does not mean invalid.',
              'rows': rows}
    (a.out/'INDEX.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}, indent=2))


if __name__ == '__main__':
    main()
