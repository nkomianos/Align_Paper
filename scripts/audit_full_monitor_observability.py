"""Full-release structural audit; missing context is not proof of wrong labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import pyarrow.parquet as pq
from prepare_cheatbench_event_view import project
from run_unexplored_screens import dump, sha


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    records = []; counts = Counter(); by_benchmark = defaultdict(Counter)
    by_label = defaultdict(Counter); raw_groups = defaultdict(list); event_groups = defaultdict(list)
    for batch in pq.ParquetFile(args.source).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            trace = json.loads(row['trace']); view, status = project(trace)
            events = trace.get('messages') or trace.get('events') or []
            flags = []
            if any(isinstance(e, dict) and e.get('kind') == 'report_prelude' for e in events):
                flags.append('retrospective_report')
            if any(isinstance(e, dict) and e.get('src') == 'user' and isinstance(e.get('msg'), str)
                   and re.fullmatch(r'\$\d+', e['msg']) for e in events):
                flags.append('unresolved_user_reference')
            empty_commands = sum(isinstance(e, dict) and isinstance(e.get('tools'), list) and
                any(isinstance(t, dict) and t.get('cmd') == '' for t in e['tools']) for e in events)
            if empty_commands: flags.append('empty_tool_command_event')
            raw_hash = hashlib.sha256(row['trace'].encode()).hexdigest()
            event_hash = hashlib.sha256(json.dumps(events, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            record = {'id': row['trace_id'], 'benchmark': row['benchmark'], 'label': row['label'],
                'source_dataset': trace.get('source_dataset'), 'schema': trace.get('schema_version'),
                'status': status['status'], 'flags': flags, 'events': len(events),
                'empty_command_events': empty_commands, 'raw_sha256': raw_hash, 'event_sha256': event_hash}
            records.append(record); raw_groups[raw_hash].append(record)
            if events: event_groups[event_hash].append(record)
            counts[status['status']] += 1
            by_benchmark[row['benchmark']][status['status']] += 1
            by_label[row['label']][status['status']] += 1
    def repeated(groups):
        return [{'sha256': h, 'ids': [r['id'] for r in rows], 'labels': [r['label'] for r in rows]}
                for h, rows in groups.items() if len(rows) > 1]
    raw_repeats = repeated(raw_groups); event_repeats = repeated(event_groups)
    report = {'classification': 'FULL_RELEASE_STRUCTURAL_AUDIT', 'n': len(records),
        'status_counts': dict(counts), 'by_benchmark': {k: dict(v) for k, v in by_benchmark.items()},
        'by_label': {k: dict(v) for k, v in by_label.items()},
        'flag_counts': dict(Counter(flag for r in records for flag in r['flags'])),
        'raw_duplicate_groups': raw_repeats, 'event_duplicate_groups': event_repeats,
        'raw_duplicate_groups_with_label_disagreement': sum(len(set(r['labels'])) > 1 for r in raw_repeats),
        'event_duplicate_groups_with_label_disagreement': sum(len(set(r['labels'])) > 1 for r in event_repeats),
        'records': records, 'source_sha256': sha(args.source), 'script_sha256': sha(Path(__file__)),
        'projection_script_sha256': sha(Path(__file__).with_name('prepare_cheatbench_event_view.py')),
        'limits': 'Projection rejection is an admission decision, not proof of a bad benchmark label. Event identity does not establish full-context identity. No neural inference or relabeling.'}
    dump(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'raw_duplicate_groups', 'event_duplicate_groups')}, indent=2))


if __name__ == '__main__': main()
