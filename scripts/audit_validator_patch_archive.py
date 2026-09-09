"""Read-only prompt/seed/parser reconstruction on one sealed phase backup.

This does not execute patches or establish a scientific endpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from validator_monoculture.prompts import patch_prompt, parse_patch_completion
from validator_monoculture.serde import load_public_tasks


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('archive', 'public', 'report'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--family', choices=['qwen3_5', 'gemma4'], required=True)
    args = parser.parse_args()
    models = {'qwen3_5': ('Qwen/Qwen3.5-9B', 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'),
              'gemma4': ('google/gemma-4-12B-it', '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7')}
    model, revision = models[args.family]
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        if any(not (m.isfile() or m.isdir()) for m in members):
            raise ValueError('nonregular archive member')
        files = {m.name: archive.extractfile(m).read() for m in members if m.isfile()}
        if len(files) != sum(m.isfile() for m in members):
            raise ValueError('duplicate archive member')
    prefix = 'patches_' + args.family + '/'
    manifest = json.loads(files[prefix + 'MANIFEST.json'])
    assert manifest['state'] == 'COMPLETE' and files[prefix + 'COMPLETE'] == b''
    assert set(files) == {prefix + k for k in manifest['files']} | {prefix + 'MANIFEST.json', prefix + 'COMPLETE'}
    for name, commitment in manifest['files'].items():
        content = files[prefix + name]
        assert sha(content) == commitment['sha256'] and len(content) == commitment['bytes']
    assert sha(args.public.read_bytes()) == manifest['public_corpus_sha256']
    tasks = load_public_tasks(args.public)
    rows = [json.loads(line) for line in files[prefix + 'raw_patch_completions.jsonl'].splitlines()]
    assert len(tasks) == 32 and len(rows) == 96
    plan = [(task, sample) for task in tasks for sample in range(3)]
    parse_failures = 0
    for row, (task, sample) in zip(rows, plan, strict=True):
        assert (row['task_id'], row['sample_index'], row['cwe_id'], row['split']) == (
            task.task_id, sample, task.cwe_id, task.split.value)
        assert (row['model_id'], row['model_revision'], row['patch_family']) == (model, revision, args.family)
        assert row['prompt_sha256'] == sha(patch_prompt(task.patch_prompt_record()).encode())
        material = f'{task.task_id}|{revision}|patch|{sample}'.encode()
        assert row['seed'] == int.from_bytes(hashlib.sha256(material).digest()[:8], 'big') % (2**63 - 1)
        assert row['completion_sha256'] == sha(row['raw_completion'].encode())
        identity = f"{args.family}|{task.task_id}|{sample}|{row['completion_sha256']}".encode()
        assert row['patch_id'] == 'patch-' + sha(identity)[:24]
        try:
            parsed = parse_patch_completion(row['raw_completion'], entrypoint=task.entrypoint, signature=task.signature)
            error = None
        except ValueError as exc:
            parsed, error = None, type(exc).__name__ + ': ' + str(exc)
            parse_failures += 1
        assert (parsed, error) == (row['parsed_source'], row['parse_error'])
    report = {'archive_sha256': sha(args.archive.read_bytes()), 'family': args.family,
              'records': len(rows), 'parse_failures': parse_failures,
              'prompt_seed_identity_hash_parser_replay': 'PASS',
              'scope': 'Collection integrity only. No semantic patch execution or endpoint verification.'}
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
