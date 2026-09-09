"""Verify terminal inventory, then dispatch the frozen Linux CPU verifier.

Run only after generation exits. This helper never writes into the evidence or
source checkout and never substitutes runner classifications for reconstruction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'checkout', 'python', 'report'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    root, checkout = args.run.resolve(), args.checkout.resolve()
    if args.report.resolve().is_relative_to(root) or args.report.resolve().is_relative_to(checkout):
        raise ValueError('report must be outside evidence and source checkout')
    if args.report.exists():
        raise FileExistsError(args.report)
    manifest = json.loads((root / 'COMPLETION_MANIFEST.json').read_text())
    if manifest['status'] != 'generation_complete__offline_analysis_pending':
        raise ValueError('generation not complete')
    if manifest['git_commit'] != args.commit:
        raise ValueError('unexpected generating commit')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('symlink in run root')
        if path.is_file() and path != root / 'COMPLETION_MANIFEST.json':
            relative = path.relative_to(root).as_posix()
            actual.add(relative)
            if digest(path) != manifest['artifacts_sha256'].get(relative):
                raise ValueError('unlisted or modified artifact: ' + relative)
    if actual != set(manifest['artifacts_sha256']):
        raise ValueError('missing artifacts')
    public = root / 'corpus/public/tasks.jsonl'
    private = root / 'corpus/private/oracles.jsonl'
    config = root / 'corpus/FROZEN_CONFIG.yaml'
    command = [str(args.python.resolve()), '-m', 'validator_monoculture.verify',
               '--evidence-root', str(root / 'evidence'),
               '--public-corpus', str(public), '--private-oracles', str(private),
               '--config', str(config),
               '--expected-public-sha256', manifest['artifacts_sha256']['corpus/public/tasks.jsonl'],
               '--expected-private-sha256', manifest['artifacts_sha256']['corpus/private/oracles.jsonl'],
               '--expected-config-sha256', manifest['config_sha256'],
               '--expected-evidence-sha256', manifest['evidence_root_sha256'],
               '--expected-code-sha256', manifest['code_tree_sha256'],
               '--expected-git-commit', args.commit,
               '--expected-run-binding-sha256', manifest['run_binding_sha256'],
               '--output-report', str(args.report.resolve())]
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONSAFEPATH='1',
               PYTHONNOUSERSITE='1', PYTHONPATH=str(checkout / 'src'))
    subprocess.run(command, cwd=checkout, env=env, check=True)


if __name__ == '__main__':
    main()
