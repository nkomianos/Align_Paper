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
    parser.add_argument('--verifier-commit')
    parser.add_argument('--verifier-code-sha256')
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
    if bool(args.verifier_commit) != bool(args.verifier_code_sha256):
        raise ValueError('both independent verifier commitments are required')
    if args.verifier_commit:
        # Permit only the documented pre-output H100 memory-unit correction.
        # All other Python sources, including scoring and controls, must match.
        changed = subprocess.check_output(
            ['git', '-C', str(checkout), 'diff', '--name-only', args.commit,
             args.verifier_commit, '--', 'src'], text=True).splitlines()
        if changed != ['src/validator_monoculture/verify.py']:
            raise ValueError('unreviewed independent-verifier source changes')
        source = 'src/validator_monoculture/verify.py'
        old = subprocess.check_output(['git', '-C', str(checkout), 'show', args.commit + ':' + source])
        new = subprocess.check_output(['git', '-C', str(checkout), 'show', args.verifier_commit + ':' + source])
        corrected = old.replace(b') < 80 * 1024**3:', b') < 80_000_000_000:').replace(
            b'generation device-memory floor is below 80 GiB',
            b'generation device-memory floor is below 80 GB')
        if corrected == old or corrected != new:
            raise ValueError('verifier differs beyond the documented memory-unit fix')
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
               '--expected-code-sha256', args.verifier_code_sha256 or manifest['code_tree_sha256'],
               '--expected-git-commit', args.verifier_commit or args.commit,
               '--expected-run-binding-sha256', manifest['run_binding_sha256'],
               '--output-report', str(args.report.resolve())]
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONSAFEPATH='1',
               PYTHONNOUSERSITE='1', PYTHONPATH=str(checkout / 'src'))
    subprocess.run(command, cwd=checkout, env=env, check=True)
    receipt = {'generation_commit': args.commit,
               'generation_code_sha256': manifest['code_tree_sha256'],
               'verifier_commit': args.verifier_commit or args.commit,
               'verifier_code_sha256': args.verifier_code_sha256 or manifest['code_tree_sha256'],
               'allowed_source_difference': 'memory admission units only' if args.verifier_commit else None,
               'completion_manifest_sha256': digest(root / 'COMPLETION_MANIFEST.json'),
               'report_sha256': digest(args.report)}
    with args.report.with_suffix('.receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)


if __name__ == '__main__':
    main()
