"""CPU-only public release audit. Never downloads weights or full video datasets.

Pinned metadata/source receipts plus a synthetic compositor contract check, NOT
a scientific test of a video editor. No remote code is executed.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reachable_alpha(mask, dilation=4, blended=True):
    """Reproduce inspected DiffuEraser mask transform, without model inference."""
    m = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.erode(m, kernel, iterations=1)
    m = cv2.dilate(m, kernel, iterations=dilation)
    if not blended:
        return m.astype(np.float32)
    blurred = cv2.GaussianBlur(m * 255, (21, 21), 0) / 255.0
    a = 1 - (1 - m) * (1 - blurred)
    return (a * 255).astype(np.uint8).astype(np.float32) / 255


def compositor_microtest():
    mask = np.zeros((96, 96), np.uint8)
    mask[40:56, 40:56] = 255
    alpha = reachable_alpha(mask)
    source = np.full((96, 96, 3), 190, np.uint8)
    generated = np.zeros_like(source)
    out = (generated * alpha[..., None] + source * (1 - alpha[..., None])).astype(np.uint8)
    outside = alpha == 0
    assert np.array_equal(out[outside], source[outside])
    assert np.any(alpha == 1) and np.any((alpha > 0) & (alpha < 1))
    # A distant hypothetical effect cannot change even with arbitrary predictions.
    assert alpha[8:16, 8:16].max() == 0
    assert np.array_equal(out[8:16, 8:16], source[8:16, 8:16])
    return {"status": "PASS_CONTRACT_ONLY", "shape": list(mask.shape),
            "object_mask_pixels": int((mask > 0).sum()),
            "reachable_pixels": int((alpha > 0).sum()),
            "fully_editable_pixels": int((alpha == 1).sum()),
            "partial_blend_pixels": int(((alpha > 0) & (alpha < 1)).sum()),
            "outside_preserved_exactly": True,
            "caveat": "Synthetic contract check; neither model capability nor benchmark invocation is established."}


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.headers['User-Agent'] = 'Align-Paper-public-contract-audit'
    receipts = []

    def fetch(url, name, cap=2_000_000):
        response = session.get(url, timeout=40)
        response.raise_for_status()
        content = response.content
        if len(content) > cap:
            raise ValueError(f'Public metadata cap exceeded: {name}')
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(content)
        receipts.append({'url': url, 'path': name, 'bytes': len(content), 'sha256': sha(content)})
        return response

    repo_revisions = {}
    for repo, branch, wanted in [
        ('lixiaowen-xw/DiffuEraser', 'master', ['diffueraser/diffueraser.py', 'LICENSE', 'README.md']),
        ('YigitEkin/BeyondMasks', 'main', ['README.md', 'eval.py']),
    ]:
        short = repo.split('/')[-1]
        commit = fetch(f'https://api.github.com/repos/{repo}/commits/{branch}', f'{short}/commit.json').json()['sha']
        repo_revisions[repo] = commit
        fetch(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1', f'{short}/tree.json')
        for name in wanted:
            fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/{name}', f'{short}/{name}')

    dataset = 'yigitekin/BeyondMasks'
    info = fetch(f'https://huggingface.co/api/datasets/{dataset}', 'dataset_info.json').json()
    revision = info['sha']
    url = f'https://huggingface.co/api/datasets/{dataset}/tree/{revision}?recursive=true&expand=false'
    listing = []
    index = 0
    while url:
        response = fetch(url, f'dataset_tree_{index}.json')
        listing.extend(response.json())
        url = response.links.get('next', {}).get('url')
        index += 1
        if index > 20:
            raise RuntimeError('Unexpected pagination size')
    files = [x for x in listing if x['type'] == 'file']
    for name in ['README.md', 'data_example.json', 'eval.py']:
        fetch(f'https://huggingface.co/datasets/{dataset}/resolve/{revision}/{name}', f'dataset/{name}')
    groups = {}
    for f in files:
        group = f['path'].split('/')[0] if '/' in f['path'] else 'root'
        g = groups.setdefault(group, {'files': 0, 'bytes': 0})
        g['files'] += 1
        g['bytes'] += f['size']
    folders = ['masks', 'object_not_present', 'objects_added']
    sets = {folder: {Path(x['path']).stem for x in files if x['path'].startswith(folder + '/')} for folder in folders}
    complete = sorted(set.intersection(*sets.values()), key=lambda s: (int(s) if s.isdigit() else 10**9, s))
    triples = []
    bypath = {x['path']: x for x in files}
    for case in complete:
        members = [bypath[f'{folder}/{case}.mp4'] for folder in folders]
        triples.append({'id': case, 'bytes': sum(x['size'] for x in members), 'files': members})
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'repositories': repo_revisions,
               'dataset_revision': revision, 'dataset_license': info.get('cardData', {}).get('license'),
               'total_files': len(files), 'total_bytes': sum(x['size'] for x in files),
               'groups': groups, 'complete_triples': len(complete),
               'unmatched_ids': {f: sorted(sets[f] - set(complete)) for f in folders},
               'smallest_triples': sorted(triples, key=lambda t: t['bytes'])[:3],
               'root_files': [x['path'] for x in files if '/' not in x['path']],
               'microtest': compositor_microtest(), 'receipts': receipts,
               'downloaded_video_bytes': 0, 'downloaded_weight_bytes': 0}
    with (root / 'summary.json').open('x') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('receipts', 'smallest_triples')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
