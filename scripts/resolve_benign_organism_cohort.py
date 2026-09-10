"""Pin the registry's core Italian-food cohort, without fetching model weights."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from inventory_benign_organism_release import get


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.inventory / 'REPORT.json').read_text())
    tree = json.loads((args.inventory / 'repository_tree.json').read_text())
    assert not tree.get('truncated')
    revision = report['repository_revision']
    receipts = {}

    def fetch(name, url):
        raw = get(url)
        (args.out / name).write_bytes(raw)
        receipts[name] = {'url': url, 'sha256': hashlib.sha256(raw).hexdigest()}
        (args.out / 'DOWNLOAD.json').write_text(json.dumps(receipts, indent=2))
        return raw

    raw = fetch('model_registry.json', 'https://raw.githubusercontent.com/'
                f'model-organisms-for-real/model-organism-lottery/{revision}/config/model_registry.json')
    entry = next(row for row in tree['tree'] if row['path'] == 'config/model_registry.json')
    blob_sha = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    assert blob_sha == entry['sha'], 'Registry content disagrees with pinned Git tree'
    registry = json.loads(raw)
    selected = [(key, row) for key, row in registry['models'].items()
                if row.get('quirk_superfamily_id') == 'italian_food'
                and 'core' in row.get('cohorts', [])]
    assert len(selected) == 14, 'Reassess cohort instead of silently changing membership'
    rows = []
    for index, (key, row) in enumerate(selected):
        model_id, ref = row['hf_model_id'], row['hf_revision']
        info = json.loads(fetch(f'checkpoint_{index:02d}.json',
            f'https://huggingface.co/api/models/{model_id}/revision/{quote(ref, safe="")}?blobs=true'))
        sha = info['sha']
        assert len(sha) == 40
        config = json.loads(fetch(f'config_{index:02d}.json',
            f'https://huggingface.co/{model_id}/resolve/{sha}/config.json'))
        weight_files = [f for f in info.get('siblings', [])
                        if f['rfilename'].endswith(('.safetensors', '.bin'))]
        rows.append({'registry_key': key, 'model_id': model_id, 'registry_ref': ref,
                     'resolved_revision': sha, 'architecture': row['model_architecture'],
                     'recipe': row['variant_id'], 'gated': info.get('gated'),
                     'model_type': config.get('model_type'),
                     'weight_files': weight_files,
                     'weight_bytes': sum(f.get('size', 0) for f in weight_files),
                     'all_weight_sizes_known': bool(weight_files) and all('size' in f for f in weight_files)})
        print(f'Resolved {index + 1}/{len(selected)}: {key}', flush=True)
    result = {'classification': 'DEVELOPMENTAL_RELEASE_PROVENANCE_ONLY',
              'repository_revision': revision, 'registry_git_blob_sha1': blob_sha,
              'cohort': rows, 'diffing_bases': registry.get('diffing_bases'),
              'limitations': ['No weights downloaded or checked.',
                  'No behavioral or intervention results produced.',
                  'Recipes are not independent training-seed replications.',
                  'Availability does not establish a novel selector or revive a failed assay.']}
    (args.out / 'REPORT.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'models': len(rows), 'total_weight_bytes': sum(r['weight_bytes'] for r in rows),
                      'all_sizes_known': all(r['all_weight_sizes_known'] for r in rows)}, indent=2))


if __name__ == '__main__':
    main()
