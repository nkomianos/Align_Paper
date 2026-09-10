"""Pin public BFCL source for qualification; do not execute benchmark tools."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'Align-Paper-source-audit'}), timeout=60) as response:
        return response.read()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    revision = json.loads(fetch('https://api.github.com/repos/ShishirPatil/gorilla/commits/main'))['sha']
    base = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/{revision}/'
    paths = ['LICENSE', 'berkeley-function-call-leaderboard/bfcl_eval/data/README.md',
             'berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json',
             'berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json']
    records = []
    for i, path in enumerate(paths):
        content = fetch(base+path); name = f'{i}_{Path(path).name}'
        (a.out/name).write_bytes(content)
        records.append({'path': path, 'local_name': name, 'url': base+path,
                        'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})
    (a.out/'DOWNLOAD.json').write_text(json.dumps({'revision': revision, 'files': records}, indent=2))
    print(json.dumps({'revision': revision, 'files': len(records)}))


if __name__ == '__main__':
    main()
