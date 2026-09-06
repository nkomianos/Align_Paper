import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def seal(root):
    root = Path(root)
    write(root / 'MANIFEST.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(root.iterdir()) if p.is_file()})


def check_manifest(root):
    root = Path(root)
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    if set(manifest) != {p.name for p in root.iterdir() if p.is_file() and p.name != 'MANIFEST.json'}:
        raise ValueError('artifact set mismatch')
    for name, expected in manifest.items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('artifact hash mismatch')


def score_rows(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate score population')
    correct = 0
    probs = []
    masses = []
    for r in rows:
        logits = r['choice_logits']
        if len(logits) != 2 or not all(math.isfinite(v) for v in logits):
            raise ValueError('invalid binary logits')
        target = r['target']
        if target not in (0, 1):
            raise ValueError('invalid target')
        pred = int(logits[1] > logits[0])
        correct += pred == target
        m = max(logits)
        denom = sum(math.exp(v - m) for v in logits)
        probs.append(math.exp(logits[target] - m) / denom)
        mass = math.exp(m - r['logsumexp']) * denom
        if not 0 <= mass <= 1.00001:
            raise ValueError('invalid choice mass')
        masses.append(min(1., mass))
    return {'n': len(rows), 'accuracy': correct / len(rows),
            'target_probability': sum(probs) / len(probs), 'choice_mass': sum(masses) / len(masses)}


def binary_threshold(rows, max_fpr=.05):
    negatives = [r['probability'] for r in rows if r['label'] == 0]
    if len(negatives) < 20:
        raise ValueError('need at least 20 reviewed calibration negatives')
    candidates = sorted(set(negatives + [1.0]))
    return next(t for t in candidates if sum(v > t for v in negatives) / len(negatives) <= max_fpr)


def monitor_metrics(rows, threshold):
    pos = [r for r in rows if r['label'] == 1]
    neg = [r for r in rows if r['label'] == 0]
    if not pos or not neg:
        raise ValueError('both classes required')
    return {'n': len(rows), 'threshold': threshold,
            'tpr': sum(r['probability'] > threshold for r in pos) / len(pos),
            'fpr': sum(r['probability'] > threshold for r in neg) / len(neg)}
