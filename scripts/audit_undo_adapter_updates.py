"""Read-only numerical checkpoint audit; not optimizer replay."""
import argparse
import json
from pathlib import Path
import torch


def compare(initial, final):
    if initial.keys() != final.keys() or not initial:
        raise ValueError('adapter key mismatch')
    squared = 0.0
    maximum = 0.0
    changed = 0
    elements = 0
    for key, a in initial.items():
        b = final[key]
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('adapter shape/nonfinite mismatch')
        delta = b.double() - a.double()
        squared += float(delta.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
        changed += int(torch.count_nonzero(delta))
        elements += delta.numel()
    return {'l2_change': squared ** .5, 'max_abs_change': maximum,
            'changed_elements': changed, 'elements': elements}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('output must be outside evidence root')
    initial = torch.load(args.root / 'initial_adapter.pt', map_location='cpu', weights_only=True)
    result = {arm: compare(initial, torch.load(args.root / arm / 'final_adapter.pt',
              map_location='cpu', weights_only=True))
              for arm in ('terminal_sft', 'canonical_distillation', 'local_relation')}
    result['scope'] = 'Numerical checkpoint differences only; run main manifest verifier first'
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
