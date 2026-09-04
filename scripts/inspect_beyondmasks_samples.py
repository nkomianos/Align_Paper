"""Inspect size-selected release triples; no model inference or effect labels."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    source = json.loads((a.audit / 'summary.json').read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for triple in source['smallest_triples']:
        videos = {}
        receipts = []
        for member in triple['files']:
            name = member['path']
            url = ('https://huggingface.co/datasets/yigitekin/BeyondMasks/resolve/'
                   + source['dataset_revision'] + '/' + name)
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            digest = hashlib.sha256(r.content).hexdigest()
            assert len(r.content) == member['size']
            assert digest == member['lfs']['oid']
            target = a.out / name
            target.parent.mkdir(exist_ok=True)
            with target.open('xb') as f:
                f.write(r.content)
            cap = cv2.VideoCapture(str(target))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            assert frames
            videos[name.split('/')[0]] = frames
            receipts.append(dict(path=name, sha256=digest, fps=fps,
                                 frames=len(frames), shape=list(frames[0].shape)))
        fg, bg, masks = [videos[k] for k in ['objects_added', 'object_not_present', 'masks']]
        equal = len(fg) == len(bg) == len(masks) and fg[0].shape == bg[0].shape == masks[0].shape
        measurements = []
        panels = []
        for idx in sorted(set([0, len(fg)//2, len(fg)-1])):
            if idx >= min(len(bg), len(masks)):
                continue
            if equal:
                delta = np.abs(fg[idx].astype(float)-bg[idx].astype(float)).mean(axis=2)
                outside = masks[idx].mean(axis=2) < 127
                measurements.append(dict(frame=idx, outside_object_mae=float(delta[outside].mean()),
                    outside_fraction_difference_over_10=float((delta[outside] > 10).mean())))
            tiles = [cv2.resize(v[idx], (320, 180)) for v in [fg, bg, masks]]
            panels.append(np.concatenate(tiles, axis=1))
        cv2.imwrite(str(a.out / (triple['id'] + '_contact.png')), np.concatenate(panels, axis=0))
        rows.append(dict(id=triple['id'], receipts=receipts, equal_dimensions_and_counts=equal,
                         descriptive_pixel_differences=measurements))
    result = dict(selection='Three smallest triples by total bytes; not representative or efficacy-selected',
                  revision=source['dataset_revision'], cases=rows,
                  caveat='Pixel differences mix physical effects, motion, synthesis changes, and compression. Not effect ground truth.')
    with (a.out / 'inspection.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
