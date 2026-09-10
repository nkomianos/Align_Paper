"""External DEV measurement qualification; timestamped images, not native video."""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from run_unexplored_screens import dump, sha


def parse_count(text):
    match = re.fullmatch(r'\s*(\d+)\s*', text)
    return int(match[1]) if match else None


def decode_views(row, folder):
    import av
    import numpy as np
    from PIL import Image
    folder.mkdir()
    end = float(row['query_time'])
    requests = np.linspace(0, end, 64).tolist()
    picked = []
    previous = None
    decoded = 0
    started = time.monotonic()
    with av.open(row['local_video']) as video:
        stream = video.streams.video[0]
        origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.
        for frame in video.decode(stream):
            if frame.pts is None:
                raise ValueError('Missing presentation timestamp')
            timestamp = float(frame.pts * frame.time_base) - origin
            decoded += 1
            while len(picked) < len(requests) and requests[len(picked)] < timestamp - 1e-8:
                if previous is None:
                    raise ValueError('No causal frame for first requested timestamp')
                picked.append(previous)
            if timestamp > end + 1e-8:
                break
            picture = frame.to_image()
            picture.thumbnail((448, 448))
            previous = (timestamp, picture)
    while len(picked) < 64:
        if previous is None or end - previous[0] > .2:
            raise ValueError('Query outside video support')
        picked.append(previous)
    paths = []
    for index, (timestamp, picture) in enumerate(picked):
        assert timestamp <= requests[index] + 1e-8
        path = folder / f'{index:02d}.png'
        picture.save(path)
        paths.append(str(path))
    preview = [np.asarray(p.resize((32, 32)).convert('L'), dtype=float) for _, p in picked]
    changes = [0.] + [float(np.abs(a-b).mean()) for a, b in zip(preview[1:], preview[:-1])]
    # Fixed quota per temporal stratum prevents spending all frames on one cut.
    adaptive = sorted([i for block in range(8) for i in sorted(range(block*8, (block+1)*8),
                       key=lambda j: (-changes[j], j))[:2]])
    uniform = np.linspace(0, 63, 16).round().astype(int).tolist()
    blank = folder / 'blank.png'
    Image.new('RGB', picked[-1][1].size, 'gray').save(blank)
    sheet = Image.new('RGB', (8*160, 8*120), 'white')
    for i, (_, p) in enumerate(picked):
        copy = p.copy(); copy.thumbnail((160, 120)); sheet.paste(copy, ((i%8)*160, (i//8)*120))
    sheet.save(folder/'contact.png')
    return {'paths': paths, 'timestamps': [t for t, _ in picked], 'requested': requests,
            'uniform16': uniform, 'change16': adaptive, 'dense64': list(range(64)),
            'changes': changes, 'blank': str(blank), 'decoded_frames': decoded,
            'decode_seconds': time.monotonic()-started,
            'note': 'ALL sequential decode work charged to every view; equal VLM frames only for uniform/change'}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    a.out.mkdir(exist_ok=False)
    rows = json.loads((a.root/'video_assets/INPUTS.json').read_text())
    ready = json.loads((a.root/'VIDEO_ASSETS_READY.json').read_text())
    assert sha(a.root/'video_assets/INPUTS.json') == ready['input_sha256']
    dump(a.out/'INPUTS.json', rows)
    dump(a.out/'PROTOCOL.json', {'code_sha256': sha(Path(__file__)), 'scope': '24-video external DEV qualification only',
        'unit': 'distinct source video; 12 snapshot and 12 action videos, not 60 independent cases',
        'input_mode': 'timestamped multi-image sequence, NOT native video processor',
        'sampling': '64 causal preview frames; all decoded frames charged; 16 uniform or 16 largest-change stratified',
        'oracle': 'snapshot last frame; action dense64 is reference, not a ground-truth oracle',
        'qualification': 'snapshot exact accuracy >=.75; parse coverage >=.95 overall; dense action accuracy >=.60',
        'signal': 'change16 net >=3 additional correct action videos over uniform16; fresh replication required',
        'parser': 'whole response nonnegative integer, no answer-extraction fallback',
        'generation': 'greedy, max_new_tokens=64; truncation reported separately',
        'no_claim': 'novel frame selector, full benchmark performance, matched total decoding savings, paper qualification'})
    views = {}
    for row in rows:
        assert sha(Path(row['local_video'])) == row['sha256']
        views[row['id']] = decode_views(row, a.out/row['id'])
    dump(a.out/'VIEWS.json', views)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied; prepared views retained')
    import torch
    import transformers
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, GenerationConfig
    from PIL import Image
    snapshot = Path(ready['snapshot'])
    for name, h in ready['model_hashes'].items(): assert sha(snapshot/name) == h
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(snapshot, local_files_only=True,
        dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    dump(a.out/'MODEL.json', {**ready, 'torch': torch.__version__, 'transformers': transformers.__version__})
    records = []
    started = time.monotonic()
    config = GenerationConfig(do_sample=False, max_new_tokens=64, use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id)
    with (a.out/'OUTPUTS.jsonl').open('x') as output, torch.inference_mode():
        for row in rows:
            view = views[row['id']]
            modes = ['snapshot', 'blank'] if row['category']=='O1-Snap' else ['uniform16', 'change16', 'dense64']
            for mode in modes:
                indices = [63] if mode in ('snapshot', 'blank') else view[mode]
                paths = [view['blank']] if mode=='blank' else [view['paths'][i] for i in indices]
                content = [{'type':'text', 'text': ('These frames are in chronological order from a video observed only '
                    f'through {row["query_time"]} seconds. ' + ('The image shows the current moment. ' if len(indices)==1 else '')
                    + row['question'] + ' Answer with only one nonnegative integer, without explanation.')}]
                for i, path in zip(indices, paths):
                    content.extend([{'type':'text','text':f'Time {view["timestamps"][i]:.6f} seconds:'}, {'type':'image'}])
                messages = [{'role':'user', 'content':content}]
                rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images = [Image.open(path).convert('RGB') for path in paths]
                inputs = processor(text=[rendered], images=images, return_tensors='pt').to('cuda')
                before = time.monotonic()
                generated = model.generate(**inputs, generation_config=config)
                ids = generated[0, inputs.input_ids.shape[1]:].tolist()
                text = processor.tokenizer.decode(ids, skip_special_tokens=True)
                pred = parse_count(text)
                record = {'id':row['id'], 'mode':mode, 'target':row['target'], 'prediction':pred,
                    'correct':pred==row['target'], 'text':text, 'generated_ids':ids,
                    'rendered':rendered, 'input_ids':inputs.input_ids[0].tolist(),
                    'image_grid_thw':inputs.image_grid_thw.tolist(), 'indices':indices,
                    'image_sha256':[sha(Path(path)) for path in paths], 'seconds':time.monotonic()-before,
                    'censored':len(ids)==64 and ids[-1]!=processor.tokenizer.eos_token_id}
                output.write(json.dumps(record, allow_nan=False)+'\n'); output.flush(); records.append(record)
                print(json.dumps({'completed':len(records), 'seconds':time.monotonic()-started}), flush=True)
    accuracy = {mode:sum(r['correct'] for r in records if r['mode']==mode)/12
                for mode in ['snapshot','blank','uniform16','change16','dense64']}
    coverage = sum(r['prediction'] is not None for r in records)/len(records)
    qualified = accuracy['snapshot']>=.75 and accuracy['dense64']>=.60 and coverage>=.95
    signal = (accuracy['change16']-accuracy['uniform16'])*12 >= 3-1e-8
    dump(a.out/'SUMMARY.json', {'classification':'DEVELOPMENTAL', 'accuracy':accuracy,
        'parse_coverage':coverage, 'censored':sum(r['censored'] for r in records),
        'route':'INVALID_PERCEPTION_OR_FORMAT' if not qualified else ('DEV_SIGNAL' if signal else 'STOP_NO_DECISIVE_SELECTION_GAIN'),
        'elapsed_inference_seconds':time.monotonic()-started})
    dump(a.out/'MANIFEST.json', {str(f.relative_to(a.out)):sha(f) for f in a.out.rglob('*') if f.is_file()})
    print('SVC_SCREEN_COMPLETE', flush=True)


if __name__=='__main__': main()
