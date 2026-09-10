"""Prospective interface diagnosis: identical dense64 pixels through native video."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from run_unexplored_screens import dump,sha
from run_svc_observation_screen import parse_count


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    old=a.root/'svc_v1';out=a.root/'svc_native_v2';out.mkdir(exist_ok=False)
    for name,h in json.loads((old/'MANIFEST.json').read_text()).items():assert sha(old/name)==h
    rows=[r for r in json.loads((old/'INPUTS.json').read_text()) if r['category']=='E1-Action']
    views=json.loads((old/'VIEWS.json').read_text());ready=json.loads((a.root/'VIDEO_ASSETS_READY.json').read_text())
    dump(out/'PROTOCOL.json',{'scope':'DEV native-video capability repair, not confirmation or a selector comparison',
        'code_sha256':sha(Path(__file__)),'parent_manifest_sha256':sha(old/'MANIFEST.json'),
        'frozen_gate':'at least 8/12 exact counts; every response parseable; no horizon truncation',
        'pixels':'identical saved dense64 RGB images to v1; native temporal patches and video timestamps now enabled',
        'timestamp_transport':'microsecond integer ticks, fps=1000000; do_sample_frames=False, not actual camera FPS',
        'generation':'same greedy integer request and 64-token horizon',
        'followup':'if qualified, freeze fresh matched native-video selector protocol; otherwise stop this task substrate'})
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    import numpy as np
    from PIL import Image
    import torch
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor,GenerationConfig
    processor=AutoProcessor.from_pretrained(ready['snapshot'],local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(ready['snapshot'],local_files_only=True,
        dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval();model.requires_grad_(False)
    config=GenerationConfig(do_sample=False,max_new_tokens=64,use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id)
    records=[];started=time.monotonic()
    with (out/'OUTPUTS.jsonl').open('x') as f,torch.inference_mode():
        for row in rows:
            paths=[old/row['id']/f'{i:02d}.png' for i in range(64)]
            video=np.stack([np.asarray(Image.open(path).convert('RGB')) for path in paths])
            times=views[row['id']]['timestamps'];ticks=[round(t*1_000_000) for t in times]
            metadata={'total_num_frames':round(row['query_time']*1_000_000)+1,'fps':1_000_000.,
                'frames_indices':ticks,'duration':row['query_time'],'video_backend':'saved_causal_frames'}
            text=('These frames are in chronological order from a video observed only '
                f'through {row["query_time"]} seconds. '+row['question']+
                ' Answer with only one nonnegative integer, without explanation.')
            messages=[{'role':'user','content':[{'type':'text','text':text},{'type':'video'}]}]
            rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[rendered],videos=[video],video_metadata=[metadata],
                do_sample_frames=False,return_tensors='pt').to('cuda')
            ids=model.generate(**inputs,generation_config=config)[0,inputs.input_ids.shape[1]:].tolist()
            response=processor.tokenizer.decode(ids,skip_special_tokens=True);prediction=parse_count(response)
            r={'id':row['id'],'target':row['target'],'prediction':prediction,'correct':prediction==row['target'],
                'text':response,'generated_ids':ids,'input_ids':inputs.input_ids[0].tolist(),
                'decoded_prompt':processor.tokenizer.decode(inputs.input_ids[0]),'video_grid_thw':inputs.video_grid_thw.tolist(),
                'image_sha256':[sha(path) for path in paths],'timestamps':times,'metadata':metadata,
                'censored':len(ids)==64 and ids[-1]!=processor.tokenizer.eos_token_id}
            assert inputs.video_grid_thw[0,0].item()==32
            f.write(json.dumps(r)+'\n');f.flush();records.append(r)
            print(json.dumps({'completed':len(records),'seconds':time.monotonic()-started}),flush=True)
    n=sum(r['correct'] for r in records);coverage=sum(r['prediction'] is not None for r in records)
    censored=sum(r['censored'] for r in records)
    dump(out/'SUMMARY.json',{'correct':n,'n':12,'parse_count':coverage,'censored':censored,
        'route':'CAPABILITY_QUALIFIED_DEV_ONLY' if n>=8 and coverage==12 and censored==0 else 'STOP_INVALID_CAPABILITY',
        'seconds':time.monotonic()-started})
    dump(out/'MANIFEST.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})


if __name__=='__main__':main()
