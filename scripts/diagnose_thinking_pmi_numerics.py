"""Post-outcome apparatus diagnosis only; no teacher or efficacy measurements."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main(result,snapshot,out):
    import torch
    from transformers import AutoModelForCausalLM
    from verify_thinking_pmi_traces import digest
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    manifest=json.loads((result/'MANIFEST.json').read_text())
    assert all(digest(result/k)==v for k,v in manifest.items())
    rows=[json.loads(s) for s in (result/'ROWS.jsonl').read_text().splitlines()]
    source=torch.load(result/'LOGITS.pt',map_location='cpu',weights_only=True)
    indices=sorted(range(len(rows)),key=lambda i:(-rows[i]['tv_base_cached'],i))[:3]
    out.mkdir(parents=True,exist_ok=False)
    (out/'PROTOCOL.json').write_text(json.dumps(dict(selection='Three largest cached/full TV discrepancies, post-outcome apparatus diagnosis',
        indices=indices,source_manifest_sha256=digest(result/'MANIFEST.json'),script_sha256=digest(Path(__file__)),
        scope='Base-only execution schedule and arithmetic precision; does not repair or replace failed efficacy assay',
        estimate='1-3 GPU minutes; no arbitrary cutoff'),indent=2))
    start=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval();model.requires_grad_(False)
    saved=[]
    with torch.inference_mode():
        for i in indices:
            r=rows[i];ids=r['inputs']['base']['ids'];prompt_len=len(ids)-r['length']
            x=torch.tensor([ids],device='cuda')
            full=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).logits[0,-1].float().cpu()
            initial=x[:,:prompt_len]
            response=model(input_ids=initial,attention_mask=torch.ones_like(initial),use_cache=True,logits_to_keep=1)
            cache=response.past_key_values
            for t in range(prompt_len,len(ids)):
                response=model(input_ids=x[:,t:t+1],attention_mask=torch.ones_like(x[:,:t+1]),
                    past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=response.past_key_values
            cached=response.logits[0,-1].float().cpu()
            saved.append(dict(full_bf16=full,cached_bf16=cached,original_full=source[i]['base'],original_cached=source[i]['cached_base']))
            del cache,response
        model.float()
        for i,values in zip(indices,saved):
            x=torch.tensor([rows[i]['inputs']['base']['ids']],device='cuda')
            values['full_fp32']=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).logits[0,-1].float().cpu()
    pairs=[('cached_bf16','original_cached'),('full_bf16','original_full'),('full_bf16','cached_bf16'),
           ('full_fp32','cached_bf16'),('full_fp32','full_bf16')]
    report=[]
    for i,v in zip(indices,saved):
        assert all(torch.isfinite(x).all() for x in v.values())
        report.append(dict(base=rows[i]['base'],kind=rows[i]['kind'],length=rows[i]['length'],
            tv={a+'__'+b:float((v[a].softmax(-1)-v[b].softmax(-1)).abs().sum()/2) for a,b in pairs}))
    torch.save(saved,out/'LOGITS.pt')
    summary=dict(rows=report,seconds=time.monotonic()-start)
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    (out/'MANIFEST.json').write_text(json.dumps({p.name:digest(p) for p in out.iterdir() if p.is_file()},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['result','snapshot','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.result,a.snapshot,a.out)
