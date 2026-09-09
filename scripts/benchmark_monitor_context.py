"""Unlabeled H100 context-length qualification; emits no scientific scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allocation-end',required=True)
    a=p.parse_args()
    end=datetime.fromisoformat(a.allocation_end.replace('Z','+00:00'))
    if (end-datetime.now(timezone.utc)).total_seconds()<1800:
        raise RuntimeError('less than 30-minute admission margin remains')
    a.out.mkdir(parents=True,exist_ok=False)
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    if a.snapshot.name!='c202236235762e1c871ad0ccb60c8ee5ba337b9a':
        raise ValueError('wrong model revision')
    write=lambda name,value:(a.out/name).write_text(json.dumps(value,indent=2),encoding='utf8')
    write('PROTOCOL.json',{'lengths':[4096,8192,16384,32768],
        'purpose':'Unlabeled inference fit/throughput qualification, not monitoring accuracy.',
        'synthetic_text':'This is an unlabeled compute qualification input. No agent behavior is being evaluated. ',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'snapshot':str(a.snapshot),'allocation_end':a.allocation_end,
        'planned_minutes':15,'admission_margin_minutes':30,'runtime_kill':False})
    tokenizer=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True)
    start=time.monotonic()
    model=Qwen3_5ForCausalLM.from_pretrained(a.snapshot,local_files_only=True,
        dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',use_kernels=False).eval()
    model.requires_grad_(False)
    write('ENVIRONMENT.json',{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
        'load_seconds':time.monotonic()-start})
    unit=tokenizer.encode('This is an unlabeled compute qualification input. No agent behavior is being evaluated. ',add_special_tokens=False)
    records=[]
    for n in (4096,8192,16384,32768):
        ids=(unit*((n+len(unit)-1)//len(unit)))[:n]
        x=torch.tensor([ids],device='cuda')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        try:
            with torch.inference_mode():
                logits=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits
            torch.cuda.synchronize()
            record={'tokens':n,'seconds':time.monotonic()-start,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'finite_logits':bool(torch.isfinite(logits).all()),
                'input_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
            del logits,x
        except torch.cuda.OutOfMemoryError:
            record={'tokens':n,'oom':True,'seconds':time.monotonic()-start}
            records.append(record);write('RESULT.json',{'records':records,'paper_green_light':False})
            print(json.dumps(record),flush=True);break
        records.append(record);write('RESULT.json',{'records':records,'paper_green_light':False})
        print(json.dumps(record),flush=True)
        if not record['finite_logits']:break


if __name__=='__main__':main()
