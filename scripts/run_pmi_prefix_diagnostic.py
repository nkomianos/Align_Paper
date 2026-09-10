"""Four-context, fixed-prefix forward-only DEV diagnostic. No model training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(data,snapshot,out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pmi_target_controls import corrected_target,distribution_kl
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied; no launch')
    rows=json.loads((data/'INPUTS.json').read_text(encoding='utf-8'))
    plan=json.loads((data/'PLAN.json').read_text(encoding='utf-8'))
    assert len(rows)==plan['rows']==48
    # Authenticate exact weights against the original generating model's record.
    expected=plan['model_manifest']['weights']
    actual={p.name:sha(p) for p in snapshot.glob('*.safetensors')}
    assert actual==expected, 'Model weights differ from saved-prefix model'
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    provenance=dict(plan_sha256=sha(data/'PLAN.json'),inputs_sha256=sha(data/'INPUTS.json'),
        runner_sha256=sha(Path(__file__)),helper_sha256=sha(Path(__file__).with_name('pmi_target_controls.py')),
        model_weights=actual,tokenizer_files={p.name:sha(p) for p in snapshot.iterdir() if p.is_file() and p.suffix in ['.json','.txt']},
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        scope=plan['scope'],rendering='Native nonthinking chat followed by the same32originalprefix tokenIDs; raw logits retained',
        instruction='Continue the partial mathematical solution. Use any supplied problem and reference. Do not discuss missing information.')
    (out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
    logits_saved=[]
    with (out/'ROWS.jsonl').open('x',encoding='utf-8') as output, torch.inference_mode():
        for i,row in enumerate(rows):
            contexts={'teacher':f"Problem:\n{row['question']}\nReference solution:\n{row['reference']}",
                      'reference':f"Reference solution:\n{row['reference']}",
                      'base':f"Problem:\n{row['question']}",'unconditional':'Continue the solution.'}
            logits={}; inputs={}
            for arm,content in contexts.items():
                rendered=tok.apply_chat_template([{'role':'system','content':provenance['instruction']},
                    {'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                prefix=row['prefix_ids']; assert len(prefix)==32 and all(0<=v<len(tok) for v in prefix)
                ids=tok.encode(rendered,add_special_tokens=False)+prefix
                assert ids[-32:]==prefix
                if len(ids)>8192: raise ValueError('Context exceeds frozen8192token limit')
                encoded=torch.tensor([ids],device='cuda')
                values=model(input_ids=encoded,attention_mask=torch.ones_like(encoded),use_cache=False).logits[0,-1].float()
                assert torch.isfinite(values).all()
                logits[arm]=values; inputs[arm]={'ids':ids,'rendered':rendered}
            purified=corrected_target(logits['base'],logits['teacher'],logits['reference'])
            control=corrected_target(logits['base'],logits['base'],logits['unconditional'])
            base=logits['base'].log_softmax(-1)
            result=dict(base=row['base'],index=row['index'],inputs=inputs,
                kl_purified_to_question_control=float(distribution_kl(purified,control)),
                kl_purified_to_base=float(distribution_kl(purified,base)),
                tv_purified_question_control=float((purified.exp()-control.exp()).abs().sum()/2),
                entropy_base=float(-(base.exp()*base).sum()),
                entropy_purified=float(-(purified.exp()*purified).sum()),
                entropy_question_control=float(-(control.exp()*control).sum()))
            output.write(json.dumps(result)+'\n');output.flush()
            logits_saved.append({k:v.cpu() for k,v in logits.items()})
            print(json.dumps({'completed':i+1,'seconds':time.monotonic()-started}),flush=True)
    # Local tensors only, produced by this runner; no external pickle loaded.
    torch.save(logits_saved,out/'LOGITS.pt')
    (out/'TIMING.json').write_text(json.dumps({'seconds':time.monotonic()-started}))
    (out/'MANIFEST.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.data,a.snapshot,a.out)
