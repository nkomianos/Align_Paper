"""Public pinned adapter preparation and CPU tensor audit; no model execution."""
import hashlib
import json
from pathlib import Path
import urllib.request
import torch
from safetensors import safe_open


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    root=Path('artifacts/opsd_checkpoint_search_20260910/qwen3-8b-thinking').resolve()
    meta=json.loads((root/'MODEL_METADATA.json').read_text())
    assert meta['sha']=='16ba69e32700ddaeb03ec0a2cf717dfca89296b9'
    entry=next(s for s in meta['siblings'] if s['rfilename']=='adapter_model.safetensors')
    dest=root/'adapter_model.safetensors'
    if not dest.exists():
        partial=root/'adapter_model.safetensors.part'
        assert partial.parent==dest.parent==root
        with urllib.request.urlopen(f'https://huggingface.co/u-opsd/qwen3-8b-thinking/resolve/{meta["sha"]}/adapter_model.safetensors',timeout=60) as response,partial.open('xb') as f:
            for chunk in iter(lambda:response.read(1024*1024),b''):f.write(chunk)
        assert sha(partial)==entry['lfs']['sha256'] and partial.stat().st_size==entry['size']
        partial.rename(dest)
    assert sha(dest)==entry['lfs']['sha256']
    config=json.loads((root/'adapter_config.json').read_text())
    assert config['base_model_name_or_path']=='Qwen/Qwen3-8B' and config['r']==64
    records=[];pairs={}
    with safe_open(dest,framework='pt',device='cpu') as reader:
        for name in reader.keys():
            tensor=reader.get_tensor(name)
            assert tensor.ndim==2 and torch.isfinite(tensor).all()
            assert name.endswith(('.lora_A.weight','.lora_B.weight'))
            stem,kind=name.rsplit('.lora_',1);pairs.setdefault(stem,{})[kind[0]]=tensor.shape
            records.append(dict(name=name,shape=list(tensor.shape),dtype=str(tensor.dtype),nonzero=int(torch.count_nonzero(tensor))))
    assert all(set(p)=={'A','B'} and p['A'][0]==p['B'][1]==64 for p in pairs.values())
    assert all(r['nonzero']>0 for r in records)
    report=dict(model=meta['id'],revision=meta['sha'],weights_sha256=sha(dest),bytes=dest.stat().st_size,
        tensors=len(records),module_pairs=len(pairs),rank=64,finite=True,all_tensors_nonzero=True,
        trainer_global_step=json.loads((root/'trainer_state.json').read_text())['global_step'],
        historical_base_revision=config.get('revision'),records=records,
        scope='Adapter integrity/shape preparation only; base historical revision unpinned; no behavior or paper qualification',
        script_sha256=sha(Path(__file__)))
    with (root/'TENSOR_AUDIT.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
