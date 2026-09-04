import hashlib
import json
import pytest
import torch

from scripts.verify_sdpo_frozen_forward import manifest_check, selected_cases, offsets, checkpoint_receipt, sha


def test_manifest_detects_tampering_and_unlisted_files(tmp_path):
    path=tmp_path/'evidence.txt'; path.write_text('original')
    (tmp_path/'MANIFEST.json').write_text(json.dumps({'evidence.txt':sha(path)}))
    manifest_check(tmp_path)
    path.write_text('modified')
    with pytest.raises(ValueError,match='checksum'): manifest_check(tmp_path)
    path.write_text('original'); (tmp_path/'extra.txt').write_text('extra')
    with pytest.raises(ValueError,match='inventory'): manifest_check(tmp_path)


def test_selector_is_fixed_not_metric_ranked():
    data=[dict(id=f'{style}-{i}',preference=style,score={'joint':style=='plain'})
          for style in ('plain','json','table','bullets') for i in range(8)]
    chosen=selected_cases(list(reversed(data)))
    assert [r['id'] for r in chosen] == [f'plain-{i}' for i in range(8)]+['json-0','json-1','json-2','table-0','table-1','table-2','bullets-0','bullets-1']
    data[0]['score']['joint']=False
    with pytest.raises(ValueError,match='correct stratum'): selected_cases(data)


def test_offsets_skip_initial_and_special_tokens():
    class Tokenizer:
        all_special_ids=[8]
        def decode(self,x,**kwargs): return {1:'{',2:'asset',3:':',8:'<|im_end|>'}[x[0]]
    assert offsets(Tokenizer(),[1,8,2,3])==[0,3]
    with pytest.raises(ValueError): offsets(Tokenizer(),[1,8,2])


def test_checkpoint_loaded_byte_receipts(tmp_path):
    checkpoint=tmp_path/'adapter.pt'
    state={'q.a':torch.tensor([[.1,.2]],dtype=torch.float32),'q.b':torch.zeros((2,1))}
    torch.save(state,checkpoint)
    receipt={'checkpoint_sha256':sha(checkpoint),'exact_equal':True,
             'tensor_sha256':{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in state.items()},
             'tensor_shapes':{k:list(v.shape) for k,v in state.items()}}
    assert checkpoint_receipt(receipt,checkpoint)==2
    receipt['tensor_sha256']['q.a']='0'*64
    with pytest.raises(ValueError,match='tensor receipt'): checkpoint_receipt(receipt,checkpoint)
