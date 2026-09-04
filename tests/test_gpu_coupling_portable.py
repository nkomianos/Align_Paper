import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from interaction_sprint.byte_clock_coupling import digest


def test_explicit_maps_and_remote_weight_attestation(tmp_path,monkeypatch):
    path=Path(__file__).parents[1]/'scripts/verify_gpu_coupling_portable.py'
    spec=importlib.util.spec_from_file_location('portable_for_test',path)
    portable=importlib.util.module_from_spec(spec);spec.loader.exec_module(portable)
    root=tmp_path/'evidence';root.mkdir();local=tmp_path/'snapshot';local.mkdir()
    (local/'source.py').write_text('immutable source')
    (root/'MANIFEST.json').write_text('{}')
    model={'name':'model','path':'/remote/model'}
    weight='a'*64
    (root/'FROZEN.json').write_text(json.dumps({'config':{'models':[model]},'model_files':{'model':{'model.safetensors':weight}}}))
    mapping=tmp_path/'map.json';mapping.write_text(json.dumps({'/remote/model':str(local),'/remote/source.py':str(local/'source.py')}))
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps({'/remote/model/model.safetensors':weight}))
    module=SimpleNamespace()
    def analyze(actual_root,out):
        assert actual_root==root
        assert module.Path('/remote/model/config.json')==local/'config.json'
        assert module.digest('/remote/source.py')==digest(local/'source.py')
        assert module.digest(local/'model.safetensors')==weight
        out.write_text('{}')
    module.analyze=analyze
    fake_spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None))
    monkeypatch.setattr(portable.importlib.util,'spec_from_file_location',lambda *a:fake_spec)
    monkeypatch.setattr(portable.importlib.util,'module_from_spec',lambda *a:module)
    output=tmp_path/'report.json'
    portable.run(root,output,mapping,receipt)
    report=json.loads((tmp_path/'report.json.portability.json').read_text())
    assert report['weights_remote_checksum_attested_not_locally_rehashed']==['/remote/model/model.safetensors']
    with pytest.raises(FileExistsError):portable.run(root,output,mapping,receipt)
    with pytest.raises(ValueError):portable.run(root,root/'bad.json',mapping,receipt)
    receipt.write_text('{}')
    with pytest.raises(AssertionError):portable.run(root,tmp_path/'new.json',mapping,receipt)
