import json
import pytest
from scripts.stage_sdpo_single_profile import prepare,INSTRUCTION
from interaction_sprint.sdpo_single_profile_reproduction import check_data,json_safe,EXPLICIT_PREFERENCE

def test_data_source_and_post_dedup(tmp_path):
    rows=[]
    for i in range(90):
        rows.append(dict(info=dict(post=f"Train content number{i}",id=f"post{i}"),split="train"))
    for i in range(40):
        rows.append(dict(info=dict(post=f"Validation content number{i}",id=f"valid{i}"),split="valid1"))
    rows.append(dict(info=dict(post="An edited version of traincontent",id="post0"),split="valid2"))
    rows.append(dict(info=dict(article="Article fallback",id="extra"),split="train"))
    rows.append(dict(info=dict(post="X"*1024,id="overlong"),split="train"))
    path=tmp_path/"data.json";path.write_text("\n".join(json.dumps(r) for r in rows))
    result,counts=prepare([path]);check_data(result)
    assert counts==dict(train=91,validation=40)
    assert all(len(r["prompt"])<1024 for part in result.values() for r in part)
    rows.append(dict(info=dict(post="Never use testdata",id="test"),split="test"))
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match="unexpected upstream split"):prepare([path])

def test_check_data_refuses_hidden_post_identity_overlap():
    data={s:[dict(id=f"{s}{i}",source_post_id=f"{s}{i}",source_split="valid1" if s=="eval" else "train") for i in range(n)]
          for s,n in (("train",64),("eval",32),("calibration",16))}
    check_data(data)
    data["eval"][0]["source_post_id"]=data["train"][0]["source_post_id"]
    with pytest.raises(ValueError,match="duplicate source"):check_data(data)

def test_generation_metadata_and_oracle_identity():
    import torch
    assert json_safe({"dtype":torch.bfloat16,"ids":{2,1}})=={"dtype":"torch.bfloat16","ids":[1,2]}
    assert "you are a user" not in EXPLICIT_PREFERENCE.lower()
    assert "user prefers" in EXPLICIT_PREFERENCE
