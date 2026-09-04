"""Post-hoc DEV qualification of prefix formatting; no cache-method comparison."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from interaction_sprint.cache_verification_bert import sha


def main(prepared, root):
    root.mkdir(parents=True, exist_ok=False)
    manifest=json.loads((prepared/"MANIFEST.json").read_text())
    assert sha(prepared/"cases.json")==manifest["files"]["cases.json"]
    cases=json.loads((prepared/"cases.json").read_text())[:24]
    model_path=Path(manifest["model_path"])
    torch.set_num_threads(2)
    model=AutoModelForMaskedLM.from_pretrained(model_path,local_files_only=True,attn_implementation="eager").eval()
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    rows=[]
    with torch.inference_mode():
        for case in cases:
            for right in (0,3):
                pos=case["position"]
                ids=case["input_ids"][:pos+right+1]+[tokenizer.sep_token_id]
                ids[pos]=tokenizer.mask_token_id
                logits=model(torch.tensor([ids])).logits[0,pos]
                pred=int(logits.argmax())
                token=tokenizer.convert_ids_to_tokens(pred)
                rows.append({"id":case["id"],"right_tokens":right,"input_ids":ids,
                             "prediction_id":pred,"token":token,"correct":pred==case["gold_id"],
                             "lexical":token.isascii() and token.isalpha(),
                             "probability":float(logits.softmax(-1)[pred])})
    summary={str(right):{"n":24,"correct":sum(r["correct"] for r in rows if r["right_tokens"]==right),
                         "lexical":sum(r["lexical"] for r in rows if r["right_tokens"]==right)} for right in (0,3)}
    evidence={"scope":"posthoc_dev_format_qualification_no_method_selection",
              "source_sha256":sha(__file__),"prepared_manifest_sha256":sha(prepared/"MANIFEST.json"),
              "forward_calls":48,"model_files":{p.name:sha(p) for p in model_path.iterdir() if p.is_file()},
              "model_path":str(model_path),"summary":summary,"rows":rows}
    (root/"result.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    (root/"MANIFEST.json").write_text(json.dumps({"result.json":sha(root/"result.json")},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"summary":summary,"manifest_sha256":sha(root/"MANIFEST.json")},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("--prepared",type=Path,default=Path("artifacts/cache_natural_drafts_prepared_v1"))
    a=p.parse_args()
    main(a.prepared,a.root)
