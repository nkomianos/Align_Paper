"""Download pinned public reproduction assets; no model loading or generation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import snapshot_download

MODEL_REV = "b968826d9c46dd6066d109eabc6255188de91218"
DATA_REV = "b8f7d168b6f4e95b2a92e84768bd6c955bed2f29"
BATCHES = [f"batch{i}.json" for i in (3,4,5,10,11,12,13,14,15,16,17,18,19,20,22,6,7,8,9)] + ["batch0_cnndm.json","cnndm0.json","cnndm2.json","edit_b2_eval_test.json"]
INSTRUCTION = "Write a brief summary of the text that begins with 'TL;DR:'.\n\n"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(files):
    pools={"train":{},"validation":{}}
    for path in files:
        for line in Path(path).read_text().splitlines():
            row=json.loads(line); info=row.get("info") or {}
            post=(info.get("post") or info.get("article") or "").strip()
            prompt=INSTRUCTION+post
            if not post or len(prompt)>=1024:
                continue
            key=hashlib.sha1(" ".join(prompt.split()).lower().encode()).hexdigest()
            split=row["split"]
            if split not in ("train","valid1","valid2"):
                raise ValueError("unexpected upstream split")
            part="train" if split=="train" else "validation"
            pools[part].setdefault(key,dict(id=key,prompt=prompt,source_post_id=info.get("id"),source_split=split))
    # Additional conservative source-post grouping beyond upstream text dedup.
    seen_ids=set()
    for part in ("train","validation"):
        kept={}
        for key,row in pools[part].items():
            post_id=row["source_post_id"]
            if key in pools["train"] and part=="validation":
                continue
            if post_id and post_id in seen_ids:
                continue
            if post_id:
                seen_ids.add(post_id)
            kept[key]=row
        pools[part]=kept
    def ordered(part):
        return sorted(pools[part].values(), key=lambda r:hashlib.sha256(("9047801/"+r["id"]).encode()).hexdigest())
    train,valid=ordered("train"),ordered("validation")
    if len(train)<80 or len(valid)<32:
        raise ValueError("insufficient natural prompts")
    return dict(calibration=train[:16],train=train[16:80],eval=valid[:32]),{k:len(v) for k,v in pools.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--reuse-raw",type=Path)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=False); raw=a.root/"raw";raw.mkdir()
    urls={"dataset_source.py":f"https://huggingface.co/datasets/openai/summarize_from_feedback/resolve/{DATA_REV}/summarize_from_feedback.py"}
    urls.update({f:"https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset/comparisons/"+f for f in BATCHES})
    def download(pair):
        name,url=pair
        if a.reuse_raw:
            import shutil
            shutil.copyfile(a.reuse_raw/name,raw/name)
        else:
            urllib.request.urlretrieve(url,raw/name)
        return name,dict(url=url,sha256=sha(raw/name),bytes=(raw/name).stat().st_size)
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes=dict(pool.map(download,urls.items()))
    data,counts=prepare([raw/f for f in BATCHES])
    for split,rows in data.items():
        (a.root/(split+".json")).write_text(json.dumps(rows,indent=2),encoding="utf-8")
    (a.root/"data_manifest.json").write_text(json.dumps(dict(dataset_revision=DATA_REV,files=hashes,eligible_counts=counts,
        selection="normalized-prompt and source-post-ID dedup; original <1024 character prompt filter; hash9047801 ordering; first16train calibration,next64train;first32validation",splits={s:sha(a.root/(s+".json")) for s in data}),indent=2))
    print("Natural data prepared",counts,flush=True)
    path=snapshot_download("Qwen/Qwen3-8B",revision=MODEL_REV,cache_dir=a.cache,
        allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.jinja","LICENSE","README.md"],max_workers=4)
    (a.root/"model_manifest.json").write_text(json.dumps(dict(model="Qwen/Qwen3-8B",revision=MODEL_REV,path=path,
        sha256={str(f.relative_to(path)):sha(f) for f in Path(path).rglob("*") if f.is_file()}),indent=2))
    print("All public assets staged",path,flush=True)


if __name__=="__main__":main()
