"""Public-text model-generated drafts, then suffix disclosure and verification.

Selection uses no model scores. This is token reconstruction, not factual QA.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM

from . import cache_verification_bert as backend

ARMS = ["draft", "fresh", "plain", "stale_uncorrected", "stale_diagonal",
        "refreshed_diagonal", "neutral_diagonal", "last_diagonal"]


def select_cases(texts, tokenizer, config):
    pool, seen = [], set()
    article = "preamble"
    for row_index, text in enumerate(texts):
        text = text.strip()
        if re.fullmatch(r"= [^=]+ =", text):
            article = text
            continue
        if not text or text.startswith("="):
            continue
        for index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", text)):
            sentence = sentence.strip()
            digest = hashlib.sha256(sentence.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            ids = tokenizer.encode(sentence, add_special_tokens=True)
            if not config["min_tokens"] <= len(ids) <= config["max_tokens"]:
                continue
            tokens = tokenizer.convert_ids_to_tokens(ids)
            side = config["min_side_tokens"]
            candidates = [p for p in range(side+1, len(ids)-side-1)
                          if tokens[p].isascii() and tokens[p].isalpha()
                          and len(tokens[p]) >= 3 and not tokens[p+1].startswith("##")]
            if not candidates:
                continue
            pos = min(candidates, key=lambda p: (abs(p-(len(ids)-1)/2), p))
            key = hashlib.sha256(f'{config["selection_seed"]}:{digest}'.encode()).hexdigest()
            pool.append({"id": f"r{row_index}s{index}", "article": article, "text": sentence,
                         "input_ids": ids, "position": pos, "gold_id": ids[pos], "selection_key": key})
    selected, counts = [], Counter()
    for case in sorted(pool, key=lambda c: c["selection_key"]):
        if counts[case["article"]] >= config["max_per_article"]:
            continue
        selected.append(case)
        counts[case["article"]] += 1
        if len(selected) == config["cases"]:
            break
    if len(selected) != config["cases"]:
        raise ValueError(f"Only {len(selected)} eligible cases across articles")
    return selected


def views(case, mask_id, sep_id):
    full = torch.tensor([case["input_ids"]], dtype=torch.long)
    full[0, case["position"]] = mask_id
    prefix = torch.cat((full[:, :case["position"]+1], torch.tensor([[sep_id]])), dim=1)
    return full, prefix


def summarize(logits, cases, config):
    assert logits.shape[:2] == (len(cases), len(ARMS))
    assert len(cases) == config["cases"] and np.isfinite(logits).all()
    gold = np.array([c["gold_id"] for c in cases])
    pred = logits.argmax(-1)
    correct = pred == gold[:, None]
    initially_wrong = ~correct[:, 0]
    recoverable = initially_wrong & correct[:, 1]
    initial_correct = correct[:, 0]
    info = {"n": len(cases), "articles": len(set(c["article"] for c in cases)),
            "initially_wrong": int(initially_wrong.sum()), "recoverable_by_fresh": int(recoverable.sum()), "arms": {}}
    for j, arm in enumerate(ARMS):
        info["arms"][arm] = {"correct": int(correct[:, j].sum()),
                              "recovered_initial_errors": int((correct[:, j] & initially_wrong).sum()),
                              "lost_initial_correct": int((~correct[:, j] & initial_correct).sum()),
                              "correct_on_fresh_recoverable": int((correct[:, j] & recoverable).sum()),
                              "retained_wrong_draft_on_fresh_recoverable": int(((pred[:, j] == pred[:, 0]) & recoverable).sum())}
    info["plain_max_error"] = float(np.abs(logits[:, 1]-logits[:, 2]).max())
    info["last_only_max_error"] = float(np.abs(logits[:, 1]-logits[:, 7]).max())
    assert info["plain_max_error"] < config["equivalence_tolerance"]
    assert info["last_only_max_error"] < config["equivalence_tolerance"]
    assert np.array_equal(pred[:, 1], pred[:, 2])
    delta = correct[:, 1].astype(int)-correct[:, 4].astype(int)
    info["fresh_over_stale_diagonal_net_correct"] = int(delta.sum())
    info["fresh_only_correct"] = int((delta == 1).sum())
    info["stale_diagonal_only_correct"] = int((delta == -1).sum())
    # Article bootstrap, preserving all paired cases from each sampled article.
    article_groups = [np.array([i for i,c in enumerate(cases) if c["article"] == a])
                      for a in sorted(set(c["article"] for c in cases))]
    rng = np.random.default_rng(77101)
    means = []
    for _ in range(2000):
        ix = np.concatenate([article_groups[i] for i in rng.integers(len(article_groups), size=len(article_groups))])
        means.append(float(delta[ix].mean()))
    info["article_bootstrap_fresh_minus_stale_accuracy_95"] = np.quantile(means, [.025,.975]).tolist()
    if recoverable.sum() < config["minimum_recoverable_drafts"]:
        decision = "INSUFFICIENT_RECOVERABLE_DRAFTS_NO_PRACTICAL_GATE_DECISION"
    elif delta.sum() < config["minimum_net_clean_gain_cases"]:
        decision = "NO_USEFUL_CLEAN_VERIFICATION_GAIN_HOLD_EXPANSION"
    else:
        decision = "DEVELOPMENTAL_CLEAN_GAIN_INVESTIGATE_NOT_PAPER_GO"
    info["decision"] = decision
    return info


def prepare(config_path, parquet, model_path, root):
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text())
    assert backend.sha(parquet) == config["dataset_sha256"]
    assert model_path.name == config["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    texts = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
    cases = select_cases(texts, tokenizer, config)
    (root/"config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    (root/"cases.json").write_text(json.dumps(cases, indent=2)+"\n", encoding="utf-8")
    manifest = {"source_sha256": backend.sha(__file__), "data_path": str(parquet.resolve()),
                "dataset_sha256": backend.sha(parquet), "model_path": str(model_path.resolve()),
                "files": {name: backend.sha(root/name) for name in ("config.json", "cases.json")}}
    (root/"MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return {"cases": len(cases), "articles": len(set(c["article"] for c in cases)), "manifest_sha256": backend.sha(root/"MANIFEST.json")}


def verify_prepared(prepared):
    m = json.loads((prepared/"MANIFEST.json").read_text())
    assert m["source_sha256"] == backend.sha(__file__)
    assert set(m["files"]) == {"cases.json", "config.json"}
    assert all(backend.sha(prepared/name) == digest for name,digest in m["files"].items())
    assert backend.sha(m["data_path"]) == m["dataset_sha256"]
    return m


def run(prepared, root):
    m = verify_prepared(prepared)
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads((prepared/"config.json").read_text())
    cases = json.loads((prepared/"cases.json").read_text())
    model_path = Path(m["model_path"])
    torch.set_num_threads(2); torch.manual_seed(77100); torch.use_deterministic_algorithms(True)
    model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=True, attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    output, calls = [], 0
    start = time.perf_counter()
    with torch.inference_mode():
        for c in cases:
            full, prefix = views(c, tokenizer.mask_token_id, tokenizer.sep_token_id)
            pos = c["position"]
            draft_logits = model(prefix).logits[0,pos].float().numpy().copy(); calls += 1
            draft = int(draft_logits.argmax())
            fresh = model(full).logits[0,pos].float().numpy().copy(); calls += 1
            result = {"draft": draft_logits, "fresh": fresh}
            with backend.Instrument(model) as inst:
                inst.position = pos
                result["plain"] = model(full).logits[0,pos].float().numpy().copy(); calls += 1
                for cache_type, input_ids in (("stale",prefix.clone()),("refreshed",full.clone()),("neutral",prefix.clone())):
                    if cache_type != "neutral":
                        input_ids[0,pos] = draft
                    inst.mode, inst.saved = "collect", []
                    model(input_ids); calls += 1
                    inst.cache = inst.saved
                    for mode in (("uncorrected", "diagonal", "last") if cache_type == "stale" else ("diagonal",)):
                        inst.mode = mode
                        name = "last_diagonal" if mode == "last" else cache_type+"_"+mode
                        result[name] = model(full).logits[0,pos].float().numpy().copy(); calls += 1
            output.append(np.stack([result[arm] for arm in ARMS]))
            with (root/"progress.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id":c["id"], "draft_id":draft, "predictions": {arm:int(result[arm].argmax()) for arm in ARMS}})+"\n")
    logits = np.stack(output)
    np.save(root/"logits.npy", logits, allow_pickle=False)
    summary = summarize(logits,cases,config)
    runtime = {"forward_calls": calls, "inference_seconds":time.perf_counter()-start,
               "device":"cpu", "torch":torch.__version__, "transformers":transformers.__version__,
               "numpy":np.__version__, "source_sha256":backend.sha(__file__),
               "backend_sha256":backend.sha(backend.__file__), "prepared":str(prepared.resolve()),
               "prepared_manifest_sha256":backend.sha(prepared/"MANIFEST.json"), "arms":ARMS,
               "model_path":str(model_path.resolve()),
               "model_files":{p.name:backend.sha(p) for p in model_path.iterdir() if p.is_file()}}
    for name,obj in (("summary.json",summary),("runtime.json",runtime)):
        (root/name).write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    manifest = {p.name:backend.sha(p) for p in root.iterdir() if p.is_file()}
    (root/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return summary


def verify(root):
    m = json.loads((root/"MANIFEST.json").read_text())
    assert set(m) == {"progress.jsonl","logits.npy","summary.json","runtime.json"}
    assert all(backend.sha(root/n)==v for n,v in m.items())
    runtime = json.loads((root/"runtime.json").read_text())
    assert runtime["source_sha256"] == backend.sha(__file__)
    assert runtime["backend_sha256"] == backend.sha(backend.__file__)
    prepared = Path(runtime["prepared"])
    assert runtime["prepared_manifest_sha256"] == backend.sha(prepared/"MANIFEST.json")
    verify_prepared(prepared)
    assert all(backend.sha(Path(runtime["model_path"])/n)==v for n,v in runtime["model_files"].items())
    cases = json.loads((prepared/"cases.json").read_text())
    config = json.loads((prepared/"config.json").read_text())
    assert runtime["forward_calls"] == 11*len(cases)
    z = np.load(root/"logits.npy", allow_pickle=False)
    progress = [json.loads(line) for line in (root/"progress.jsonl").read_text().splitlines()]
    assert [p["id"] for p in progress] == [c["id"] for c in cases]
    for i,p in enumerate(progress):
        assert p["draft_id"] == int(z[i,0].argmax())
        assert p["predictions"] == {a:int(z[i,j].argmax()) for j,a in enumerate(ARMS)}
    summary = summarize(z,cases,config)
    assert summary == json.loads((root/"summary.json").read_text())
    return {"verified":True,"scope":"bytes_and_metric_recomputation_not_forward_replay",
            "manifest_sha256":backend.sha(root/"MANIFEST.json"),"summary":summary}


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["prepare","run","verify"])
    p.add_argument("root",type=Path)
    p.add_argument("--prepared",type=Path)
    p.add_argument("--config",type=Path,default=Path("configs/cache_natural_drafts_v1.json"))
    p.add_argument("--parquet",type=Path)
    p.add_argument("--model-path",type=Path)
    a=p.parse_args()
    result = prepare(a.config,a.parquet,a.model_path,a.root) if a.command=="prepare" else run(a.prepared,a.root) if a.command=="run" else verify(a.root)
    print(json.dumps(result,indent=2))
