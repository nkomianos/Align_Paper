"""Frozen released-C2C engineering baseline on public validation examples.

No training; no private-answer-key reads; no update-effect or novelty claim.
Invoke only after the active scientific process has exited and been secured.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SPEC = {"scope": "C2C_RELEASED_BASELINE_DEV_NOT_UPDATE_STUDY", "cases": 128,
        "cases_sha256": "a279178264c7b2e66c65d852193723925b42482d532ef6dc568a5bf3d0ce046b",
        "upstream_commit": "113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a",
        "sender": "Qwen/Qwen3-4B", "sender_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "receiver": "Qwen/Qwen3-0.6B", "receiver_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "fuser_revision": "f01fc3258b305e280e04c7238f4f2cf31b7dc70d",
        "arms": ["receiver", "sender", "c2c", "disabled_fuser"],
        "max_new_tokens": 64, "max_input_tokens": 2048, "thinking": False,
        "torch": "2.7.1+cu128", "transformers": "4.52.4", "upstream_torch_match": False}


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def verified_assets(root):
    manifest = json.loads((root / "MANIFEST.json").read_text())["files"]
    for name, digest in manifest.items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("asset evidence checksum mismatch")
    assets = json.loads((root / "COMPLETE.json").read_text())["assets"]
    if assets["receiver"]["revision"] != SPEC["receiver_revision"] or assets["fuser"]["revision"] != SPEC["fuser_revision"]:
        raise ValueError("wrong staged assets")
    for item in assets.values():
        for row in item["files"]:
            path = Path(item["snapshot"]) / row["name"]
            kind = row["digest_kind"]
            if kind not in ("sha256", "git_blob_sha1"):
                raise ValueError("unknown checksum kind")
            h = hashlib.new("sha256" if kind == "sha256" else "sha1")
            if kind == "git_blob_sha1":
                h.update(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    h.update(chunk)
            if h.hexdigest() != row["digest"] or path.stat().st_size != row["bytes"]:
                raise ValueError("staged asset changed")
    return assets


def run(upstream, assets_root, cases_path, output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        write(output / "spec.json", SPEC)
        (output / "runner.py").write_bytes(Path(__file__).read_bytes())
        commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
        if commit != SPEC["upstream_commit"] or dirty:
            raise ValueError("upstream revision mismatch or tracked edits")
        raw_cases = cases_path.read_bytes()
        if hashlib.sha256(raw_cases).hexdigest() != SPEC["cases_sha256"]:
            raise ValueError("case digest mismatch")
        cases = json.loads(raw_cases)
        if len(cases) != SPEC["cases"]:
            raise ValueError("wrong case count")
        (output / "cases.json").write_bytes(raw_cases)
        assets = verified_assets(assets_root)
        write(output / "assets.json", assets)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if torch.__version__ != SPEC["torch"] or transformers.__version__ != SPEC["transformers"]:
            raise ValueError("dependency mismatch")
        torch.manual_seed(20260904)
        sys.path.insert(0, str(upstream.resolve()))
        from rosetta.model.projector import load_projector
        from rosetta.model.wrapper import RosettaModel
        from rosetta.utils.evaluate import build_prompt
        tokenizers, models = {}, {}
        for role in ("receiver", "sender"):
            tokenizers[role] = AutoTokenizer.from_pretrained(SPEC[role], revision=SPEC[role + "_revision"], local_files_only=True, trust_remote_code=False)
            models[role] = AutoModelForCausalLM.from_pretrained(SPEC[role], revision=SPEC[role + "_revision"],
                local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
                device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
            if models[role].config._commit_hash != SPEC[role + "_revision"]:
                raise ValueError("resolved model revision mismatch")
        fuser_dir = Path(assets["fuser"]["snapshot"]) / "qwen3_0.6b+qwen3_4b_Fuser/final"
        projectors = []
        for index in range(28):
            proj = load_projector(str(fuser_dir / f"projector_{index}.json"))
            state = torch.load(fuser_dir / f"projector_{index}.pt", map_location="cpu", weights_only=True)
            proj.load_state_dict(state, strict=True)
            projectors.append(proj)
        fused = RosettaModel([models["receiver"], models["sender"]], base_model_idx=0, projector_list=projectors).to("cuda:0").eval()
        fused.load_projector_config(str(fuser_dir / "projector_config.json"))
        mapping = copy.deepcopy(fused.projector_dict)
        write(output / "runtime.json", {"torch": torch.__version__, "transformers": transformers.__version__,
              "device": torch.cuda.get_device_name(0), "strict_projectors_loaded": 28,
              "models": {r: m.config._commit_hash for r, m in models.items()}, "mapping": mapping})
        prepared = []
        for case in cases:
            choices = "".join(f"{chr(65+i)}. {text}\n" for i, text in enumerate(case["choices"]))
            prompt = build_prompt("mmlu-redux", "", case["question"], choices, False, True)
            chats = {r: t.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                      add_generation_prompt=True, enable_thinking=False) for r, t in tokenizers.items()}
            ids = {r: tokenizers[r](chats[r], return_tensors="pt").input_ids for r in chats}
            if chats["receiver"] != chats["sender"] or not torch.equal(ids["receiver"], ids["sender"]):
                raise ValueError("same-tokenizer inference path not applicable")
            if ids["receiver"].shape[1] > SPEC["max_input_tokens"]:
                raise ValueError("input budget exceeded")
            prepared.append((case, chats["receiver"], ids["receiver"]))
        count = 0
        with torch.inference_mode(), (output / "raw.jsonl").open("x", encoding="utf-8") as stream:
            for case, chat, ids_cpu in prepared:
                ids = ids_cpu.to("cuda:0")
                mask = torch.ones_like(ids)
                for arm in SPEC["arms"]:
                    kwargs = {"input_ids": ids, "attention_mask": mask, "do_sample": False,
                              "max_new_tokens": SPEC["max_new_tokens"], "repetition_penalty": 1.0}
                    if arm in ("c2c", "disabled_fuser"):
                        fused.projector_dict = mapping if arm == "c2c" else {}
                        kwargs["kv_cache_index"] = [torch.tensor([1, 0], device="cuda:0").repeat(ids.shape[1]-1, 1).unsqueeze(0),
                                                   torch.tensor([[-1, 0]], device="cuda:0").unsqueeze(0)]
                        kwargs["position_ids"] = mask.long().cumsum(-1)-1
                        model, tok = fused, tokenizers["receiver"]
                    else:
                        model, tok = models[arm], tokenizers[arm]
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    generated = model.generate(**kwargs)[0, ids.shape[1]:]
                    torch.cuda.synchronize()
                    row = {"case_id": case["case_id"], "dataset": case["dataset"], "arm": arm,
                           "chat": chat, "input_ids": ids_cpu[0].tolist(), "generated_ids": generated.tolist(),
                           "completion": tok.decode(generated, skip_special_tokens=True),
                           "hit_token_limit": len(generated) == SPEC["max_new_tokens"], "seconds": time.monotonic()-start}
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    count += 1
            fused.projector_dict = mapping
        write(output / "COMPLETE.json", {"calls": count, "updates": 0})
    except BaseException as exc:
        write(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write(output / "MANIFEST.json", {"files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("upstream", "assets", "cases", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.upstream, args.assets, args.cases, args.output)
