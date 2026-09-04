"""Paired Stage B: old/new sender, frozen C2C, text transfer and controls.

Requires the checksum-pinned local release ticket for both qualified updates.
No private keys, training, repair, or automatic expansion. Fresh output only.
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

from latent_contract.update_comparison import ARMS, DECISION_RULE
from scripts.run_c2c_baseline_dev import SPEC as BASE_SPEC, verified_assets
from scripts.run_c2c_sender_update import SPEC as UPDATE_SPEC, sha256, write, pinned_prompt
from scripts.run_c2c_text_baseline import messages

SPEC = {**BASE_SPEC, "scope": "NATURAL_UPDATE_PAIRED_DEV_NO_REPAIR", "cases": 256,
        "cases_sha256": "66fca7bc7e5725f380a74db8454a1bc49f2304d7b5c267cd3ebadb965cc2e83f",
        "arms": ARMS, "calls": 2816, "decision_rule": DECISION_RULE}
SOURCE_FILES = ["scripts/run_c2c_paired_update.py", "scripts/run_c2c_baseline_dev.py",
                "scripts/run_c2c_sender_update.py", "scripts/run_c2c_text_baseline.py",
                "src/latent_contract/update_comparison.py", "src/latent_contract/answer_scoring.py"]


def validate_ticket(path, expected_sha, update_root, seed):
    if sha256(path) != expected_sha:
        raise ValueError("release-ticket checksum mismatch")
    ticket = json.loads(path.read_text())
    if ticket["scope"] != "RELEASE_PAIRED_DEV_ONLY_NOT_PAPER_EXPANSION" or ticket["final_cases_sha256"] != SPEC["cases_sha256"]:
        raise ValueError("wrong release scope")
    if set(ticket["updates"]) != {str(s) for s in UPDATE_SPEC["seeds"]} or seed not in UPDATE_SPEC["seeds"]:
        raise ValueError("both fixed seeds required")
    for label, entry in ticket["updates"].items():
        if entry["seed"] != int(label) or entry["qualification"]["seed"] != int(label) or entry["qualification"]["decision"] != "QUALIFIED_FOR_PAIRED_INTERFACE_MEASUREMENT":
            raise ValueError("unqualified update in ticket")
    entry = ticket["updates"][str(seed)]
    if sha256(update_root / "MANIFEST.json") != entry["manifest_sha256"]:
        raise ValueError("updated checkpoint root differs from locally verified root")
    merged = update_root / "merged_sender"
    actual = {p.relative_to(merged).as_posix() for p in merged.rglob("*") if p.is_file()}
    if actual != set(entry["merged_files"]):
        raise ValueError("merged checkpoint file coverage mismatch")
    for name, digest in entry["merged_files"].items():
        target = (merged / name).resolve()
        if not target.is_relative_to(merged.resolve()) or sha256(target) != digest:
            raise ValueError("merged model checksum mismatch")
    return ticket


def fused_kwargs(ids, mask):
    import torch
    return {"kv_cache_index": [torch.tensor([1, 0], device=ids.device).repeat(ids.shape[1]-1, 1).unsqueeze(0),
                                torch.tensor([[-1, 0]], device=ids.device).unsqueeze(0)],
            "position_ids": mask.long().cumsum(-1)-1}


def generate_record(model, tokenizer, turns, case, arm, device, fused=False, max_new_tokens=64):
    import torch
    ids = tokenizer.apply_chat_template(turns, tokenize=True, add_generation_prompt=True,
                                       enable_thinking=False, return_tensors="pt").to(device)
    if not 1 < ids.shape[1] <= SPEC["max_input_tokens"]:
        raise ValueError("input budget exceeded")
    mask = torch.ones_like(ids)
    kwargs = {"input_ids": ids, "attention_mask": mask, "do_sample": False,
              "max_new_tokens": max_new_tokens, "repetition_penalty": 1.0}
    if fused:
        kwargs.update(fused_kwargs(ids, mask))
    if ids.is_cuda:
        torch.cuda.synchronize()
    start = time.monotonic()
    generated = model.generate(**kwargs)[0, ids.shape[1]:]
    if ids.is_cuda:
        torch.cuda.synchronize()
    return {"case_id": case["case_id"], "dataset": case["dataset"], "arm": arm, "messages": turns,
            "input_ids": ids[0].tolist(), "generated_ids": generated.tolist(),
            "completion": tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "hit_token_limit": len(generated) == max_new_tokens, "seconds": time.monotonic()-start}


def run(upstream, assets_root, cases_path, ticket_path, ticket_sha, update_root, seed, output):
    ticket = validate_ticket(ticket_path, ticket_sha, update_root, seed)
    if sha256(cases_path) != SPEC["cases_sha256"]:
        raise ValueError("final DEV digest differs")
    cases = json.loads(cases_path.read_text())
    if len(cases) != SPEC["cases"] or any("answer" in c for c in cases):
        raise ValueError("case count or label visibility differs")
    output.mkdir(parents=True, exist_ok=False)
    try:
        write(output / "spec.json", {**SPEC, "seed": seed})
        (output / "cases.json").write_bytes(cases_path.read_bytes())
        (output / "release_ticket.json").write_bytes(ticket_path.read_bytes())
        source_root = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        hashes = {}
        for relative in SOURCE_FILES:
            frozen = subprocess.check_output(["git", "-C", str(source_root), "show", commit+":"+relative])
            if (source_root / relative).read_bytes() != frozen:
                raise ValueError("runner source is not frozen")
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(frozen)
            hashes[relative] = hashlib.sha256(frozen).hexdigest()
        write(output / "source.json", {"commit": commit, "files": hashes})
        resolved = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
        if resolved != SPEC["upstream_commit"] or dirty:
            raise ValueError("upstream is not pinned/clean")
        build_prompt = pinned_prompt(upstream)
        assets = verified_assets(assets_root)
        write(output / "assets.json", assets)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if torch.__version__ != SPEC["torch"] or transformers.__version__ != SPEC["transformers"]:
            raise ValueError("dependency mismatch")
        torch.manual_seed(202609043)
        sys.path.insert(0, str(upstream.resolve()))
        from rosetta.model.projector import load_projector
        from rosetta.model.wrapper import RosettaModel
        models, tokenizers = {}, {}
        for role in ("receiver", "old", "new"):
            source_role = "receiver" if role == "receiver" else "sender"
            name = str(update_root / "merged_sender") if role == "new" else SPEC[source_role]
            pin = {} if role == "new" else {"revision": SPEC[source_role+"_revision"]}
            tokenizers[role] = AutoTokenizer.from_pretrained(name, **pin, local_files_only=True, trust_remote_code=False)
            models[role] = AutoModelForCausalLM.from_pretrained(name, **pin, local_files_only=True, trust_remote_code=False,
                                  torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
            if role != "new" and models[role].config._commit_hash != SPEC[source_role+"_revision"]:
                raise ValueError("model revision differs")
        folder = Path(assets["fuser"]["snapshot"]) / "qwen3_0.6b+qwen3_4b_Fuser/final"
        projectors = []
        for index in range(28):
            module = load_projector(str(folder / f"projector_{index}.json"))
            module.load_state_dict(torch.load(folder / f"projector_{index}.pt", map_location="cpu", weights_only=True), strict=True)
            projectors.append(module)
        fused = RosettaModel([models["receiver"], models["old"]], base_model_idx=0, projector_list=projectors).to("cuda:0").eval()
        fused.load_projector_config(str(folder / "projector_config.json"))
        mapping = copy.deepcopy(fused.projector_dict)
        write(output / "runtime.json", {"torch": torch.__version__, "transformers": transformers.__version__,
              "device": torch.cuda.get_device_name(0), "models": {r: m.config._commit_hash for r, m in models.items()},
              "mapping": mapping, "strict_projectors_loaded": 28, "ticket_sha256": ticket_sha,
              "updated_model_files": ticket["updates"][str(seed)]["merged_files"]})
        count = 0
        with torch.inference_mode(), (output / "raw.jsonl").open("x", encoding="utf-8") as stream:
            def record(*args, **kwargs):
                nonlocal count
                row = generate_record(*args, **kwargs)
                stream.write(json.dumps(row)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
                count += 1
                return row
            for index, case in enumerate(cases):
                choices = "".join(f"{chr(65+i)}. {t}\n" for i, t in enumerate(case["choices"]))
                prompt = build_prompt("mmlu-redux", "", case["question"], choices, False, True)
                turns = [{"role": "user", "content": prompt}]
                for candidate_turns in (turns, messages(case["question"], prompt)):
                    encoded = [tok.apply_chat_template(candidate_turns, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                               for tok in tokenizers.values()]
                    if any(x != encoded[0] for x in encoded) or len(encoded[0]) > SPEC["max_input_tokens"]:
                        raise ValueError("tokenizer/input contract differs")
                record(models["receiver"], tokenizers["receiver"], turns, case, "receiver", "cuda:0")
                # Counterbalance version order by predetermined row index; no outcome-based ordering.
                for version in (("old", "new") if index % 2 == 0 else ("new", "old")):
                    fused.model_list[1] = models[version]
                    record(models[version], tokenizers[version], turns, case, version+"_sender", "cuda:0")
                    for arm in ("c2c", "disabled"):
                        fused.projector_dict = mapping if arm == "c2c" else {}
                        record(fused, tokenizers["receiver"], turns, case, version+"_"+arm, "cuda:0", fused=True)
                    background = record(models[version], tokenizers[version], messages(case["question"], prompt),
                                        case, version+"_background", "cuda:0")
                    record(models["receiver"], tokenizers["receiver"], messages(case["question"], prompt, background["completion"]),
                           case, version+"_text", "cuda:0")
                fused.projector_dict = mapping
        write(output / "COMPLETE.json", {"calls": count, "seed": seed, "updates": 0, "repairs": 0})
    except BaseException as exc:
        write(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write(output / "MANIFEST.json", {"files": {p.relative_to(output).as_posix(): sha256(p)
              for p in output.rglob("*") if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "assets", "cases", "ticket", "update-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--ticket-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=UPDATE_SPEC["seeds"], required=True)
    args = parser.parse_args()
    run(args.upstream, args.assets, args.cases, args.ticket, args.ticket_sha256, args.update_root, args.seed, args.output)
