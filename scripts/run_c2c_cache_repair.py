"""Conditional Stage C: fit existing cache repairs and measure held-out answers."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import time

from scripts.run_c2c_paired_update import (SPEC as PAIRED_SPEC, SOURCE_FILES as PAIRED_SOURCES,
                                          validate_ticket, generate_record)
from scripts.run_c2c_sender_update import SPEC as UPDATE_SPEC, sha256, write, pinned_prompt
from scripts.run_c2c_baseline_dev import verified_assets
from latent_contract.repair_analysis import REPAIR_ARMS

SPEC = {**PAIRED_SPEC, "scope": "EXISTING_CACHE_REPAIR_BASELINES_DEV", "arms": REPAIR_ARMS,
        "calls": 1280, "calibration_cases": 128, "fit_cases_per_dataset": 48,
        "calibration_sha256": "9f2fa5d28ea80144de5e9260a5bfc446ea5a87cd7a90177c770d5f345d81f559",
        "maximum_positions_per_case": 32, "ridge": .001,
        "retune_steps_per_layer": 100, "retune_lr": .0001, "retune_batch_tokens": 64,
        "primary_repair": "ridge", "min_recovery_fraction": .75}
SOURCE_FILES = list(dict.fromkeys(["scripts/run_c2c_cache_repair.py", "src/latent_contract/cache_repair.py",
               "src/latent_contract/cache_calibration.py", "src/latent_contract/repair_analysis.py", *PAIRED_SOURCES]))


def validate_parent(report_path, digest, ticket_sha):
    if sha256(report_path) != digest:
        raise ValueError("verified paired-report digest differs")
    report = json.loads(report_path.read_text())
    if report["decision"] != "REPEATED_EXTRA_LATENT_LOSS_REPAIR_STUDY_NEEDED":
        raise ValueError("no repeated paired signal; repair expansion not released")
    if {r["seed"] for r in report["per_seed"]} != set(UPDATE_SPEC["seeds"]) or len(report["per_seed"]) != 2:
        raise ValueError("both fixed seed reports required")
    if any(r["ticket_sha256"] != ticket_sha or r["decision"] != "EXTRA_LATENT_LOSS_SIGNAL_REPAIR_STUDY_NEEDED" for r in report["per_seed"]):
        raise ValueError("paired evidence not tied to this qualification ticket")
    return report


def run(args):
    ticket = validate_ticket(args.ticket, args.ticket_sha256, args.update_root, args.seed)
    validate_parent(args.paired_report, args.paired_report_sha256, args.ticket_sha256)
    for path, digest in ((args.cases, SPEC["cases_sha256"]), (args.calibration, SPEC["calibration_sha256"])):
        if sha256(path) != digest:
            raise ValueError("frozen data digest differs")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    try:
        write(output / "spec.json", {**SPEC, "seed": args.seed})
        for source, name in ((args.ticket, "release_ticket.json"), (args.paired_report, "paired_report.json"),
                             (args.cases, "cases.json"), (args.calibration, "calibration.json")):
            (output / name).write_bytes(source.read_bytes())
        source_root = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        hashes = {}
        for relative in SOURCE_FILES:
            frozen = subprocess.check_output(["git", "-C", str(source_root), "show", commit+":"+relative])
            if (source_root / relative).read_bytes() != frozen:
                raise ValueError("repair source not frozen")
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(frozen)
            hashes[relative] = hashlib.sha256(frozen).hexdigest()
        write(output / "source.json", {"commit": commit, "files": hashes})
        resolved = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
        if resolved != SPEC["upstream_commit"] or dirty:
            raise ValueError("wrong/modified C2C source")
        build_prompt = pinned_prompt(args.upstream)
        assets = verified_assets(args.assets)
        write(output / "assets.json", assets)
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from latent_contract.cache_calibration import collect_bank, fit_layer
        from latent_contract.cache_repair import CacheRepairProjector
        if torch.__version__ != SPEC["torch"] or transformers.__version__ != SPEC["transformers"]:
            raise ValueError("wrong frozen environment")
        sys.path.insert(0, str(args.upstream.resolve()))
        from rosetta.model.wrapper import RosettaModel
        from rosetta.model.projector import load_projector
        models, tokenizers = {}, {}
        for role in ("receiver", "old", "new"):
            family = "receiver" if role == "receiver" else "sender"
            name = str(args.update_root / "merged_sender") if role == "new" else SPEC[family]
            pin = {} if role == "new" else {"revision": SPEC[family+"_revision"]}
            tokenizers[role] = AutoTokenizer.from_pretrained(name, **pin, local_files_only=True, trust_remote_code=False)
            models[role] = AutoModelForCausalLM.from_pretrained(name, **pin, local_files_only=True, trust_remote_code=False,
                                  torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
            if role != "new" and models[role].config._commit_hash != SPEC[family+"_revision"]:
                raise ValueError("model revision mismatch")
        directory = Path(assets["fuser"]["snapshot"]) / "qwen3_0.6b+qwen3_4b_Fuser/final"
        projectors = []
        for index in range(28):
            projector = load_projector(str(directory / f"projector_{index}.json"))
            projector.load_state_dict(torch.load(directory / f"projector_{index}.pt", weights_only=True, map_location="cpu"), strict=True)
            projectors.append(projector)
        fused = RosettaModel([models["receiver"], models["new"]], projector_list=projectors).to("cuda:0").eval()
        fused.load_projector_config(str(directory / "projector_config.json"))
        write(output / "runtime.json", {"torch": torch.__version__, "transformers": transformers.__version__,
              "models": {r: m.config._commit_hash for r, m in models.items()}, "device": torch.cuda.get_device_name(0),
              "mapping": fused.projector_dict, "strict_projectors_loaded": 28,
              "updated_model_files": ticket["updates"][str(args.seed)]["merged_files"],
              "ticket_sha256": args.ticket_sha256, "paired_report_sha256": args.paired_report_sha256})
        calibration = json.loads(args.calibration.read_text())
        torch.cuda.synchronize()
        start = time.monotonic()
        bank = collect_bank(models, tokenizers, calibration, build_prompt, [(i, i+8) for i in range(28)], output / "cache_bank",
                            fit_per_dataset=SPEC["fit_cases_per_dataset"], maximum_tokens=SPEC["maximum_positions_per_case"])
        torch.cuda.synchronize()
        collection_seconds = time.monotonic()-start
        start = time.monotonic()
        repair_lists = {arm: [] for arm in REPAIR_ARMS}
        for index, projector in enumerate(projectors):
            cache = torch.load(output / "cache_bank" / f"layer_{index:02d}.pt", weights_only=True, map_location="cpu")
            layer_root = output / "repairs" / f"layer_{index:02d}"
            fit_layer(projector, cache, layer_root, seed=args.seed+index, device="cuda:0", steps=SPEC["retune_steps_per_layer"],
                      ridge=SPEC["ridge"], retune_lr=SPEC["retune_lr"], batch_tokens=SPEC["retune_batch_tokens"])
            for arm in REPAIR_ARMS:
                if arm == "retuned":
                    module = copy.deepcopy(projector)
                    module.load_state_dict(torch.load(layer_root / "retuned_bfloat16.pt", weights_only=True, map_location="cuda:0"), strict=True)
                else:
                    maps = torch.load(layer_root / (arm+".pt"), weights_only=True, map_location="cuda:0")
                    module = CacheRepairProjector(projector, maps["key"], maps["value"])
                repair_lists[arm].append(module.eval())
            del cache
        torch.cuda.synchronize()
        write(output / "repair_cost.json", {"cache_collection_seconds": collection_seconds, "fitting_and_loading_seconds": time.monotonic()-start,
              "backbone_forwards": bank["backbone_forwards"], "requires_original_sender_for_calibration": True,
              "interpretation": "Shared calibration plus all fits; generation timings below do not amortize these costs away"})
        count = 0
        with torch.inference_mode(), (output / "raw.jsonl").open("x", encoding="utf-8") as stream:
            for case_index, case in enumerate(json.loads(args.cases.read_text())):
                choices = "".join(f"{chr(65+i)}. {t}\n" for i, t in enumerate(case["choices"]))
                prompt = build_prompt("mmlu-redux", "", case["question"], choices, False, True)
                # Rotate method order independently of answers.
                offset = case_index % len(REPAIR_ARMS)
                for arm in REPAIR_ARMS[offset:] + REPAIR_ARMS[:offset]:
                    # Assign after wrapper construction: retain float32 repair maps.
                    fused.projector_list = torch.nn.ModuleList(repair_lists[arm])
                    row = generate_record(fused, tokenizers["receiver"], [{"role": "user", "content": prompt}], case, arm, "cuda:0", fused=True)
                    stream.write(json.dumps(row)+"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    count += 1
        write(output / "COMPLETE.json", {"calls": count, "seed": args.seed, "calibration_cases": 128, "retuned_layers": 28})
    except BaseException as exc:
        write(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write(output / "MANIFEST.json", {"files": {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob("*") if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "assets", "calibration", "cases", "ticket", "paired-report", "update-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("ticket-sha256", "paired-report-sha256"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--seed", type=int, choices=UPDATE_SPEC["seeds"], required=True)
    run(parser.parse_args())
