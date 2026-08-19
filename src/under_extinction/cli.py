"""Command-line entry point. Heavy ML dependencies are imported only by GPU commands."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

from .analysis import write_analysis
from .artifacts import collect_results
from .config import load_config, output_root
from .deployment import create_bundle, install_frozen_data
from .evaluation import evaluate_adapter, merge_predictions
from .generator import generate_datasets
from .io import sha256_file, write_json
from .oracle import run_oracles
from .preflight import full_gpu_preflight, metadata_preflight
from .readiness import dry_run_summary
from .training import train_controller


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="under-extinction", description="Causal controller audit pilot")
    parser.add_argument("--config", required=True, help="Frozen YAML experiment configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Generate deterministic datasets and hashes")
    build.add_argument("--destination")

    oracle = subparsers.add_parser("oracle", help="Run CPU ground-truth organisms (not empirical evidence)")
    oracle.add_argument("--data-dir")
    oracle.add_argument("--destination")

    analyze = subparsers.add_parser("analyze", help="Compute fingerprints, uncertainty, and gates")
    analyze.add_argument("--predictions", required=True)
    analyze.add_argument("--destination")
    analyze.add_argument("--data-dir")

    subparsers.add_parser("dry-run", help="Print workload, hashes, revision, and budget without loading a model")
    subparsers.add_parser("smoke", help="Build, run oracle validation, and analyze entirely on CPU")

    train = subparsers.add_parser("train", help="Train one preregistered LoRA organism")
    train.add_argument("--controller", required=True, choices=("intended", "proxy", "cached"))
    train.add_argument("--seed", required=True, type=int)
    train.add_argument("--data-dir")
    train.add_argument("--run-dir")
    train.add_argument("--resume", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Score one adapter on legal A/B sequences")
    evaluate.add_argument("--adapter", required=True)
    evaluate.add_argument("--controller", required=True, choices=("intended", "proxy", "cached"))
    evaluate.add_argument("--seed", required=True, type=int)
    evaluate.add_argument("--data-dir")
    evaluate.add_argument("--destination")
    evaluate.add_argument("--dev-only", action="store_true")

    merge = subparsers.add_parser("merge", help="Merge per-adapter predictions with duplicate checks")
    merge.add_argument("--inputs", nargs="+", required=True, help="Paths or glob patterns")
    merge.add_argument("--destination", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate paid-instance environment")
    preflight.add_argument("--metadata-only", action="store_true")
    preflight.add_argument("--destination")

    bundle = subparsers.add_parser("bundle", help="Package only this project and frozen data")
    bundle.add_argument("--destination")

    install_data = subparsers.add_parser("install-data", help="Verify and install transferred frozen data")
    install_data.add_argument("--source", required=True)

    collect = subparsers.add_parser("collect", help="Archive results plus only the latest resumable checkpoint per run")
    collect.add_argument("--destination")

    gate = subparsers.add_parser("gate", help="Exit nonzero unless named completed gates pass")
    gate.add_argument("--report", required=True)
    gate.add_argument("--data-dir")
    gate.add_argument("--require", nargs="+", choices=("P", "A", "B", "C"), default=["A", "B"])

    bridge_build = subparsers.add_parser(
        "bridge-build", help="Build or verify the frozen same-environment bridge corpus"
    )
    bridge_build.add_argument("--destination")

    bridge_oracle = subparsers.add_parser(
        "bridge-oracle", help="Run CPU bridge positive controls (apparatus evidence only)"
    )
    bridge_oracle.add_argument("--split", required=True, choices=("dev", "test"))
    bridge_oracle.add_argument("--destination")
    bridge_oracle.add_argument("--data-dir")
    bridge_oracle.add_argument("--unlock-test", action="store_true")

    bridge_train = subparsers.add_parser(
        "bridge-train", help="Train one paired reward-acquisition arm"
    )
    bridge_train.add_argument("--objective", required=True, choices=("genuine", "proxy"))
    bridge_train.add_argument("--seed", required=True, type=int)
    bridge_train.add_argument("--data-dir")
    bridge_train.add_argument("--run-dir", required=True)
    bridge_train.add_argument("--resume", action="store_true")

    bridge_evaluate = subparsers.add_parser(
        "bridge-evaluate", help="Evaluate every frozen checkpoint under extinction"
    )
    bridge_evaluate.add_argument("--run-dir", required=True)
    bridge_evaluate.add_argument("--split", required=True, choices=("dev", "test"))
    bridge_evaluate.add_argument("--destination", required=True)
    bridge_evaluate.add_argument("--data-dir")
    bridge_evaluate.add_argument("--unlock-test", action="store_true")
    bridge_evaluate.add_argument(
        "--unchanged-base", action="store_true",
        help="Score the unchanged base on checkpoint-zero DEV as a separate negative control",
    )

    bridge_analyze = subparsers.add_parser(
        "bridge-analyze", help="Analyze a complete paired bridge checkpoint series"
    )
    bridge_analyze.add_argument("--predictions", nargs="+", required=True)
    bridge_analyze.add_argument("--split", required=True, choices=("dev", "test"))
    bridge_analyze.add_argument("--destination", required=True)
    bridge_analyze.add_argument(
        "--base-control", help="Separate unchanged-base DEV prediction file"
    )

    bridge_gate = subparsers.add_parser(
        "bridge-gate", help="Verify a hash-bound bridge continuation gate"
    )
    bridge_gate.add_argument("--report", required=True)
    bridge_gate.add_argument("--require", required=True, choices=("smoke", "stage1", "replication"))

    bridge_preflight_attest = subparsers.add_parser(
        "bridge-preflight-attest",
        help="Create the hash-bound PASS handoff after a successful paid smoke test",
    )
    bridge_preflight_attest.add_argument("--stage1-config", required=True)
    bridge_preflight_attest.add_argument("--report", required=True)
    bridge_preflight_attest.add_argument("--destination")

    bridge_preflight_verify = subparsers.add_parser(
        "bridge-preflight-verify",
        help="Require an exact paid-preflight PASS before formal Stage 1",
    )
    bridge_preflight_verify.add_argument("--smoke-config", required=True)
    bridge_preflight_verify.add_argument("--attestation", required=True)
    return parser


def _expand_inputs(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches or [pattern])
    return sorted(set(paths))


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _bridge_run_identity(config: dict[str, Any], run_dir: str | Path) -> tuple[str, int]:
    manifest_path = Path(run_dir).resolve() / "bridge_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing bridge run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arm = str(manifest.get("arm", ""))
    seed = int(manifest.get("pair_seed", -1))
    if arm not in config["bridge"]["objectives"] or seed not in config["bridge"]["seeds"]:
        raise ValueError("Bridge run identity is not preregistered by this configuration")
    if manifest.get("config_sha256") != config["_config_sha256"]:
        raise ValueError("Bridge run manifest/config hash mismatch")
    return arm, seed


def _bridge_dry_run_summary(config: dict[str, Any]) -> dict[str, Any]:
    from .bridge_training import BridgeTrainingSpec

    root = output_root(config)
    manifest_path = root / "data" / "MANIFEST.json"
    checks: dict[str, bool] = {}
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in (manifest.get("files") or {}).values():
            path = root / "data" / item["path"]
            checks[str(item["path"])] = path.is_file() and sha256_file(path) == item["sha256"]
    data_ready = bool(manifest) and manifest.get("config_sha256") == config["_config_sha256"] and all(checks.values())
    spec = BridgeTrainingSpec.from_config(config)
    workload_profile: dict[str, Any] | None = None
    workload_profile_error: str | None = None
    if spec.updates > 1:
        try:
            from .bridge_budget import _load_workload_profile

            smoke_config = load_config(
                Path(config["_config_path"]).resolve().with_name("bridge_smoke.yaml")
            )
            workload_profile, profile_path, _, _ = _load_workload_profile(
                smoke_config, config
            )
            workload_profile = {
                "path": str(profile_path),
                "profile_sha256": workload_profile["profile_sha256"],
                "training_update_scale": workload_profile["training_update_scale"],
                "evaluation_per_record_scale": workload_profile[
                    "evaluation_per_record_scale"
                ],
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            workload_profile_error = str(exc)
    nominal = float(config["budget"]["nominal_usd"])
    reserve = float(config["budget"]["reserve_fraction"])
    hourly = float(config["budget"]["hourly_usd"])
    return {
        "experiment_name": config["experiment_name"],
        "experiment_family": config["experiment_family"],
        "config_sha256": config["_config_sha256"],
        "model": json.loads(json.dumps(config["model"], sort_keys=True)),
        "data_ready_and_hash_matched": data_ready,
        "data_hash_checks": checks,
        "data_counts": manifest.get("counts") if manifest else None,
        "paired_seed_count": len(config["bridge"]["seeds"]),
        "reward_arm_count": len(config["bridge"]["objectives"]),
        "optimizer_updates_per_arm": spec.updates,
        "checkpoint_updates": list(spec.checkpoint_updates),
        "checkpoint_count_per_arm": len(spec.checkpoint_updates),
        "rollout_batch_size": spec.batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "microbatch_size": spec.microbatch_size,
        "expected_lora_module_count": int(
            config["training"]["expected_lora_module_count"]
        ),
        "expected_lora_trainable_parameter_count": int(
            config["training"]["expected_lora_trainable_parameter_count"]
        ),
        "workload_profile": workload_profile,
        "workload_profile_error": workload_profile_error,
        "budget": {
            "nominal_usd": nominal,
            "reserve_usd": nominal * reserve,
            "usable_hours_at_recorded_rate": nominal * (1.0 - reserve) / hourly,
        },
        "ready_for_paid_preflight": data_ready and (
            spec.updates == 1 or workload_profile is not None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "bridge-build":
        from .bridge_env import build_bridge_data

        _print(build_bridge_data(config, destination=args.destination))
    elif args.command == "bridge-oracle":
        if args.split == "test" and not args.unlock_test:
            raise PermissionError("Locked bridge oracle data requires --unlock-test")
        from .bridge_oracle import run_bridge_oracles

        print(run_bridge_oracles(
            config, split=args.split, destination=args.destination, data_dir=args.data_dir
        ))
    elif args.command == "bridge-train":
        from .bridge_env import load_bridge_environment
        from .bridge_training import train_bridge_arm

        if args.seed not in config["bridge"]["seeds"]:
            raise ValueError(f"Seed {args.seed} is not preregistered")
        environment = load_bridge_environment(
            config, data_dir=args.data_dir, allowed_splits=("train",)
        )
        print(train_bridge_arm(
            config, environment, arm=args.objective, pair_seed=args.seed,
            run_dir=args.run_dir, resume=args.resume,
        ))
    elif args.command == "bridge-evaluate":
        from .bridge_env import load_bridge_environment
        from .bridge_evaluation import evaluate_bridge_run, evaluate_unchanged_base_control

        arm, seed = _bridge_run_identity(config, args.run_dir)
        environment = load_bridge_environment(
            config, data_dir=args.data_dir, allowed_splits=(args.split,)
        )
        if args.unchanged_base:
            if args.split != "dev" or args.unlock_test:
                raise PermissionError("The unchanged-base negative control is DEV-only")
            result = evaluate_unchanged_base_control(
                config, environment, anchor_run_dir=args.run_dir, anchor_arm=arm,
                pair_seed=seed, split=args.split, destination=args.destination,
            )
        else:
            result = evaluate_bridge_run(
                config, environment, run_dir=args.run_dir, arm=arm, pair_seed=seed,
                split=args.split, dev_only=args.split == "dev",
                unlock_test=args.unlock_test, destination=args.destination,
            )
        print(result)
    elif args.command == "bridge-analyze":
        from .bridge_analysis import write_bridge_analysis

        paths = _expand_inputs(args.predictions)
        print(write_bridge_analysis(
            config, paths, split=args.split, destination=args.destination,
            base_control_path=args.base_control,
        ))
    elif args.command == "bridge-gate":
        from .bridge_analysis import verify_bridge_gate_report

        result = verify_bridge_gate_report(config, args.report, required=args.require)
        _print(result)
        return 0 if result["pass"] else 3
    elif args.command == "bridge-preflight-attest":
        from .bridge_preflight import write_bridge_preflight_attestation

        stage1_config = load_config(args.stage1_config)
        print(write_bridge_preflight_attestation(
            config,
            stage1_config,
            smoke_report_path=args.report,
            destination=args.destination,
        ))
    elif args.command == "bridge-preflight-verify":
        from .bridge_preflight import verify_bridge_preflight_attestation

        smoke_config = load_config(args.smoke_config)
        _print(verify_bridge_preflight_attestation(
            smoke_config,
            config,
            attestation_path=args.attestation,
        ))
    elif args.command == "build":
        _print(generate_datasets(config, args.destination))
    elif args.command == "oracle":
        print(run_oracles(config, data_dir=args.data_dir, destination=args.destination))
    elif args.command == "analyze":
        print(write_analysis(config, args.predictions, args.destination, data_dir=args.data_dir))
    elif args.command == "dry-run":
        summary = (
            _bridge_dry_run_summary(config)
            if config.get("experiment_family") == "same_environment_rl_bridge"
            else dry_run_summary(config)
        )
        _print(summary)
        return 0 if summary["ready_for_paid_preflight"] else 2
    elif args.command == "smoke":
        manifest = generate_datasets(config)
        predictions = run_oracles(config)
        report = write_analysis(config, predictions)
        _print({"data": manifest, "predictions": str(predictions), "report": str(report)})
    elif args.command == "train":
        print(train_controller(
            config, controller=args.controller, seed=args.seed, data_dir=args.data_dir,
            run_dir=args.run_dir, resume=args.resume,
        ))
    elif args.command == "evaluate":
        print(evaluate_adapter(
            config, adapter_path=args.adapter, controller=args.controller, seed=args.seed,
            data_dir=args.data_dir, destination=args.destination, dev_only=args.dev_only,
        ))
    elif args.command == "merge":
        print(merge_predictions(_expand_inputs(args.inputs), args.destination))
    elif args.command == "preflight":
        if args.metadata_only:
            result = metadata_preflight(config, require_gpu=False)
            destination = Path(args.destination).resolve() if args.destination else output_root(config) / "preflight" / "metadata.json"
            write_json(destination, result)
            print(destination)
        else:
            print(full_gpu_preflight(config, args.destination))
    elif args.command == "bundle":
        print(create_bundle(config, args.destination))
    elif args.command == "install-data":
        print(install_frozen_data(config, args.source))
    elif args.command == "collect":
        project_root = Path(config["_config_path"]).parent.parent
        print(collect_results(project_root, args.destination))
    elif args.command == "gate":
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        source = Path(args.data_dir).resolve() if args.data_dir else output_root(config) / "data"
        manifest_path = source / "MANIFEST.json"
        binding_failures: list[str] = []
        if report.get("config_sha256") != config["_config_sha256"]:
            binding_failures.append("report/config hash mismatch")
        if not manifest_path.exists() or report.get("data_manifest_sha256") != sha256_file(manifest_path):
            binding_failures.append("report/data-manifest hash mismatch")
        prediction_path = Path(report.get("prediction_path", ""))
        if not prediction_path.exists() or report.get("prediction_sha256") != sha256_file(prediction_path):
            binding_failures.append("report/prediction hash mismatch")
        if any(name in {"P", "A", "B"} for name in args.require) and report.get("evidence_kinds") != ["lora_model_organism"]:
            binding_failures.append("paid-stage gate requires homogeneous LoRA-model evidence")
        mapping = {
            "P": "stage1_diagnostic",
            "A": "gate_a_organism_validity",
            "B": "gate_b_construct_validity",
            "C": "gate_c_prospective_value",
        }
        failed = [name for name in args.require if not report["gates"][mapping[name]]["pass"]]
        passed = not failed and not binding_failures
        _print({"required": args.require, "failed": failed, "binding_failures": binding_failures, "pass": passed})
        return 0 if passed else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
