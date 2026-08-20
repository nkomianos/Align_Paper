"""Dependency-closure policy for the DID-v1 paid inference bootstrap.

The experiment dependencies and provider PyTorch have intentionally different
origin policies.  Locked experiment dependencies must resolve from the isolated
venv, with provider Torch as a traversal boundary.  Torch's own support closure
may resolve outside the venv only through the explicit allowlist below.  Both
closures are version-checked and fully attested.
"""

from __future__ import annotations

from collections import deque
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, Callable, Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from under_extinction.dev_diag_deployment import (
    DEPENDENCY_CLOSURE_POLICY,
    PROVIDER_TORCH_SUPPORT_ALLOWLIST,
)


DistributionGetter = Callable[[str], Any]


def _active_requirements(distribution_value: Any, extras: frozenset[str]) -> list[Requirement]:
    active: list[Requirement] = []
    for raw in distribution_value.requires or ():
        requirement = Requirement(raw)
        marker_active = requirement.marker is None or any(
            requirement.marker.evaluate(environment={"extra": extra})
            for extra in ({""} | set(extras))
        )
        if marker_active:
            active.append(requirement)
    return active


def _record(distribution_value: Any, venv_root: Path) -> dict[str, Any]:
    name = str(distribution_value.metadata["Name"])
    location = Path(distribution_value.locate_file("")).resolve()
    return {
        "name": name,
        "canonical_name": canonicalize_name(name),
        "version": str(distribution_value.version),
        "location": str(location),
        "inside_venv": location.is_relative_to(venv_root),
    }


def _walk_closure(
    roots: Sequence[Requirement],
    *,
    venv_root: Path,
    getter: DistributionGetter,
    kind: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    pending = deque((requirement, frozenset(requirement.extras)) for requirement in roots)
    seen: set[tuple[str, frozenset[str]]] = set()
    records: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    while pending:
        required, active_extras = pending.popleft()
        key = (canonicalize_name(required.name), active_extras)
        # Every incoming edge has its own version constraint.  Resolve and
        # validate it even when this distribution/extras node was already
        # traversed through another parent; only recursive expansion is
        # deduplicated.  Otherwise a later, incompatible transitive constraint
        # could be silently skipped.
        try:
            installed = getter(required.name)
        except PackageNotFoundError:
            failures.append(f"{kind}: missing distribution {required}")
            continue
        if required.specifier and installed.version not in required.specifier:
            failures.append(
                f"{kind}: requires {required}, found "
                f"{installed.metadata['Name']}=={installed.version}"
            )
        row = _record(installed, venv_root)
        canonical_name = str(row["canonical_name"])
        if kind == "experiment":
            if canonical_name == "torch":
                row["origin_policy"] = "provider_torch_boundary"
                records[canonical_name] = row
                # Torch support dependencies are audited in their own closure.
                continue
            if not row["inside_venv"]:
                failures.append(
                    f"experiment: dependency {row['name']} leaked from outside the venv: "
                    f"{row['location']}"
                )
                row["origin_policy"] = "forbidden_external_experiment"
            else:
                row["origin_policy"] = "diagnostic_venv"
        elif kind == "provider_torch":
            if row["inside_venv"]:
                row["origin_policy"] = "diagnostic_venv_satisfies_torch"
            elif canonical_name in PROVIDER_TORCH_SUPPORT_ALLOWLIST:
                row["origin_policy"] = "attested_provider_torch_support"
            else:
                failures.append(
                    "provider_torch: unallowlisted provider dependency "
                    f"{row['name']} outside the venv: {row['location']}"
                )
                row["origin_policy"] = "forbidden_provider_origin"
        else:
            raise ValueError(f"Unknown dependency closure kind: {kind}")
        records[canonical_name] = row
        if key in seen:
            continue
        seen.add(key)
        for dependency in _active_requirements(installed, active_extras):
            pending.append((dependency, frozenset(dependency.extras)))
    return records, failures


def parse_lock_requirements(lock_path: str | Path) -> list[Requirement]:
    """Parse exact top-level requirements from the transferred GPU lock."""

    path = Path(lock_path)
    requirements: list[Requirement] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith(("#", "--")):
            requirement = Requirement(line)
            specifiers = list(requirement.specifier)
            if (
                len(specifiers) != 1
                or specifiers[0].operator != "=="
                or "*" in specifiers[0].version
            ):
                raise ValueError(f"Unpinned diagnostic lock requirement: {requirement}")
            requirements.append(requirement)
    if not requirements:
        raise ValueError("Diagnostic GPU lock contains no requirements")
    names = [canonicalize_name(requirement.name) for requirement in requirements]
    if len(names) != len(set(names)):
        raise ValueError("Diagnostic GPU lock contains duplicate distributions")
    if "torch" in names:
        raise ValueError("Provider Torch must not appear in the diagnostic pip lock")
    return requirements


def plan_experiment_dependency_overlay(
    lock_path: str | Path,
    venv_root: str | Path,
    *,
    distribution_getter: DistributionGetter = distribution,
) -> list[str]:
    """Return exact non-Torch requirements that must be overlaid into the venv.

    A system-site-packages venv can cause pip to treat matching provider packages
    as already satisfied.  The bootstrap installs this returned list with
    ``--ignore-installed --no-deps`` and then runs the strict closure audit.
    """

    root = Path(venv_root).resolve()
    locked_roots = parse_lock_requirements(lock_path)
    experiment, failures = _walk_closure(
        locked_roots,
        venv_root=root,
        getter=distribution_getter,
        kind="experiment",
    )
    fatal = [failure for failure in failures if " leaked from outside the venv:" not in failure]
    if fatal:
        raise ValueError(
            "Cannot plan experiment dependency overlay:\n- "
            + "\n- ".join(sorted(set(fatal)))
        )
    return sorted(
        f"{row['name']}=={row['version']}"
        for name, row in experiment.items()
        if name != "torch" and not row["inside_venv"]
    )


def audit_dependency_closures(
    lock_path: str | Path,
    venv_root: str | Path,
    *,
    distribution_getter: DistributionGetter = distribution,
) -> dict[str, Any]:
    """Validate and attest isolated experiment and provider-Torch closures.

    ``distribution_getter`` is injectable so the exact origin policy can be
    tested without a GPU or provider image.
    """

    root = Path(venv_root).resolve()
    locked_roots = parse_lock_requirements(lock_path)
    experiment, experiment_failures = _walk_closure(
        locked_roots,
        venv_root=root,
        getter=distribution_getter,
        kind="experiment",
    )
    provider, provider_failures = _walk_closure(
        [Requirement("torch>=2.5,<3")],
        venv_root=root,
        getter=distribution_getter,
        kind="provider_torch",
    )
    failures = sorted(set(experiment_failures + provider_failures))
    if failures:
        raise ValueError(
            "Experiment/provider dependency closure is inconsistent:\n- "
            + "\n- ".join(failures)
        )

    provider_snapshot: dict[str, dict[str, Any]] = {}
    for canonical_name in sorted(PROVIDER_TORCH_SUPPORT_ALLOWLIST):
        try:
            installed = distribution_getter(canonical_name)
        except PackageNotFoundError:
            continue
        provider_snapshot[canonical_name] = _record(installed, root)

    return {
        "schema_version": "1.0",
        "policy": DEPENDENCY_CLOSURE_POLICY,
        "experiment_roots": [str(requirement) for requirement in locked_roots],
        "provider_torch_root": "torch>=2.5,<3",
        "provider_torch_support_allowlist": sorted(PROVIDER_TORCH_SUPPORT_ALLOWLIST),
        "experiment_closure": {
            name: experiment[name] for name in sorted(experiment)
        },
        "provider_torch_closure": {name: provider[name] for name in sorted(provider)},
        "installed_provider_allowlist_snapshot": provider_snapshot,
        "checks": {
            "all_locked_roots_version_matched": True,
            "all_non_torch_experiment_dependencies_inside_venv": True,
            "provider_torch_is_explicit_experiment_boundary": True,
            "provider_torch_closure_fully_traversed": True,
            "every_external_torch_support_distribution_allowlisted": True,
        },
    }


__all__ = [
    "PROVIDER_TORCH_SUPPORT_ALLOWLIST",
    "audit_dependency_closures",
    "parse_lock_requirements",
    "plan_experiment_dependency_overlay",
]
