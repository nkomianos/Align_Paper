from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
from packaging.utils import canonicalize_name

from under_extinction.dev_diag_bootstrap import (
    audit_dependency_closures,
    parse_lock_requirements,
    plan_experiment_dependency_overlay,
)


class _FakeDistribution:
    def __init__(
        self,
        name: str,
        version: str,
        location: Path,
        requires: tuple[str, ...] = (),
    ) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self._location = location
        self.requires = requires

    def locate_file(self, _value: str) -> Path:
        return self._location


def _getter(distributions: dict[str, _FakeDistribution]):
    normalized = {canonicalize_name(name): value for name, value in distributions.items()}

    def get(name: str) -> _FakeDistribution:
        try:
            return normalized[canonicalize_name(name)]
        except KeyError as exc:
            raise PackageNotFoundError(name) from exc

    return get


def _provider_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, _FakeDistribution]]:
    venv = tmp_path / "runtime" / ".venv"
    venv_site = venv / "lib/python3.12/site-packages"
    provider_site = tmp_path / "provider/python"
    lock = tmp_path / "gpu.lock"
    lock.write_text("experiment-lib==1.0\n", encoding="utf-8")
    distributions = {
        "experiment-lib": _FakeDistribution(
            "experiment-lib",
            "1.0",
            venv_site,
            ("helper-lib>=2", "torch>=2.5"),
        ),
        "helper-lib": _FakeDistribution("helper-lib", "2.1", venv_site),
        "torch": _FakeDistribution(
            "torch",
            "2.7.1",
            provider_site,
            ("sympy>=1.13", "nvidia-nccl-cu12==2.26.2"),
        ),
        "sympy": _FakeDistribution("sympy", "1.13.3", venv_site),
        "nvidia-nccl-cu12": _FakeDistribution(
            "nvidia-nccl-cu12", "2.26.2", provider_site
        ),
    }
    return lock, venv, distributions


def test_dependency_audit_separates_venv_and_provider_torch_closures(
    tmp_path: Path,
) -> None:
    lock, venv, distributions = _provider_fixture(tmp_path)
    report = audit_dependency_closures(
        lock,
        venv,
        distribution_getter=_getter(distributions),
    )
    assert report["checks"] == {
        "all_locked_roots_version_matched": True,
        "all_non_torch_experiment_dependencies_inside_venv": True,
        "provider_torch_is_explicit_experiment_boundary": True,
        "provider_torch_closure_fully_traversed": True,
        "every_external_torch_support_distribution_allowlisted": True,
    }
    assert report["experiment_closure"]["helper-lib"]["origin_policy"] == "diagnostic_venv"
    assert (
        report["experiment_closure"]["torch"]["origin_policy"]
        == "provider_torch_boundary"
    )
    assert (
        report["provider_torch_closure"]["nvidia-nccl-cu12"]["origin_policy"]
        == "attested_provider_torch_support"
    )
    assert (
        report["provider_torch_closure"]["sympy"]["origin_policy"]
        == "diagnostic_venv_satisfies_torch"
    )


def test_dependency_audit_rejects_external_experiment_dependency(tmp_path: Path) -> None:
    lock, venv, distributions = _provider_fixture(tmp_path)
    distributions["helper-lib"]._location = tmp_path / "provider/python"
    with pytest.raises(ValueError, match="helper-lib leaked from outside the venv"):
        audit_dependency_closures(
            lock,
            venv,
            distribution_getter=_getter(distributions),
        )


def test_overlay_plan_forces_matching_provider_roots_but_never_torch(tmp_path: Path) -> None:
    lock, venv, distributions = _provider_fixture(tmp_path)
    provider = tmp_path / "provider/python"
    distributions["experiment-lib"]._location = provider
    distributions["helper-lib"]._location = provider
    overlay = plan_experiment_dependency_overlay(
        lock,
        venv,
        distribution_getter=_getter(distributions),
    )
    assert overlay == ["experiment-lib==1.0", "helper-lib==2.1"]
    assert all(not requirement.lower().startswith("torch==") for requirement in overlay)


def test_dependency_audit_rejects_unallowlisted_provider_support(tmp_path: Path) -> None:
    lock, venv, distributions = _provider_fixture(tmp_path)
    distributions["torch"].requires += ("mystery-kernel==1.0",)
    distributions["mystery-kernel"] = _FakeDistribution(
        "mystery-kernel", "1.0", tmp_path / "provider/python"
    )
    with pytest.raises(ValueError, match="unallowlisted provider dependency mystery-kernel"):
        audit_dependency_closures(
            lock,
            venv,
            distribution_getter=_getter(distributions),
        )


def test_dependency_audit_checks_every_incoming_constraint(tmp_path: Path) -> None:
    lock, venv, distributions = _provider_fixture(tmp_path)
    venv_site = venv / "lib/python3.12/site-packages"
    lock.write_text("parent-a==1.0\nparent-b==1.0\n", encoding="utf-8")
    distributions.update(
        {
            "parent-a": _FakeDistribution(
                "parent-a", "1.0", venv_site, ("shared-lib<2",)
            ),
            "parent-b": _FakeDistribution(
                "parent-b", "1.0", venv_site, ("shared-lib>=3",)
            ),
            "shared-lib": _FakeDistribution("shared-lib", "1.5", venv_site),
        }
    )
    with pytest.raises(ValueError, match=r"requires shared-lib>=3, found shared-lib==1\.5"):
        audit_dependency_closures(
            lock,
            venv,
            distribution_getter=_getter(distributions),
        )


@pytest.mark.parametrize("requirement", ["example>=1.0", "example==1.*", "example<2,>=1"])
def test_lock_parser_requires_one_exact_non_wildcard_pin(
    requirement: str, tmp_path: Path
) -> None:
    lock = tmp_path / "not-exact.lock"
    lock.write_text(requirement + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unpinned diagnostic lock requirement"):
        parse_lock_requirements(lock)
