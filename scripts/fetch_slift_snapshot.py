"""Fetch the official anonymous SLIFT snapshot with a frozen checksum."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

from interaction_sprint.hindsight_slift_baseline import (
    SLIFT_RELEASE_LAST_UPDATE_UTC,
    SLIFT_RELEASE_PAGE,
    SLIFT_RELEASE_URL,
    SLIFT_RELEASE_TREE_SHA256,
)


def canonical_tree(payload: bytes) -> tuple[str, list[dict[str, object]]]:
    """Hash normalized member paths and contents, ignoring ZIP metadata."""
    inventory: list[dict[str, object]] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/"):
                raise ValueError(f"unsafe ZIP member: {info.filename}")
            content = archive.read(info)
            inventory.append({
                "path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            })
    inventory.sort(key=lambda item: str(item["path"]))
    if not inventory or len({item["path"] for item in inventory}) != len(inventory):
        raise ValueError("empty or duplicate-path SLIFT archive")
    serialized = json.dumps(
        inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest(), inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise SystemExit(f"refusing to overwrite: {args.destination}")
    request = urllib.request.Request(
        SLIFT_RELEASE_URL,
        headers={"User-Agent": "Align-Paper-SLIFT-baseline-audit/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    archive_sha256 = hashlib.sha256(payload).hexdigest()
    tree_sha256, inventory = canonical_tree(payload)
    if tree_sha256 != SLIFT_RELEASE_TREE_SHA256:
        raise SystemExit(
            "SLIFT release content-tree mismatch: expected "
            f"{SLIFT_RELEASE_TREE_SHA256}, got {tree_sha256}"
        )
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_bytes(payload)
    print(json.dumps({
        "verified": True,
        "release_page": SLIFT_RELEASE_PAGE,
        "archive_url": SLIFT_RELEASE_URL,
        "observed_archive_sha256": archive_sha256,
        "content_tree_sha256": tree_sha256,
        "file_count": len(inventory),
        "reported_last_update_utc": SLIFT_RELEASE_LAST_UPDATE_UTC,
        "destination": str(args.destination.resolve()),
        "license_file_present_in_snapshot": False,
        "redistributed_by_this_repository": False,
    }, indent=2))


if __name__ == "__main__":
    main()
