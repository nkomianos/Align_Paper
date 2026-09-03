"""Read-only readiness/integrity check. Does not connect to or launch a GPU."""
import hashlib
import json
from pathlib import Path

from efference_pair.pilot import validate_seal
from latent_contract.channel import validate
from visual_hindsight_g0.corpus import validate_corpus


def main():
    repo = Path(__file__).resolve().parents[1]
    queue = json.loads((repo / "configs/research_queue_20260902.json").read_text())
    for item in [*queue["experiments"], queue["real_video_development"]]:
        root = repo / item["prepared"]
        if "manifest_sha256" in item:
            digest = hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest()
            if digest != item["manifest_sha256"]:
                raise ValueError("prepared manifest pin mismatch")
            if item.get("id") == "efference_pair_ep0":
                validate_seal(root)
            else:
                validate(root)
        else:
            report = validate_corpus(root)
            if report["ordered_frame_digest"] != item["ordered_frame_digest"]:
                raise ValueError("prepared video frame digest mismatch")
        print(item.get("id", "real_video_development") + ": prepared bytes verified")
    print("Offline readiness only. CUDA/model loading untested; no GPU experiment launched.")


if __name__ == "__main__":
    main()
