"""Version-pinned v3 verifier; delegates mechanics but preserves v2 checker."""
import argparse
import json

from scripts.sdpo_format_verifier_common_v3 import APPARATUS_SHA as CHECKER_SHA, verify

APPARATUS_SHA = "097c3b4ab4edc0c99b696cf91de717bfe9f811126b564dd7a5a9bc7791b54604"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root"); parser.add_argument("report")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-path")
    args = parser.parse_args()
    result = verify(args.root, args.tokenizer_path, args.model_path,
                    apparatus_sha=APPARATUS_SHA, checker_sha=CHECKER_SHA)
    result["apparatus_version"] = "v3-privileged-layout-example-repair"
    result["strict_checker_version"] = "v2-unchanged"
    result["development_caveat"] = "Prospective repair designed after v2 calibration failures; not independent confirmatory evidence or a paper gate."
    with open(args.report, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
