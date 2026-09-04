"""One prospective explicit-layout repair; strict v2 scoring remains unchanged."""
import argparse
import hashlib
import json
from pathlib import Path

from . import sdpo_format_control_data as CHECKER

SEED = "sdpo-format-control-20260904-v3-layout-examples"
score = CHECKER.score
reference = CHECKER.reference
RULES = CHECKER.RULES
STYLES = CHECKER.STYLES


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def format_guide(style):
    dummy = dict(preference=style, facts={"asset": "unit-example", "zone": "sector-example"})
    extra = {
        "json": "Use exactly two keys and substitute the record's values.",
        "bullets": "Use exactly two bullet items, asset then zone, with a colon after each key.",
        "table": "Use exactly four lines: header, ONE separator line, asset row, zone row. Do not repeat the separator line.",
        "plain": "Use exactly two lines, asset then zone, with a COLON after each key. Do not use equals signs, semicolons, bullets, or a table.",
    }
    return (CHECKER.RULES[style] + " " + extra[style] +
            "\nFormat example with unrelated dummy values: copy its layout, NOT its values. "
            "Replace both dummy values with the values from the supplied record.\n" + reference(dummy))


def build():
    result = {}
    for split, previous in CHECKER.build().items():
        result[split] = []
        for old in previous:
            uid = old["user_id"]  # same stationary users; fresh episode IDs and facts
            index = old["id"].rsplit("/", 1)[1]
            cid = f"{split}/{uid}/v3-{index}"
            nonce = sha(f"{SEED}/{cid}")
            facts = dict(asset="unit-" + nonce[:8], zone="sector-" + nonce[8:16])
            prompt = [dict(m) for m in old["prompt"]]
            for message in prompt:
                for a, b in [(old["user_id"], uid), *[(old["facts"][k], facts[k]) for k in facts]]:
                    message["content"] = message["content"].replace(a, b)
            explicit = [dict(m) for m in prompt]
            explicit[-1]["content"] += "\nMy response-format preference: " + format_guide(old["preference"])
            result[split].append(dict(id=cid, user_id=uid, prompt=prompt,
                preference=old["preference"], facts=facts, calibration_prompt=explicit))
    return result


def oracle_feedback(record):
    return "My preferred response format is as follows. " + format_guide(record["preference"]) + " Use the supplied record's asset and zone exactly."


def feedback(text, record):
    result = score(text, record)
    if result["success"]:
        return "Thanks, that matches my preferred format and the record."
    parts = []
    if not result["format_ok"]:
        parts.append("That is not my preferred response format. " + format_guide(record["preference"]))
    if not result["content_ok"] or (result["format_ok"] and not result["success"]):
        parts.append("Please correct the asset and zone to match the supplied record exactly.")
    return " ".join(parts)


def prepare(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for split, rows in build().items():
        path = root / f"{split}.json"
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        manifest[path.name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), count=len(rows))
    manifest["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["checker_sha256"] = hashlib.sha256(Path(CHECKER.__file__).read_bytes()).hexdigest()
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("root"); prepare(p.parse_args().root)
