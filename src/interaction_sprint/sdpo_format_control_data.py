"""Grounded, stationary-preference SDPO apparatus control; no new paper claim.

Opaque user identities recur across train/eval. Case IDs and nonce fact content
are disjoint. The core prompt never includes the user's hidden style preference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

STYLES = ("json", "bullets", "table", "plain")
SEED = "sdpo-format-control-20260904-v1"
RULES = {
    "json": "Return only one JSON object with exactly the string keys asset and zone and their values. A JSON code fence is acceptable.",
    "bullets": "Return only a two-item bullet list, one asset: value item and one zone: value item. Use -, *, or a bullet character.",
    "table": "Return only a two-column Markdown table with header Field | Value, a separator row, and one row each for asset and zone.",
    "plain": "Return only two unbulleted key: value lines, one for asset and one for zone. Do not use JSON, a table, or a bullet list.",
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build():
    users = ["user_" + digest(f"{SEED}/identity/{i}")[:12] for i in range(8)]
    # Styles balanced but not revealed by identity order or printed in core prompt.
    assignment = ("table", "json", "plain", "bullets", "json", "table", "bullets", "plain")
    result = {}
    for split, count in (("train", 8), ("eval", 8), ("calibration", 4)):
        records = []
        for uid, style in zip(users, assignment):
            for j in range(count):
                cid = f"{split}/{uid}/{j}"
                nonce = digest(f"{SEED}/{cid}")
                facts = {"asset": "unit-" + nonce[:8], "zone": "sector-" + nonce[8:16]}
                prompt = [{"role": "system", "content": "Answer accurately using the supplied record."},
                    {"role": "user", "content": f"User identity: {uid}\nRecord: asset = {facts['asset']}; zone = {facts['zone']}.\nReport the asset and zone from this record."}]
                calibration_prompt = [dict(m) for m in prompt]
                calibration_prompt[-1]["content"] += "\nMy response-format preference: " + RULES[style]
                records.append({"id": cid, "user_id": uid, "prompt": prompt, "preference": style,
                                "facts": facts, "calibration_prompt": calibration_prompt})
        result[split] = records
    return result


def _pairs(lines, bullet=False):
    values = {}
    for line in lines:
        pattern = r"\s*[-*•]\s+([A-Za-z]+)\s*:\s*(\S+)\s*" if bullet else r"\s*([A-Za-z]+)\s*:\s*(\S+)\s*"
        match = re.fullmatch(pattern, line)
        if not match:
            return None
        key, value = match.groups()
        key = key.lower()
        if key in values:
            return None
        values[key] = value
    return values if set(values) == {"asset", "zone"} else None


def normalize_markdown_keys(text):
    """Ignore balanced emphasis on labels only, never alter nonce values."""
    return re.sub(r"\*\*(asset|zone|field|value)(\s*:?)\*\*", r"\1\2", text, flags=re.I)


def parse_style(text, style):
    text = text.strip()
    if style == "json":
        if text.startswith("```"):
            match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.S | re.I)
            if not match:
                return None
            text = match.group(1).strip()
        pairs = []
        try:
            obj = json.loads(text, object_pairs_hook=lambda x: pairs.extend(x) or dict(x))
        except (ValueError, TypeError):
            return None
        if not isinstance(obj, dict) or len(pairs) != 2 or set(obj) != {"asset", "zone"} or not all(isinstance(v, str) for v in obj.values()):
            return None
        return {k: v.strip() for k, v in obj.items()}
    lines = [line.strip() for line in normalize_markdown_keys(text).splitlines() if line.strip()]
    if style in ("plain", "bullets"):
        return _pairs(lines, bullet=style == "bullets") if len(lines) == 2 else None
    if style == "table":
        if len(lines) != 4:
            return None
        cells = [[c.strip() for c in line.strip("|").split("|")] for line in lines]
        if any(len(row) != 2 for row in cells) or [x.lower() for x in cells[0]] != ["field", "value"]:
            return None
        if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells[1]):
            return None
        return _pairs([f"{k}: {v}" for k, v in cells[2:]])
    raise ValueError(style)


def score(text, record):
    parsed = parse_style(text, record["preference"])
    # Separate grounding from preferred presentation: a correctly bound fact
    # can be present despite a leading explanation or a different output style.
    found = {key: set() for key in record["facts"]}
    for key in found:
        pattern = rf"\b{key}\b[\"']?\s*(?::|=|\bis\b|\|)\s*[\"']?([A-Za-z0-9_-]+)"
        found[key] = set(re.findall(pattern, normalize_markdown_keys(text), flags=re.I))
    grounded = all(found[k] == {v} for k, v in record["facts"].items())
    correct = parsed == record["facts"]
    return {"format_ok": parsed is not None, "content_ok": grounded,
            "success": bool(correct and grounded), "format_valid": parsed is not None,
            "content_valid": grounded, "joint": bool(correct and grounded),
            "extracted_bindings": {k: sorted(v) for k, v in found.items()}}


def oracle_feedback(record):
    """Explicit preference calibration; never place in a core policy prompt."""
    return "My preferred response format is as follows. " + RULES[record["preference"]] + " Use the asset and zone in the supplied record exactly."


def feedback(text, record):
    result = score(text, record)
    if result["success"]:
        return "Thanks, that matches my preferred format and the record."
    parts = []
    if not result["format_ok"]:
        parts.append("That is not my preferred response format. " + RULES[record["preference"]])
    if not result["content_ok"] or (result["format_ok"] and not result["success"]):
        parts.append("Please correct the asset and zone to match the supplied record exactly.")
    return " ".join(parts)


def reference(record, style=None):
    style = style or record["preference"]
    a, z = record["facts"]["asset"], record["facts"]["zone"]
    if style == "json":
        return json.dumps(record["facts"])
    if style == "bullets":
        return f"- asset: {a}\n- zone: {z}"
    if style == "table":
        return f"| Field | Value |\n| --- | --- |\n| asset | {a} |\n| zone | {z} |"
    if style == "plain":
        return f"asset: {a}\nzone: {z}"
    raise ValueError(style)


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for split, rows in build().items():
        path = root / f"{split}.json"
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        manifest[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "count": len(rows)}
    manifest["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    prepare(parser.parse_args().root)
