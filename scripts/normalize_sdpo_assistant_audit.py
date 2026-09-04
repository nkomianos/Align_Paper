"""Mechanically normalize an existing audit; no relabeling or inference."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    original=a.source.read_bytes();source=json.loads(original)
    records=[]
    for row in source["cases"]:
        record={"id":row["id"]}
        for condition in ("ordinary","explicit","teacher"):
            item=row[condition]
            record[condition]={"style_satisfactory":item["style_satisfactory"],
                "material_content_error":item["material_content_error"],
                "reason":"Style: "+item["style_reason"]+" Content: "+item["content_reason"]}
        records.append(record)
    result=dict(calibration_manifest_sha256=source["source_manifest_sha256"],reviewer_kind="assistant_model_audit",
        reviewer_identity=source["auditor"],rubric=source["rubric"],records=records,
        source_audit_sha256=hashlib.sha256(original).hexdigest(),method=source["method"],
        protocol_deviation=source["protocol_deviation"],approval=source["approval"],
        normalization="Field renaming and concatenation only; labels and source audit unchanged")
    with a.output.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2,allow_nan=False)
if __name__=="__main__":main()
