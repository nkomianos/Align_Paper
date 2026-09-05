"""PDF extraction and contact-sheet QA, never a substitute for visual inspection."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    from pypdf import PdfReader
    from PIL import Image, ImageDraw
    pdf = ROOT / "output/pdf/main.pdf"
    reader = PdfReader(pdf)
    pages = [page.extract_text() or "" for page in reader.pages]
    reference_pages = [i+1 for i,t in enumerate(pages) if re.search(r"\bREFERENCES\b", t)]
    assert reference_pages, "references section not found"
    assert reference_pages[0] <= 10, "main text extends beyond nine pages"
    assert all("Working draft" in t for t in pages), "missing truthful draft header"
    assert "Under review as" not in "\n".join(pages)
    assert "Paper under double-blind review" not in "\n".join(pages)
    assert "Not run" in "\n".join(pages)
    assert "??" not in "\n".join(pages), "unresolved citation/reference"
    log = (ROOT / "output/pdf/main.log").read_text(encoding="utf-8", errors="replace")
    issues = [line for line in log.splitlines() if "Overfull" in line or "undefined" in line or "Font Warning" in line]
    assert not issues, issues
    render = ROOT / "artifacts/paper_render_20260905"
    image_paths = sorted(render.glob("page-*.png"))
    assert len(image_paths) == len(pages), "render latest PDF before QA"
    for offset in range(0, len(image_paths), 4):
        sheet = Image.new("RGB", (1200, 1660), "#c9c9c9")
        draw = ImageDraw.Draw(sheet)
        for j,path in enumerate(image_paths[offset:offset+4]):
            with Image.open(path) as page:
                page.thumbnail((570, 780))
                x, y = 15+(j%2)*600, 30+(j//2)*830
                sheet.paste(page,(x,y))
                draw.text((x,y-20),f"Page {offset+j+1}",fill="black")
        sheet.save(render/f"contact-{offset//4+1}.png")
    result = {"pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "pages": len(pages),
              "references_begin_page": reference_pages[0], "main_text_within_nine_pages": True,
              "draft_and_unrun_status_present": True, "unresolved_references": False,
              "overfull_or_font_warnings": issues,
              "page_text_characters": [len(t) for t in pages],
              "visual_review": "Contact sheets generated; a separate actual image inspection is required."}
    (render / "qa.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
