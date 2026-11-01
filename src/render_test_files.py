"""
Save every test resume as three kinds of file, so Step 1 is tested on real files:
  * <id>.pdf   – a normal, text-based PDF
  * <id>.docx  – a Word document
  * <id>.png   – a "scanned" page: the PDF turned into an image, then slightly
                 rotated, blurred and speckled with noise, like a phone scan

Run:  python src/render_test_files.py
"""
import json
import random

import cv2
import numpy as np
from docx import Document
from docx.shared import Pt
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from config import DATA_DIR, RAW_DIR, SEED
from step1_extract import render_pdf_pages, write_image


def write_pdf(text, path):
    width, height = A4
    margin = 50
    c = canvas.Canvas(str(path), pagesize=A4)
    y = height - margin
    lines = text.splitlines()
    for i, line in enumerate(lines):
        bold = i == 0 or (line.strip() and line.strip().isupper())
        font = "Helvetica-Bold" if bold else "Helvetica"
        size = 14 if i == 0 else 10.5
        # shrink a very long line instead of wrapping it
        while stringWidth(line, font, size) > width - 2 * margin and size > 6:
            size -= 0.5
        if y < margin:
            c.showPage()
            y = height - margin
        c.setFont(font, size)
        c.drawString(margin, y, line)
        y -= size + 5
    c.save()


def write_docx(text, path):
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    for i, line in enumerate(text.splitlines()):
        p = doc.add_paragraph()
        run = p.add_run(line)
        if i == 0:
            run.bold = True
            run.font.size = Pt(16)
        p.paragraph_format.space_after = Pt(2)
    doc.save(str(path))


def write_scan(pdf_path, png_path, rng):
    page = render_pdf_pages(pdf_path, dpi=200, max_pages=1)[0]
    img = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    # slight rotation, like a page placed crookedly on a scanner
    angle = rng.uniform(-2.0, 2.0)
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=255)

    # blur + grey background + random noise
    img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.4, 1.0))
    img = (img.astype(np.float32) * 0.85 + 25)
    noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, rng.uniform(8, 18), img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    write_image(png_path, img)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    docs = [json.loads(line) for line in open(DATA_DIR / "test.jsonl", encoding="utf-8")]
    for doc in docs:
        pdf_path = RAW_DIR / f"{doc['id']}.pdf"
        write_pdf(doc["text"], pdf_path)
        write_docx(doc["text"], RAW_DIR / f"{doc['id']}.docx")
        write_scan(pdf_path, RAW_DIR / f"{doc['id']}.png", rng)
    print(f"Saved {len(docs)} test resumes x 3 formats to {RAW_DIR}")


if __name__ == "__main__":
    main()
