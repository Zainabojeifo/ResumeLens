"""
STEP 1 – DATA EXTRACTION
I pull the raw text out of a resume, whatever format it comes in.

  * .pdf  -> read the text layer with pdfplumber. If the PDF has no text layer
             (a scanned PDF), I turn each page into an image and use OCR instead.
  * .docx -> read the paragraphs with python-docx.
  * .png / .jpg / .tiff -> clean the image with OpenCV, then run Tesseract OCR.

Only one program must be installed outside Python: Tesseract OCR. PDFs are turned
into images with PyMuPDF, a normal pip package, so Poppler is NOT needed.

Usage:
    from step1_extract import extract_text
    text = extract_text("my_resume.pdf")
"""
import os
import shutil
from pathlib import Path

import cv2
import pymupdf  # PDF -> image, no Poppler needed
import numpy as np
import pdfplumber
import pytesseract
from docx import Document

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# --psm 4: treat the page as one column of text lines of different sizes
TESSERACT_CONFIG = "--oem 1 --psm 4 -c preserve_interword_spaces=1"
TARGET_WIDTH = 2480


# ---------------------------------------------------------------------------
# Finding Tesseract (on Windows it is often installed but not on PATH)
# ---------------------------------------------------------------------------
WINDOWS_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


def find_tesseract():
    """Return the path to tesseract, or None. Checks (in order): the TESSERACT_CMD
    environment variable, PATH, then the usual Windows install folders."""
    candidates = [os.environ.get("TESSERACT_CMD"), shutil.which("tesseract")]
    if os.name == "nt":
        candidates += WINDOWS_TESSERACT_PATHS
    for path in candidates:
        if path and Path(path).is_file():
            return path
    return None


_TESSERACT = find_tesseract()
if _TESSERACT:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT

TESSERACT_HELP = (
    "Tesseract OCR was not found. Install it, then run again:\n"
    "  Windows: download the installer from https://github.com/UB-Mannheim/tesseract/wiki\n"
    "           (keep the default folder C:\\Program Files\\Tesseract-OCR)\n"
    "  macOS:   brew install tesseract\n"
    "  Ubuntu:  sudo apt install tesseract-ocr\n"
    "If you installed it somewhere else, set the TESSERACT_CMD environment variable\n"
    "to the full path of tesseract.exe."
)


# ---------------------------------------------------------------------------
# Reading / writing images and PDF pages (no Poppler needed)
# ---------------------------------------------------------------------------
def read_image(path):
    """cv2.imread fails on Windows paths with non-English letters; this does not."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path, image):
    ok, buffer = cv2.imencode(Path(path).suffix or ".png", image)
    if not ok:
        raise ValueError(f"Could not encode image for {path}")
    buffer.tofile(str(path))


def render_pdf_pages(path, dpi=300, max_pages=None):
    """Turn PDF pages into BGR images (numpy arrays) with PyMuPDF."""
    images = []
    with pymupdf.open(str(path)) as pdf:
        for i, page in enumerate(pdf):
            if max_pages is not None and i >= max_pages:
                break
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            images.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return images


# ---------------------------------------------------------------------------
# Image preparation for OCR
# ---------------------------------------------------------------------------
def find_skew_angle(binary, max_angle=5.0, step=0.25):
    """Find how much the page is tilted.

    I try small rotations and keep the one where the text lines are the most
    'horizontal' – that is when the row sums of dark pixels vary the most
    (sharp peaks for lines of text, empty valleys between them).
    """
    small = cv2.resize(binary, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    center = (w / 2, h / 2)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        score = np.var(rotated.sum(axis=1, dtype=np.float64))
        if score > best_score:
            best_angle, best_score = angle, score
    return best_angle


def preprocess_image(image):
    """Turn a photo or scan into a clean, straight, black-on-white image."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tesseract reads best at about 300 dpi (an A4 page ~2480px wide): upscale smaller images
    if image.shape[1] < TARGET_WIDTH:
        scale = TARGET_WIDTH / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # remove grain/speckle noise. Non-local means denoising keeps thin letters like
    # "l" and "1" intact better than a median filter (I tested both on 25 scans).
    denoised = cv2.fastNlMeansDenoising(image, None, h=15, templateWindowSize=7, searchWindowSize=21)

    # black and white: Otsu picks the best threshold automatically
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # straighten a tilted page
    angle = find_skew_angle(binary)
    if abs(angle) > 0.1:
        h, w = denoised.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        denoised = cv2.warpAffine(denoised, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return cv2.bitwise_not(binary)  # back to black text on white


def ocr_image(image):
    if not _TESSERACT:
        raise RuntimeError(TESSERACT_HELP)
    clean = preprocess_image(image)
    return pytesseract.image_to_string(clean, lang="eng", config=TESSERACT_CONFIG)


# ---------------------------------------------------------------------------
# One reader per file type
# ---------------------------------------------------------------------------
def extract_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text(x_tolerance=1.5) or "" for page in pdf.pages]
    text = "\n".join(pages)
    if len(text.strip()) > 50:
        return text, "pdf-text"

    # no text layer -> scanned PDF -> OCR every page
    text = "\n".join(ocr_image(img) for img in render_pdf_pages(path, dpi=300))
    return text, "pdf-ocr"


def extract_from_docx(path):
    doc = Document(path)
    lines = [p.text for p in doc.paragraphs]
    # text inside tables (some resume templates use them for layout)
    for table in doc.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(lines), "docx"


def extract_from_image(path):
    image = read_image(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return ocr_image(image), "image-ocr"


def extract_text(path, return_method=False):
    """Return the raw text of a resume file (and optionally which method was used)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text, method = extract_from_pdf(path)
    elif suffix == ".docx":
        text, method = extract_from_docx(path)
    elif suffix in IMAGE_TYPES:
        text, method = extract_from_image(path)
    elif suffix == ".txt":
        text, method = path.read_text(encoding="utf-8", errors="ignore"), "txt"
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return (text, method) if return_method else text


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        text, method = extract_text(f, return_method=True)
        print(f"===== {f}  [{method}] =====\n{text}")
