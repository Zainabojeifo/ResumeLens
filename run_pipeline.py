"""
Run the whole ResumeLens project from start to finish:

    python run_pipeline.py            # everything (about 15 minutes on a laptop)
    python run_pipeline.py --skip-train   # reuse the saved model

Or parse one resume with the trained model:

    python run_pipeline.py --parse path/to/resume.pdf
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"


def check_setup():
    """Stop early with a clear message if something is missing."""
    missing = [pkg for pkg, module in [
        ("spacy", "spacy"), ("pytesseract", "pytesseract"), ("opencv-python", "cv2"),
        ("pdfplumber", "pdfplumber"), ("pymupdf", "pymupdf"), ("python-docx", "docx"),
        ("reportlab", "reportlab"), ("Faker", "faker"), ("pandas", "pandas"),
        ("scikit-learn", "sklearn"), ("scipy", "scipy"), ("matplotlib", "matplotlib"),
    ] if importlib.util.find_spec(module) is None]
    if missing:
        sys.exit("Missing Python packages: " + ", ".join(missing) +
                 "\nRun:  pip install -r requirements.txt")

    sys.path.insert(0, str(SRC))
    from step1_extract import TESSERACT_HELP, find_tesseract
    tesseract = find_tesseract()
    if not tesseract:
        sys.exit(TESSERACT_HELP)
    print(f"Setup OK. Tesseract: {tesseract}")


def run(script, *args):
    print(f"\n{'=' * 70}\n>>> {script} {' '.join(args)}\n{'=' * 70}", flush=True)
    # PYTHONUTF8=1 makes Windows read/write files and print as UTF-8 (accents, dashes)
    env = {**os.environ, "PYTHONUTF8": "1"}
    subprocess.run([sys.executable, str(SRC / script), *args], cwd=SRC, check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--parse", help="parse a single resume file and print the profile")
    args = parser.parse_args()

    if args.parse:
        sys.path.insert(0, str(SRC))
        from step5_integrate import parse_resume
        profile = parse_resume(args.parse)
        profile.pop("clean_text"); profile.pop("entities")
        print(json.dumps(profile, indent=2, ensure_ascii=False))
        return

    check_setup()
    run("generate_dataset.py")       # labelled data
    run("render_test_files.py")      # test resumes as PDF / Word / scanned image
    run("step3_ner.py", *(["--test"] if args.skip_train else []))   # Step 3 (uses Steps 1-2)
    run("step5_integrate.py")        # Steps 1-5 end to end + matching
    run("step6_report.py")           # Step 6 charts + report


if __name__ == "__main__":
    main()
