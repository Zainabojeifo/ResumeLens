"""Shared settings and paths for the ResumeLens project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_files"          # test resumes as PDF / DOCX / scanned PNG
MODEL_DIR = ROOT / "models" / "resume_ner"
RESULTS_DIR = ROOT / "results"

SEED = 42

# The entities my NER model learns to recognise
LABELS = [
    "NAME", "EMAIL", "PHONE", "LOCATION",
    "JOB_TITLE", "COMPANY", "DATE",
    "DEGREE", "COLLEGE", "SKILL",
]

# How many resumes to generate for each split
N_TRAIN, N_DEV, N_TEST = 450, 100, 100
