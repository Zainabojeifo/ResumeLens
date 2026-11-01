# ResumeLens: An OCR and NLP Pipeline for Automated Resume Parsing and Candidate Profiling

ResumeLens turns a resume file (PDF, Word or a scanned image) into a structured candidate
profile and scores the candidate against job descriptions. It uses Tesseract OCR, OpenCV,
pdfplumber and a spaCy Named Entity Recognition model that I train myself.

```
resume file ─► 1 Extract ─► 2 Clean ─► 3 Find entities ─► 4 Link entities ─► 5 Profile + job match ─► 6 Report
               (OCR)        (normalise)  (spaCy NER)       (title↔company↔dates)
```

## Project structure

```
resumelens/
├── run_pipeline.py            # runs everything, or parses one file (--parse)
├── app.py                     # Streamlit demo: upload a resume, see the profile
├── requirements.txt
├── src/
│   ├── config.py              # paths, labels, dataset sizes
│   ├── generate_dataset.py    # labelled train / dev / test resumes
│   ├── render_test_files.py   # test resumes as PDF, Word and noisy scans
│   ├── step1_extract.py       # STEP 1  text extraction + OCR
│   ├── step2_clean.py         # STEP 2  cleaning + section detection
│   ├── step3_ner.py           # STEP 3  train + test the spaCy NER model
│   ├── step4_relations.py     # STEP 4  relationship extraction
│   ├── step5_integrate.py     # STEP 5  full parser, profiles, matching, evaluation
│   └── step6_report.py        # STEP 6  charts + results/REPORT.md
├── data/challenge/            # 6 hand-written resumes + their true answers
├── models/resume_ner/         # the trained model (created by step 3)
└── results/                   # scores, profiles (JSON/CSV), charts, REPORT.md
```

## Setup

1. Python 3.10 or newer.
2. Install Tesseract OCR, the only program needed outside Python
   (Poppler is **not** needed; PDFs are turned into images with PyMuPDF, a pip package):
   * Windows: run the installer from https://github.com/UB-Mannheim/tesseract/wiki and keep the
     default folder `C:\Program Files\Tesseract-OCR`. The code finds it there automatically, even if
     it is not on PATH. If you installed it elsewhere, set the `TESSERACT_CMD` environment variable
     to the full path of `tesseract.exe`.
   * macOS: `brew install tesseract`
   * Ubuntu: `sudo apt install tesseract-ocr`
3. Install the Python packages (inside your virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
   `run_pipeline.py` checks the setup first and tells you exactly what is missing.

## Run it

```bash
python run_pipeline.py                     # whole project, about 15 minutes
python run_pipeline.py --skip-train        # reuse the saved model
python run_pipeline.py --parse my_cv.pdf   # parse one resume
streamlit run app.py                       # demo app
```

Each step can also be run on its own, for example `python src/step1_extract.py file.png`
prints the extracted text and `python src/step4_relations.py file.pdf` prints the linked jobs.

## How each step works

**Data.** Real resumes with labelled entities are private, so `generate_dataset.py` creates
650 varied resumes (different layouts, headings, date formats, bullet styles and names from
many countries) where the position of every entity is known. Train, dev and test use
different random people, companies and schools. 12 skills appear only in the test set.

**Step 1 – Extraction.** PDFs are read with pdfplumber; if a PDF has no text layer it is
OCR'd page by page. Word files are read with python-docx. Images go through OpenCV
(grayscale → upscale to ~300 dpi → non-local-means denoising → Otsu thresholding →
deskew) and then Tesseract.

**Step 2 – Cleaning.** Fixes PDF glyph codes, ligatures, quotes and dashes; turns all bullet
styles into `- ` and inline separators into ` | `; removes blank lines and stray OCR
symbols; re-joins broken sentences; detects sections. The same cleaning is applied to the
training data, with an offset map that moves the entity labels to the right place.

**Step 3 – Entity recognition.** A blank spaCy English pipeline with an NER component is
trained on 10 labels: NAME, EMAIL, PHONE, LOCATION, JOB_TITLE, COMPANY, DATE, DEGREE,
COLLEGE, SKILL. Training uses dropout, early stopping on dev F1 and averaged weights. 20%
of training skills are swapped for invented tool names so the model learns from context.

**Step 4 – Relationship extraction.** Within the Experience section, entities are grouped
into entries (one title, one company, up to two dates, the skills in its bullets). The
Education section is grouped into degree, college and graduation date.

**Step 5 – Integration and model building.** `parse_resume()` combines Steps 1–4 into one
profile (regex is used for emails and phones, NER as fallback). Profiles are saved to
JSON and CSV. A matching model scores candidates against job descriptions:
70% share of required skills + 30% TF-IDF similarity.

**Step 6 – Presentation.** `step6_report.py` writes `results/REPORT.md` with tables and charts.

## How I test it

* **NER test**: precision, recall and F1 per entity on 100 unseen resumes (spaCy scorer).
* **End-to-end test**: every test resume is parsed from its PDF, Word and scanned file,
  and each field is compared with the true answer (exact match, ignoring case, accents and
  punctuation). Links count only when both sides are correct and connected.
* **Challenge set**: 6 resumes I wrote by hand in layouts the generator never produces.
  This is the most realistic estimate of performance on resumes from the real world.
* **Matching test**: Spearman correlation between rankings from parsed vs true profiles.

See `results/REPORT.md` for the numbers.

## Limitations

The training data is generated from templates, so the generated test set is easier than
real resumes. The challenge set shows the gap: on unfamiliar layouts, company names and
job links are found only about half the time. Labelling a few hundred real, anonymised
resumes and fine-tuning is the most important next step.
