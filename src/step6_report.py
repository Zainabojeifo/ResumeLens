"""
STEP 6 – PRESENTATION
I turn the saved results into charts and a written report (results/REPORT.md)
that I can use for my presentation.

Run:  python src/step6_report.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import MODEL_DIR, RESULTS_DIR  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua (fixed order)
FIELD_NAMES = {
    "name": "Name", "email": "Email", "phone": "Phone", "location": "Location",
    "skills": "Skills", "job_titles": "Job titles", "companies": "Companies",
    "degrees": "Degrees", "colleges": "Colleges", "job_links": "Title↔Company links",
    "job_links_with_dates": "Title↔Company↔Dates", "education_links": "Degree↔College links",
}


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)


def score(entry):
    return entry.get("f1", entry.get("accuracy"))


def chart_training(history, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    epochs = [h["epoch"] for h in history]
    f1 = [h["dev_f1"] for h in history]
    ax.plot(epochs, f1, color=SERIES[0], lw=2, marker="o", ms=4)
    best = max(range(len(f1)), key=f1.__getitem__)
    ax.annotate(f"best F1 {f1[best]:.3f} (epoch {epochs[best]})", (epochs[best], f1[best]),
                xytext=(0, -28), textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_xlabel("Epoch", color=MUTED); ax.set_ylabel("Dev F1", color=MUTED)
    ax.set_ylim(0, 1.05); ax.grid(axis="y", color=GRID, lw=0.8)
    style(ax, "NER training: F1 on the dev set after each epoch")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def chart_ner(ner, path):
    per = ner["per_entity"]
    labels = list(per)[::-1]
    values = [per[k]["f"] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, values, color=SERIES[0], height=0.6)
    for y, v in enumerate(values):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", color=INK, fontsize=8)
    ax.set_xlim(0, 1.1); ax.set_xlabel("F1 score", color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    style(ax, "NER test F1 by entity (100 unseen resumes, clean text)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def grouped_bars(groups, series_names, fields, title, path, ylim=(0, 1.05)):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    n = len(series_names)
    width = 0.8 / n
    for i, name in enumerate(series_names):
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(fields))]
        ax.bar(xs, [score(groups[name][f]) for f in fields], width * 0.92,
               color=SERIES[i], label=name)
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels([FIELD_NAMES[f] for f in fields], rotation=30, ha="right")
    ax.set_ylim(*ylim); ax.set_ylabel("Accuracy / F1", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, ncol=n, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    style(ax, title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def table(groups, columns, fields):
    lines = ["| Field | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for f in fields:
        kind = "acc." if "accuracy" in groups[columns[0]][f] else "F1"
        lines.append(f"| {FIELD_NAMES[f]} ({kind}) | " +
                     " | ".join(f"{score(groups[c][f]):.3f}" for c in columns) + " |")
    return "\n".join(lines)


def load_history():
    """Training history from results/, or the copy saved next to the model.
    Returns None if neither exists (e.g. the model was trained elsewhere)."""
    for path in (RESULTS_DIR / "training_history.json", MODEL_DIR / "training_history.json"):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    print("Note: no training_history.json found - the training section is left out.")
    return None


def main():
    history = load_history()
    ner = json.loads((RESULTS_DIR / "ner_test_scores.json").read_text(encoding="utf-8"))
    e2e = json.loads((RESULTS_DIR / "end_to_end_scores.json").read_text(encoding="utf-8"))
    fields = list(FIELD_NAMES)

    if history:
        chart_training(history, RESULTS_DIR / "chart_training.png")
    chart_ner(ner, RESULTS_DIR / "chart_ner_f1.png")
    by_format = e2e["field_accuracy_by_format"]
    grouped_bars({"PDF": by_format["pdf"], "Word": by_format["docx"], "Scanned image": by_format["scanned"]},
                 ["PDF", "Word", "Scanned image"], fields,
                 "End-to-end accuracy by file type (100 test resumes)", RESULTS_DIR / "chart_by_format.png")
    ch = e2e["challenge_set"]
    grouped_bars({"Generated test set (PDF)": by_format["pdf"], "Hand-written challenge set": ch["text"]},
                 ["Generated test set (PDF)", "Hand-written challenge set"], fields,
                 "Generated test resumes vs hand-written resumes in new layouts",
                 RESULTS_DIR / "chart_challenge.png")

    o = ner["overall"]
    sk = ner["skill_recall_seen_vs_unseen"]
    if history:
        best = max(history, key=lambda h: h["dev_f1"])
        training_lines = (f"* Blank spaCy English pipeline + NER, trained {len(history)} epochs, "
                          f"early stopping on dev F1.\n"
                          f"* Best dev F1: **{best['dev_f1']:.3f}** at epoch {best['epoch']}.")
        training_chart = "![training](chart_training.png)"
    else:
        training_lines = ("* Blank spaCy English pipeline + NER, early stopping on dev F1 "
                          "(training history not available for this run).")
        training_chart = ""
    report = f"""# ResumeLens – Results

## 1. What I built
A pipeline that turns a resume file (PDF, Word or scanned image) into a structured
candidate profile, then scores the candidate against job descriptions:

**File → Step 1 extract (pdfplumber / python-docx / OpenCV + Tesseract OCR) → Step 2 clean
→ Step 3 spaCy NER → Step 4 link entities → Step 5 profile + job match.**

## 2. Data
* {e2e['n_test_resumes']} test resumes, plus 450 training and 100 dev resumes, generated with
  known labels (10 entity types). Each split uses different random people, companies and schools.
* 12 skills appear **only** in the test set, to check the model learns from context.
* Every test resume was saved as PDF, Word and a noisy, slightly rotated "scan" (PNG).
* 6 extra resumes written by hand in layouts the generator never produces (challenge set).

## 3. NER model training
{training_lines}
* Augmentation: 20% of training skills were replaced by invented tool names, which raised
  recall on never-seen skills from 0.858 (first run, no augmentation) to {sk['unseen']:.3f}.

{training_chart}

## 4. NER test results (unseen resumes, clean text)
Overall precision **{o['precision']:.3f}**, recall **{o['recall']:.3f}**, F1 **{o['f1']:.3f}**.
Skill recall: seen in training {sk['seen']:.3f}, never seen {sk['unseen']:.3f}.

![ner](chart_ner_f1.png)

## 5. End-to-end results (file → profile)
Exact match after ignoring case, accents and punctuation. A link counts as correct only
if both sides are right and connected to each other.

{table(by_format, ['pdf', 'docx', 'scanned'], fields)}

OCR text accuracy on the scans: **{e2e['text_accuracy_by_format']['scanned']:.4f}**
(share of characters matching the original). Time per resume: PDF
{e2e['seconds_per_resume']['pdf']}s, Word {e2e['seconds_per_resume']['docx']}s,
scan {e2e['seconds_per_resume']['scanned']}s.

![formats](chart_by_format.png)

## 6. Challenge set – the honest real-world estimate
{table(ch, ['text', 'scanned'], fields)}

![challenge](chart_challenge.png)

## 7. Candidate–job matching
Score = 70% share of required skills + 30% TF-IDF text similarity. I compared the ranking
built from **parsed** profiles with the ranking built from the **true** profiles (Spearman):

| Role | PDF | Word | Scanned |
|---|---|---|---|
""" + "\n".join(
        f"| {role} | " + " | ".join(f"{e2e['matching_agreement_by_format'][f][role]['spearman']:.3f}"
                                      for f in ['pdf', 'docx', 'scanned']) + " |"
        for role in e2e["matching_agreement_by_format"]["pdf"]) + """

## 8. What the results mean, and limitations
* On resumes that look like the training data the parser is close to perfect for PDF and
  Word files. Scans lose a few points because OCR misreads some characters (e.g. `5` → `S`
  in emails, dropped accents), and one wrong letter makes a field "wrong" under exact match.
* The generated data comes from templates, so the generated test set is easier than real
  life. The **challenge set** shows the real gap: on layouts the model never saw ("Name:",
  "Company: / Role: / Duration:", "Title, Company (dates)" with unfamiliar company names),
  company and job-link accuracy drop to about 50%.
* The challenge set has only 6 resumes, so its numbers are a rough indication, not a
  precise measurement.

## 9. Next steps
1. Label 200–300 real, anonymised resumes (e.g. with Label Studio or Doccano) and fine-tune.
2. Add more layouts to the generator (label-value fields, tables, two-column resumes).
3. Add a skills dictionary (spaCy EntityRuler) next to the model for known skills.
4. Use layout information (position, font size) from the PDF, not only the text.
"""
    (RESULTS_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print("Wrote results/REPORT.md and 4 charts")


if __name__ == "__main__":
    main()
