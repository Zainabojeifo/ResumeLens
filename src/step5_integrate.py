"""
STEP 5 – DATA INTEGRATION AND MODEL BUILDING
I put Steps 1-4 together into one function, `parse_resume(file)`, which returns a
structured candidate profile. Then I:

  1. parse every test resume in all three formats (PDF, Word, scanned image),
  2. save the profiles as JSON and CSV,
  3. measure how accurate each field is against the true answers,
  4. score every candidate against job descriptions (candidate–job matching) and
     check whether parsing mistakes change the ranking.

Run:  python src/step5_integrate.py
"""
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from functools import partial

import pandas as pd
import spacy
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DATA_DIR, MODEL_DIR, RAW_DIR, RESULTS_DIR
from step1_extract import extract_text
from step2_clean import clean_text, find_sections, section_at
from step4_relations import extract_relations

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\(?\d[\d\s().-]{7,}\d")

_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load(MODEL_DIR)
    return _NLP


# ---------------------------------------------------------------------------
# The full parser
# ---------------------------------------------------------------------------
def first_entity(doc, label, sections=None, prefer_section=None):
    ents = [e for e in doc.ents if e.label_ == label]
    if prefer_section and sections:
        preferred = [e for e in ents if section_at(sections, e.start_char) == prefer_section]
        ents = preferred or ents
    return ents[0].text if ents else None


def find_phone(text):
    for m in PHONE_RE.finditer(text):
        if len(re.sub(r"\D", "", m.group())) >= 9:
            return m.group().strip()
    return None


def build_profile(text, source=None, method=None):
    """Turn raw resume text into a structured profile (Steps 2, 3 and 4)."""
    cleaned = clean_text(text)                       # Step 2
    doc = get_nlp()(cleaned)                          # Step 3
    sections = find_sections(cleaned)
    relations = extract_relations(doc)                # Step 4

    # Emails and phones follow strict patterns, so a regex is the most reliable
    # tool; the NER prediction is the fallback.
    email = EMAIL_RE.search(cleaned)
    skills, seen = [], set()
    for ent in doc.ents:
        if ent.label_ == "SKILL" and ent.text.lower() not in seen:
            seen.add(ent.text.lower())
            skills.append(ent.text)

    return {
        "source": source,
        "extraction_method": method,
        "name": first_entity(doc, "NAME", sections, "header"),
        "email": email.group() if email else first_entity(doc, "EMAIL"),
        "phone": find_phone(cleaned) or first_entity(doc, "PHONE"),
        "location": first_entity(doc, "LOCATION", sections, "header"),
        "skills": skills,
        "jobs": relations["jobs"],
        "education": relations["education"],
        "entities": [{"text": e.text, "label": e.label_} for e in doc.ents],
        "clean_text": cleaned,
    }


def parse_resume(path):
    """Full pipeline for one file: extract -> clean -> NER -> relations -> profile."""
    text, method = extract_text(path, return_method=True)   # Step 1
    return build_profile(text, source=str(path), method=method)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def norm(value):
    """Compare values without caring about case, accents, punctuation or spacing."""
    if value is None:
        return ""
    # fold accents (é -> e): OCR often drops accents, which is a minor slip
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9+#]+", " ", value.lower()).strip()


def digits(value):
    return re.sub(r"\D", "", value or "")


def near(a, b, threshold=0.9):
    """'Near-exact': at least 90% of characters match (allows a small OCR slip)."""
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and SequenceMatcher(None, a, b).ratio() >= threshold


def set_counts(predicted, true, fuzzy=False):
    """True positives, number predicted, number true – for micro P/R/F1."""
    pred = [p for p in dict.fromkeys(predicted) if p]
    gold = [t for t in dict.fromkeys(true) if t]
    unmatched = list(gold)
    tp = 0
    for p in pred:
        for t in unmatched:
            if (near(p, t) if fuzzy else p == t):
                tp += 1
                unmatched.remove(t)
                break
    return tp, len(pred), len(gold)


def prf(tp, n_pred, n_true):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def evaluate_profiles(rows):
    """rows: list of (profile, truth). Returns accuracy / P-R-F1 for every field."""
    single = {"name": [0, 0], "email": [0, 0], "phone": [0, 0], "location": [0, 0]}
    single_near = {k: [0, 0] for k in single}
    sets = {k: [0, 0, 0] for k in ["skills", "job_titles", "companies", "degrees", "colleges",
                                    "job_links", "job_links_with_dates", "education_links"]}
    sets_near = {k: [0, 0, 0] for k in sets}

    for prof, truth in rows:
        for field in single:
            if field == "phone":
                ok = digits(prof["phone"]) == digits(truth["phone"])
                ok_near = ok
            else:
                ok = norm(prof[field]) == norm(truth[field])
                ok_near = ok or near(prof[field], truth[field])
            single[field][0] += ok; single[field][1] += 1
            single_near[field][0] += ok_near; single_near[field][1] += 1

        pj, tj = prof["jobs"], truth["jobs"]
        pe, te = prof["education"], truth["education"]
        pairs = {
            "skills": ([norm(s) for s in prof["skills"]], [norm(s) for s in truth["skills"]]),
            "job_titles": ([norm(j["title"]) for j in pj], [norm(j["title"]) for j in tj]),
            "companies": ([norm(j["company"]) for j in pj], [norm(j["company"]) for j in tj]),
            "degrees": ([norm(e["degree"]) for e in pe], [norm(e["degree"]) for e in te]),
            "colleges": ([norm(e["college"]) for e in pe], [norm(e["college"]) for e in te]),
            # a link is correct only if BOTH sides are right and connected together
            "job_links": ([norm(j["title"]) + " @ " + norm(j["company"]) for j in pj],
                          [norm(j["title"]) + " @ " + norm(j["company"]) for j in tj]),
            "job_links_with_dates": (
                [" | ".join([norm(j["title"]), norm(j["company"]), norm(j["start"]), norm(j["end"])]) for j in pj],
                [" | ".join([norm(j["title"]), norm(j["company"]), norm(j["start"]), norm(j["end"])]) for j in tj]),
            "education_links": ([norm(e["degree"]) + " @ " + norm(e["college"]) for e in pe],
                                [norm(e["degree"]) + " @ " + norm(e["college"]) for e in te]),
        }
        for key, (pred, true) in pairs.items():
            for store, fuzzy in ((sets, False), (sets_near, True)):
                tp, n_p, n_t = set_counts(pred, true, fuzzy)
                store[key][0] += tp; store[key][1] += n_p; store[key][2] += n_t

    result = {}
    for field in single:
        result[field] = {"accuracy": round(single[field][0] / single[field][1], 4),
                         "accuracy_near": round(single_near[field][0] / single_near[field][1], 4)}
    for key in sets:
        result[key] = {**prf(*sets[key]), "f1_near": prf(*sets_near[key])["f1"]}
    return result


# ---------------------------------------------------------------------------
# Candidate–job matching
# ---------------------------------------------------------------------------
JOB_DESCRIPTIONS = [
    {"role": "Data Scientist",
     "required_skills": ["Python", "Machine Learning", "SQL", "Statistics", "Scikit-learn"],
     "text": "We need a Data Scientist to build machine learning models in Python, query data "
             "with SQL and apply statistics. Experience with Scikit-learn and Pandas is a plus."},
    {"role": "Data Engineer",
     "required_skills": ["Python", "SQL", "Apache Spark", "Airflow", "ETL", "AWS"],
     "text": "Data Engineer to design ETL pipelines with Apache Spark and Airflow on AWS. "
             "Strong Python and SQL skills required."},
    {"role": "BI Analyst",
     "required_skills": ["SQL", "Tableau", "Power BI", "Microsoft Excel", "Data Visualization"],
     "text": "Business Intelligence Analyst to create dashboards in Tableau and Power BI, "
             "analyse data in SQL and Microsoft Excel, and present data visualization to managers."},
    {"role": "Backend Developer",
     "required_skills": ["Python", "Django", "REST APIs", "PostgreSQL", "Docker"],
     "text": "Backend Developer to build REST APIs with Python and Django, using PostgreSQL "
             "and Docker. Git and Linux experience expected."},
]


def profile_text(profile):
    parts = list(profile["skills"])
    for job in profile["jobs"]:
        parts += [job.get("title") or "", *(job.get("skills") or [])]
    return " ".join(p for p in parts if p)


def match_scores(profiles, jd):
    """Score = 70% share of required skills the candidate has + 30% text similarity."""
    required = {norm(s) for s in jd["required_skills"]}
    coverage = [len(required & {norm(s) for s in p["skills"]}) / len(required) for p in profiles]
    texts = [profile_text(p) for p in profiles]
    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit(texts + [jd["text"]])
    sims = cosine_similarity(vec.transform([jd["text"]]), vec.transform(texts))[0]
    return [round(0.7 * c + 0.3 * s, 4) for c, s in zip(coverage, sims)]


def evaluate_matching(parsed, truths, top_k=10):
    """Does the ranking from PARSED profiles agree with the ranking from TRUE profiles?"""
    out = {}
    for jd in JOB_DESCRIPTIONS:
        pred = match_scores(parsed, jd)
        true = match_scores(truths, jd)
        rho = spearmanr(pred, true).correlation
        top_pred = set(sorted(range(len(pred)), key=lambda i: -pred[i])[:top_k])
        top_true = set(sorted(range(len(true)), key=lambda i: -true[i])[:top_k])
        out[jd["role"]] = {"spearman": round(float(rho), 4),
                           f"top{top_k}_overlap": round(len(top_pred & top_true) / top_k, 4)}
    return out


# ---------------------------------------------------------------------------
# Challenge set: hand-written resumes in layouts the generator never produces
# ---------------------------------------------------------------------------
def evaluate_challenge():
    """6 resumes I wrote by hand (data/challenge). They use different wording and
    layouts ("Role: ...", "2020 - 2024 | Title | Company", grouped skills, education
    first), real-world companies and skills the model never saw. This is the most
    honest estimate of how the parser does on resumes from outside my generator.
    Each one is tested as plain text and as a noisy scan."""
    import random
    from render_test_files import write_pdf, write_scan

    folder = DATA_DIR / "challenge"
    truths = json.loads((folder / "challenge_truth.json").read_text(encoding="utf-8"))
    rng = random.Random(7)
    rows = {"text": [], "scanned": []}
    for cid, truth in truths.items():
        txt = folder / f"{cid}.txt"
        pdf, png = folder / f"{cid}.pdf", folder / f"{cid}.png"
        write_pdf(txt.read_text(encoding="utf-8"), pdf)
        write_scan(pdf, png, rng)
        rows["text"].append((parse_resume(txt), truth))
        rows["scanned"].append((parse_resume(png), truth))
    return {fmt: evaluate_profiles(r) for fmt, r in rows.items()}


# ---------------------------------------------------------------------------
# Run everything on the test set
# ---------------------------------------------------------------------------
def main():
    records = [json.loads(line) for line in open(DATA_DIR / "test.jsonl", encoding="utf-8")]
    truths = {r["id"]: r["truth"] for r in records}
    for t in truths.values():  # truth profiles need the same shape for matching
        t.setdefault("skills", [])

    formats = {"pdf": ".pdf", "docx": ".docx", "scanned": ".png"}
    all_profiles, results, timings, text_accuracy = {}, {}, {}, {}
    workers = os.cpu_count() or 1
    for fmt, ext in formats.items():
        start = time.time()
        paths = [RAW_DIR / f"{r['id']}{ext}" for r in records]
        # Step 1 (OCR is slow) runs in parallel; Steps 2-4 run here
        with ProcessPoolExecutor(max_workers=workers) as pool:
            extracted = list(pool.map(partial(extract_text, return_method=True), paths))
        profiles = []
        for r, path, (text, method) in zip(records, paths, extracted):
            prof = build_profile(text, source=str(path), method=method)
            prof["id"] = r["id"]
            profiles.append(prof)
        timings[fmt] = round((time.time() - start) / len(records), 3)
        all_profiles[fmt] = profiles
        results[fmt] = evaluate_profiles([(p, truths[p["id"]]) for p in profiles])
        # Step 1 quality: how much of the true text survives extraction + cleaning
        text_accuracy[fmt] = round(sum(
            SequenceMatcher(None, p["clean_text"], clean_text(r["text"]), autojunk=False).ratio()
            for p, r in zip(profiles, records)) / len(records), 4)
        print(f"Parsed {len(profiles)} {fmt} resumes ({timings[fmt]}s each, "
              f"text accuracy {text_accuracy[fmt]:.4f})")

    matching = {fmt: evaluate_matching(all_profiles[fmt], [truths[r["id"]] for r in records])
                for fmt in formats}

    # ---- save profiles ----------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in p.items() if k not in ("entities", "clean_text")}
            for p in all_profiles["scanned"]]
    (RESULTS_DIR / "profiles_scanned.json").write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")
    table = pd.DataFrame([{
        "id": p["id"], "name": p["name"], "email": p["email"], "phone": p["phone"],
        "location": p["location"], "skills": "; ".join(p["skills"]),
        "latest_title": p["jobs"][0]["title"] if p["jobs"] else None,
        "latest_company": p["jobs"][0]["company"] if p["jobs"] else None,
        "n_jobs": len(p["jobs"]),
        "highest_degree": p["education"][0]["degree"] if p["education"] else None,
        **{f"match_{jd['role']}": s for jd, s in
           zip(JOB_DESCRIPTIONS, [None] * len(JOB_DESCRIPTIONS))},
    } for p in all_profiles["scanned"]])
    for jd in JOB_DESCRIPTIONS:
        table[f"match_{jd['role']}"] = match_scores(all_profiles["scanned"], jd)
    table.to_csv(RESULTS_DIR / "candidate_profiles.csv", index=False, encoding="utf-8-sig")  # -sig: Excel shows accents correctly

    challenge = evaluate_challenge()

    summary = {"field_accuracy_by_format": results, "matching_agreement_by_format": matching,
               "challenge_set": challenge,
               "text_accuracy_by_format": text_accuracy,
               "seconds_per_resume": timings, "n_test_resumes": len(records)}
    (RESULTS_DIR / "end_to_end_scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- print a readable summary -----------------------------------------
    print("\nEND-TO-END TEST (file -> profile), exact match")
    print(f"{'Field':<22}" + "".join(f"{f:>10}" for f in formats))
    for field in ["name", "email", "phone", "location"]:
        print(f"{field + ' (accuracy)':<22}" + "".join(f"{results[f][field]['accuracy']:>10.3f}" for f in formats))
    for field in ["skills", "job_titles", "companies", "degrees", "colleges",
                  "job_links", "job_links_with_dates", "education_links"]:
        print(f"{field + ' (F1)':<22}" + "".join(f"{results[f][field]['f1']:>10.3f}" for f in formats))
    print("\nCHALLENGE SET (6 hand-written resumes, new layouts), exact match")
    print(f"{'Field':<22}{'text':>10}{'scanned':>10}")
    for field in ["name", "email", "phone", "location"]:
        print(f"{field + ' (accuracy)':<22}" + "".join(f"{challenge[f][field]['accuracy']:>10.3f}" for f in challenge))
    for field in ["skills", "job_titles", "companies", "degrees", "colleges",
                  "job_links", "job_links_with_dates", "education_links"]:
        print(f"{field + ' (F1)':<22}" + "".join(f"{challenge[f][field]['f1']:>10.3f}" for f in challenge))

    print("\nMATCHING: agreement between ranking from parsed vs true profiles (Spearman)")
    for jd in JOB_DESCRIPTIONS:
        print(f"{jd['role']:<22}" + "".join(f"{matching[f][jd['role']]['spearman']:>10.3f}" for f in formats))


if __name__ == "__main__":
    main()
