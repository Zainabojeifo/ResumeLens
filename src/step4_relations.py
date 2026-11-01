"""
STEP 4 – RELATIONSHIP EXTRACTION
The NER model gives me separate pieces: a job title here, a company there, some
dates. In this step I connect the pieces that belong together:

  * JOB_TITLE  <->  COMPANY  <->  start DATE / end DATE  <->  SKILLS used in that job
  * DEGREE     <->  COLLEGE  <->  graduation DATE

How: I use the section of the resume each entity sits in (from Step 2) and the
order the entities appear in. Inside the Experience section a resume is a list of
"entries"; each entry has at most one title, one company and two dates. When I meet
a second title (or a second company, or a third date) a new entry has started.
This works for all common layouts: "Title at Company", "Company / Title | dates",
"dates / Title - Company", and so on.
"""
from step2_clean import find_sections, section_at


def _new_job():
    return {"title": None, "company": None, "dates": [], "skills": [], "_skill_after_header": False}


def _job_is_started(job):
    return job["title"] is not None or job["company"] is not None


def link_jobs(entities):
    """entities: list of spaCy spans from the Experience section, in reading order."""
    jobs, job = [], _new_job()

    def push():
        if _job_is_started(job):
            jobs.append(job)

    for ent in entities:
        label = ent.label_
        if label in ("JOB_TITLE", "COMPANY"):
            key = "title" if label == "JOB_TITLE" else "company"
            # a second title/company, or a title after the bullets -> new entry
            if job[key] is not None or job["_skill_after_header"]:
                push()
                job = _new_job()
            job[key] = ent.text
        elif label == "DATE":
            new_entry = len(job["dates"]) >= 2 or (
                _job_is_started(job) and job["dates"] and job["_skill_after_header"])
            if new_entry:
                push()
                job = _new_job()
            job["dates"].append(ent.text)
        elif label == "SKILL" and (_job_is_started(job) or job["dates"]):
            job["skills"].append(ent.text)
            if _job_is_started(job):
                job["_skill_after_header"] = True
    push()

    return [{
        "title": j["title"], "company": j["company"],
        "start": j["dates"][0] if j["dates"] else None,
        "end": j["dates"][1] if len(j["dates"]) > 1 else None,
        "skills": sorted(set(j["skills"]), key=str.lower),
    } for j in jobs]


def link_education(entities):
    records, rec = [], {"degree": None, "college": None, "year": None}

    def push():
        if rec["degree"] or rec["college"]:
            records.append(dict(rec))

    previous = None
    for ent in entities:
        key = {"DEGREE": "degree", "COLLEGE": "college", "DATE": "year"}.get(ent.label_)
        if key is None:
            continue
        if key == "year" and previous == "year":
            rec["year"] = ent.text   # "2017 - 2020": the second date is the graduation date
        else:
            if rec[key] is not None:   # second degree / college / date -> next record
                push()
                rec = {"degree": None, "college": None, "year": None}
            rec[key] = ent.text
        previous = key
    push()
    return records


def extract_relations(doc):
    """Group the entities of a parsed resume (spaCy Doc) into jobs and education."""
    sections = find_sections(doc.text)
    by_section = {}
    for ent in doc.ents:
        by_section.setdefault(section_at(sections, ent.start_char), []).append(ent)

    experience = by_section.get("experience", [])
    education = by_section.get("education", [])

    # Fallback for resumes without clear headings: use the whole document
    if not experience:
        experience = [e for e in doc.ents if e.label_ in ("JOB_TITLE", "COMPANY", "DATE", "SKILL")
                      and section_at(sections, e.start_char) not in ("education", "skills", "summary")]
    if not education:
        education = [e for e in doc.ents if e.label_ in ("DEGREE", "COLLEGE")]

    return {"jobs": link_jobs(experience), "education": link_education(education)}


if __name__ == "__main__":
    import json
    import sys

    import spacy

    from config import MODEL_DIR
    from step1_extract import extract_text
    from step2_clean import clean_text

    nlp = spacy.load(MODEL_DIR)
    doc = nlp(clean_text(extract_text(sys.argv[1])))
    print(json.dumps(extract_relations(doc), indent=2, ensure_ascii=False))
