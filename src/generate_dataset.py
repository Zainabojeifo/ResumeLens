"""
Build a labelled resume dataset.

Real resumes with hand-labelled entities are hard to get (privacy), so I generate
realistic, varied resumes where I know exactly where every entity is. Each resume
comes with:
  * text      – the resume as plain text
  * entities  – (start, end, label) character spans, used to train / test the NER model
  * truth     – the structured answer (name, jobs, education, skills ...), used to
                test the full pipeline end to end

To keep the test honest:
  * the test set uses different random people, companies and schools from training
  * a group of skills (HELD_OUT_SKILLS) never appears in training, so I can measure
    whether the model generalises to skills it has never seen
  * to help it generalise, 20% of skills in the training set are invented tool
    names (augmentation) so the model learns from context, not from a word list
  * the test resumes are also saved as PDF, Word and noisy "scanned" images, so the
    end-to-end test goes through real extraction and OCR

Run:  python src/generate_dataset.py
"""
import json
import random

from faker import Faker

from config import DATA_DIR, N_DEV, N_TEST, N_TRAIN, SEED

# ----------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------
SKILLS = [
    "Python", "SQL", "R", "Java", "JavaScript", "TypeScript", "C++", "Go", "Scala",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Data Analysis",
    "Data Visualization", "Statistics", "Pandas", "NumPy", "Scikit-learn",
    "TensorFlow", "PyTorch", "Keras", "spaCy", "OpenCV", "Tableau", "Power BI",
    "Microsoft Excel", "Apache Spark", "Hadoop", "Airflow", "Docker", "Kubernetes",
    "AWS", "Azure", "Google Cloud", "Git", "Linux", "PostgreSQL", "MySQL", "MongoDB",
    "React", "Node.js", "Django", "Flask", "FastAPI", "REST APIs", "HTML", "CSS",
    "Agile", "Scrum", "Project Management", "Jira", "Figma", "SEO", "Google Analytics",
    "Financial Modelling", "Accounting", "Customer Service", "Public Speaking",
    "A/B Testing", "ETL", "Snowflake", "Looker", "MATLAB", "SAS", "Selenium",
]
# Never shown to the model during training – used to test generalisation
HELD_OUT_SKILLS = [
    "Rust", "Kotlin", "Dbt", "Terraform", "Redis", "Elasticsearch", "LightGBM",
    "XGBoost", "Streamlit", "Power Automate", "BigQuery", "Kafka",
]

JOB_TITLES = [
    "Data Scientist", "Senior Data Scientist", "Data Analyst", "Junior Data Analyst",
    "Machine Learning Engineer", "Data Engineer", "Software Engineer",
    "Senior Software Engineer", "Backend Developer", "Frontend Developer",
    "Full Stack Developer", "Business Analyst", "Business Intelligence Analyst",
    "Product Manager", "Project Manager", "Research Assistant", "AI Engineer",
    "DevOps Engineer", "Cloud Engineer", "Database Administrator", "QA Engineer",
    "Financial Analyst", "Marketing Analyst", "Operations Analyst", "IT Support Specialist",
    "Systems Analyst", "Data Science Intern", "Software Engineering Intern",
    "Lead Data Engineer", "Analytics Manager", "NLP Engineer", "Computer Vision Engineer",
    "Customer Success Manager", "Digital Marketing Specialist", "Accountant",
]

DEGREES = [
    "BSc Computer Science", "B.Sc. Computer Science", "BSc Statistics",
    "BSc Mathematics", "BSc Economics", "BEng Electrical Engineering",
    "B.Tech Information Technology", "BA Business Administration",
    "MSc Data Science", "M.Sc. Artificial Intelligence", "MSc Computer Science",
    "MBA", "Master of Business Administration", "Bachelor of Science in Physics",
    "Master of Science in Statistics", "PhD Computer Science", "PhD Machine Learning",
    "HND Computer Science", "BSc Accounting", "MSc Finance",
]

UNIVERSITY_PATTERNS = [
    "University of {city}", "{city} State University", "{city} Institute of Technology",
    "{city} Polytechnic", "{surname} University", "{city} College of Science",
]
AFRICAN_FIRST = ["Zainab", "Chinedu", "Aisha", "Tunde", "Ngozi", "Emeka", "Fatima",
                 "Kwame", "Amara", "Ibrahim", "Halima", "Segun", "Kemi", "Yusuf",
                 "Adaeze", "Musa", "Folake", "Kofi", "Wanjiru", "Thabo"]
AFRICAN_LAST = ["Idrisu", "Okafor", "Bello", "Adeyemi", "Mensah", "Okonkwo", "Abubakar",
                "Eze", "Balogun", "Mwangi", "Nwosu", "Danjuma", "Ogunleye", "Boateng",
                "Ndlovu", "Afolabi", "Usman", "Chukwu", "Owusu", "Lawal"]
COUNTRIES = ["Nigeria", "Ghana", "Kenya", "United Kingdom", "USA", "Canada", "India",
             "South Africa", "Germany", "France", "Netherlands", "Spain"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_LONG = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
LOCALES = ["en_US", "en_GB", "en_IN", "fr_FR", "de_DE", "es_ES", "nl_NL"]

# Section headings vary between resumes, like in real life
HEADINGS = {
    "summary": ["SUMMARY", "Profile", "PROFESSIONAL SUMMARY", "About Me", "Objective"],
    "experience": ["EXPERIENCE", "Work Experience", "WORK HISTORY",
                   "Professional Experience", "EMPLOYMENT HISTORY"],
    "education": ["EDUCATION", "Education", "ACADEMIC BACKGROUND", "Qualifications"],
    "skills": ["SKILLS", "Technical Skills", "CORE SKILLS", "Skills & Tools", "Key Skills"],
}
BULLETS = ["•", "-", "*", "•"]

ACHIEVEMENT_TEMPLATES = [
    "Built {a} models that improved forecast accuracy by {n}%.",
    "Developed dashboards in {a} used by {n} managers every week.",
    "Automated reporting with {a} and {b}, saving {n} hours per month.",
    "Cleaned and analysed large datasets using {a} and {b}.",
    "Designed data pipelines with {a}, cutting processing time by {n}%.",
    "Led a team of {k} to deliver a {a} project on time.",
    "Deployed services with {a} and {b} for {n}k daily users.",
    "Worked closely with stakeholders to define KPIs.",
    "Mentored {k} junior colleagues and ran weekly code reviews.",
    "Used {a} to run experiments that lifted conversion by {n}%.",
    "Migrated legacy systems to {a}, reducing costs by {n}%.",
]
SUMMARY_TEMPLATES = [
    "{title} with {y} years of experience in {a} and {b}.",
    "Results-driven {title} skilled in {a}, {b} and teamwork.",
    "Curious {title} who enjoys solving problems with {a}.",
    "Detail-oriented professional with {y}+ years using {a} and {b}.",
]


class ResumeBuilder:
    """Writes resume text piece by piece and remembers where every entity is."""

    def __init__(self):
        self.text = ""
        self.entities = []

    def add(self, piece, label=None):
        start = len(self.text)
        self.text += piece
        if label:
            self.entities.append((start, len(self.text), label))

    def newline(self, n=1):
        self.text += "\n" * n

    def add_template(self, template, values):
        """Fill a template like 'Used {a} and {b}' where values maps key->(text, label)."""
        parts = template.replace("{", "\x00{").replace("}", "}\x00").split("\x00")
        for part in parts:
            if part.startswith("{") and part.endswith("}"):
                text, label = values[part[1:-1]]
                self.add(text, label)
            else:
                self.add(part)


def random_date(rng, year, style):
    month = rng.randrange(12)
    if style == "short":
        return f"{MONTHS[month]} {year}"
    if style == "long":
        return f"{MONTHS_LONG[month]} {year}"
    if style == "numeric":
        return f"{month + 1:02d}/{year}"
    return str(year)


def make_person(rng, fakers):
    if rng.random() < 0.3:
        first, last = rng.choice(AFRICAN_FIRST), rng.choice(AFRICAN_LAST)
    else:
        f = rng.choice(fakers)
        first, last = f.first_name(), f.last_name()
    name = f"{first} {last}"
    domain = rng.choice(["gmail.com", "yahoo.com", "outlook.com", "mail.com", "proton.me"])
    sep = rng.choice([".", "_", ""])
    user = f"{first}{sep}{last}".lower().replace(" ", "").replace("'", "")
    user = user.encode("ascii", "ignore").decode() + (str(rng.randint(1, 99)) if rng.random() < 0.4 else "")
    phone = rng.choice([
        f"+234 {rng.randint(700, 909)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}",
        f"+1 ({rng.randint(200, 999)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
        f"+44 {rng.randint(7000, 7999)} {rng.randint(100000, 999999)}",
        f"0{rng.randint(700, 909)}{rng.randint(1000000, 9999999)}",
        f"{rng.randint(200, 999)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
    ])
    city = rng.choice(fakers).city()
    location = rng.choice([f"{city}, {rng.choice(COUNTRIES)}", city])
    return name, f"{user}@{domain}", phone, location


def make_college(rng, fakers):
    f = rng.choice(fakers)
    pattern = rng.choice(UNIVERSITY_PATTERNS)
    return pattern.format(city=f.city(), surname=f.last_name())


SYLLABLES = ["ka", "zor", "lin", "vex", "tra", "mo", "qui", "dex", "flo", "ran", "pix",
             "sol", "ter", "nu", "bri", "gal", "cor", "zen", "op", "ly"]
SKILL_SUFFIXES = ["", "", "", "DB", ".js", "ML", "Flow", "Hub", "Stack", " Cloud", "QL", "OS"]


def invent_skill(rng):
    """A made-up tool name like 'Zorvex', 'Kalin.js' or 'TradexDB'.

    Data augmentation: if the model only ever sees real skill names it memorises
    the list instead of learning from context ("built ... using ___"). Mixing in
    invented names forces it to use the context, so it can find skills it has
    never seen.
    """
    word = "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3)))
    return word.capitalize() + rng.choice(SKILL_SUFFIXES)


def generate_resume(rng, fakers, skill_pool, invent_rate=0.0):
    b = ResumeBuilder()
    name, email, phone, location = make_person(rng, fakers)
    date_style = rng.choice(["short", "long", "numeric", "year"])
    dash = rng.choice([" - ", " – ", " to "])
    bullet = rng.choice(BULLETS)

    skills = rng.sample(skill_pool, rng.randint(5, 10))
    skills = [invent_skill(rng) if rng.random() < invent_rate else sk for sk in skills]
    n_jobs = rng.randint(1, 4)
    titles = rng.sample(JOB_TITLES, n_jobs)
    truth = {"name": name, "email": email, "phone": phone, "location": location,
             "skills": skills, "jobs": [], "education": []}

    # ---- Header ---------------------------------------------------------
    b.add(name.upper() if rng.random() < 0.3 else name, "NAME")
    b.newline()
    if rng.random() < 0.5:
        b.add(titles[0], "JOB_TITLE")
        b.newline()
    contact_style = rng.randrange(3)
    if contact_style == 0:
        b.add(location, "LOCATION"); b.add(" | ")
        b.add(phone, "PHONE"); b.add(" | ")
        b.add(email, "EMAIL")
        b.newline()
    elif contact_style == 1:
        b.add("Email: "); b.add(email, "EMAIL"); b.newline()
        b.add("Phone: "); b.add(phone, "PHONE"); b.newline()
        b.add("Address: "); b.add(location, "LOCATION"); b.newline()
    else:
        b.add(email, "EMAIL"); b.add("  ·  "); b.add(phone, "PHONE"); b.newline()
        b.add(location, "LOCATION"); b.newline()

    # ---- Summary --------------------------------------------------------
    sections = ["experience", "education", "skills"]
    if rng.random() < 0.3:
        sections = ["skills", "experience", "education"]
    if rng.random() < 0.7:
        b.newline()
        b.add(rng.choice(HEADINGS["summary"])); b.newline()
        a, c = rng.sample(skills, 2)
        b.add_template(rng.choice(SUMMARY_TEMPLATES), {
            "title": (titles[0], "JOB_TITLE"), "a": (a, "SKILL"), "b": (c, "SKILL"),
            "y": (str(rng.randint(2, 15)), None)})
        b.newline()

    for section in sections:
        b.newline()
        b.add(rng.choice(HEADINGS[section])); b.newline()

        if section == "experience":
            year = rng.randint(2023, 2026)
            entry_style = rng.randrange(4)
            for i, title in enumerate(titles):
                company = rng.choice(fakers).company()
                end_year = year
                start_year = end_year - rng.randint(1, 4)
                start = random_date(rng, start_year, date_style)
                end = "Present" if i == 0 and rng.random() < 0.6 else random_date(rng, end_year, date_style)
                year = start_year

                if entry_style == 0:      # Title at Company \n dates
                    b.add(title, "JOB_TITLE"); b.add(" at "); b.add(company, "COMPANY"); b.newline()
                    b.add(start, "DATE"); b.add(dash); b.add(end, "DATE"); b.newline()
                elif entry_style == 1:    # Title, Company (dates)
                    b.add(title, "JOB_TITLE"); b.add(", "); b.add(company, "COMPANY")
                    b.add(" ("); b.add(start, "DATE"); b.add(dash); b.add(end, "DATE"); b.add(")"); b.newline()
                elif entry_style == 2:    # Company \n Title | dates
                    b.add(company, "COMPANY"); b.newline()
                    b.add(title, "JOB_TITLE"); b.add(" | ")
                    b.add(start, "DATE"); b.add(dash); b.add(end, "DATE"); b.newline()
                else:                     # dates \n Title — Company
                    b.add(start, "DATE"); b.add(dash); b.add(end, "DATE"); b.newline()
                    b.add(title, "JOB_TITLE"); b.add(" — "); b.add(company, "COMPANY"); b.newline()

                job_skills = []
                for template in rng.sample(ACHIEVEMENT_TEMPLATES, rng.randint(1, 3)):
                    a, c = rng.sample(skills, 2)
                    if "{a}" in template:
                        job_skills.append(a)
                    if "{b}" in template:
                        job_skills.append(c)
                    b.add(f"{bullet} ")
                    b.add_template(template, {
                        "a": (a, "SKILL"), "b": (c, "SKILL"),
                        "n": (str(rng.randint(5, 60)), None), "k": (str(rng.randint(2, 9)), None)})
                    b.newline()
                truth["jobs"].append({"title": title, "company": company, "start": start,
                                      "end": end, "skills": sorted(set(job_skills))})

        elif section == "education":
            grad_year = rng.randint(2008, 2024)
            for _ in range(rng.choice([1, 1, 2])):
                degree, college = rng.choice(DEGREES), make_college(rng, fakers)
                grad = random_date(rng, grad_year, rng.choice(["year", date_style]))
                style = rng.randrange(3)
                if style == 0:
                    b.add(degree, "DEGREE"); b.add(", "); b.add(college, "COLLEGE")
                    b.add(" ("); b.add(grad, "DATE"); b.add(")"); b.newline()
                elif style == 1:
                    b.add(college, "COLLEGE"); b.newline()
                    b.add(degree, "DEGREE"); b.add(" | "); b.add(grad, "DATE"); b.newline()
                else:
                    b.add(degree, "DEGREE"); b.newline()
                    b.add(college, "COLLEGE"); b.add(", "); b.add("Graduated "); b.add(grad, "DATE"); b.newline()
                truth["education"].append({"degree": degree, "college": college, "year": grad})
                grad_year -= rng.randint(2, 5)

        else:  # skills
            sep = rng.choice([", ", " | ", " • "])
            if rng.random() < 0.3:  # one per line
                for s in skills:
                    b.add(f"{bullet} "); b.add(s, "SKILL"); b.newline()
            else:
                for j, s in enumerate(skills):
                    if j:
                        b.add(sep)
                    b.add(s, "SKILL")
                b.newline()

    return {"text": b.text.strip() + "\n", "entities": b.entities, "truth": truth}


def build_split(n, seed, skill_pool, invent_rate=0.0):
    rng = random.Random(seed)
    fakers = []
    for i, loc in enumerate(LOCALES):
        f = Faker(loc)
        f.seed_instance(seed * 100 + i)
        fakers.append(f)
    return [generate_resume(rng, fakers, skill_pool, invent_rate) for _ in range(n)]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    splits = {
        # different seeds -> different people, companies and schools in each split
        # 20% of training skills are replaced by invented names (augmentation)
        "train": build_split(N_TRAIN, SEED, SKILLS, invent_rate=0.2),
        "dev": build_split(N_DEV, SEED + 1, SKILLS),
        # the test set also includes skills the model has never seen
        "test": build_split(N_TEST, SEED + 2, SKILLS + HELD_OUT_SKILLS),
    }
    for name, docs in splits.items():
        for i, doc in enumerate(docs):
            doc["id"] = f"{name}_{i:03d}"
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for doc in docs:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        n_ents = sum(len(d["entities"]) for d in docs)
        print(f"{name:5s}: {len(docs)} resumes, {n_ents} labelled entities -> {path.name}")

    meta = {"held_out_skills": HELD_OUT_SKILLS}
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
