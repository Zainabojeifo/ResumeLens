"""
ResumeLens demo app.   Run:  streamlit run app.py

Upload a resume (PDF, Word or image) and see the parsed profile and job-match scores.
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from step5_integrate import JOB_DESCRIPTIONS, match_scores, parse_resume  # noqa: E402

COLORS = {"NAME": "#cfe2fb", "EMAIL": "#d7f2e6", "PHONE": "#d7f2e6", "LOCATION": "#fde7c2",
          "JOB_TITLE": "#fbd5c4", "COMPANY": "#e3defa", "DATE": "#eeeeec",
          "DEGREE": "#f9d7e4", "COLLEGE": "#f9d7e4", "SKILL": "#cdeedf"}

st.set_page_config(page_title="ResumeLens", layout="wide")
st.title("ResumeLens")
st.caption("Upload a resume. I extract the text (OCR for scans), find the key details with my "
           "trained spaCy model, link them together and score the candidate against job roles.")

upload = st.file_uploader("Resume file", type=["pdf", "docx", "png", "jpg", "jpeg", "tiff", "txt"])
if upload:
    with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as tmp:
        tmp.write(upload.getbuffer())
    with st.spinner("Parsing..."):
        profile = parse_resume(tmp.name)

    left, right = st.columns([1, 1])
    with left:
        st.subheader(profile["name"] or "Unknown name")
        st.write(f"**Email:** {profile['email']}  \n**Phone:** {profile['phone']}  \n"
                 f"**Location:** {profile['location']}  \n"
                 f"*Extracted with:* `{profile['extraction_method']}`")
        st.markdown("**Skills:** " + ", ".join(profile["skills"]))
        st.markdown("#### Experience")
        st.dataframe(pd.DataFrame(profile["jobs"]), use_container_width=True)
        st.markdown("#### Education")
        st.dataframe(pd.DataFrame(profile["education"]), use_container_width=True)
    with right:
        st.markdown("#### Job match")
        rows = [{"Role": jd["role"], "Score": match_scores([profile], jd)[0],
                 "Required skills": ", ".join(jd["required_skills"])} for jd in JOB_DESCRIPTIONS]
        st.dataframe(pd.DataFrame(rows).sort_values("Score", ascending=False), use_container_width=True)
        st.markdown("#### Entities found")
        html = profile["clean_text"]
        for ent in sorted({(e["text"], e["label"]) for e in profile["entities"]}, key=lambda x: -len(x[0])):
            html = html.replace(ent[0], f"<mark style='background:{COLORS[ent[1]]};padding:0 3px;"
                                        f"border-radius:3px'>{ent[0]} <sub>{ent[1]}</sub></mark>", 1)
        st.markdown(f"<div style='white-space:pre-wrap;font-size:0.9rem'>{html}</div>",
                    unsafe_allow_html=True)
