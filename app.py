import streamlit as st
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity

# A basic skill dictionary (you can expand this later)
SKILL_SET = {
    'python', 'java', 'sql', 'machine learning', 'deep learning', 'data analysis',
    'power bi', 'excel', 'communication', 'teamwork', 'pandas', 'numpy', 'keras',
    'tensorflow', 'scikit-learn', 'matplotlib', 'tableau', 'nlp', 'transformers'
}

# Load the pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="Resume–Job Fit Checker", page_icon="🧠")
st.title("🧠 Resume–Job Fit Predictor")
st.write("Upload your resume and paste a job description to check how well they match!")

import re

def extract_skills(text):
    text = text.lower()
    found = set()
    for skill in SKILL_SET:
        # use word boundaries to match whole words only
        if re.search(rf'\b{re.escape(skill)}\b', text):
            found.add(skill)
    return found


# --- Resume Input ---
resume_text = ""

resume_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
if resume_file:
    resume_text = extract_text(resume_file)
else:
    resume_text = st.text_area("Or paste your Resume text", height=200)

# --- Job Description Input ---
job_text = st.text_area("Paste the Job Description", height=200)

# --- Predict Fit Score ---
if st.button("Check Fit Score"):
    # Extract and match skills
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matched_skills = resume_skills & job_skills
    missing_skills = job_skills - resume_skills

    st.subheader("🧩 Skill Match Breakdown")
    st.write(f"✅ Matched Skills ({len(matched_skills)}): {', '.join(matched_skills) if matched_skills else 'None'}")
    st.write(f"❌ Missing Skills ({len(missing_skills)}): {', '.join(missing_skills) if missing_skills else 'None'}")

    skill_match_ratio = len(matched_skills) / (len(job_skills) or 1)
    st.progress(skill_match_ratio)

    if not resume_text or not job_text:
        st.error("Please provide both resume and job description.")
    else:
        # Embed both texts
        resume_vec = model.encode([resume_text])
        job_vec = model.encode([job_text])

        # Calculate cosine similarity
        score = cosine_similarity(resume_vec, job_vec)[0][0]

        st.subheader("🔎 Fit Score:")
        st.metric("Match Score", f"{score:.2f}")

        # Score interpretation
        if score >= 0.8:
            st.success("High Match ✅")
        elif score >= 0.5:
            st.warning("Medium Match ⚠️")
        else:
            st.error("Low Match ❌")
