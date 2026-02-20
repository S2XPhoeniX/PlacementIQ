import streamlit as st
import numpy as np
import joblib
import pandas as pd
import time
from nlp.resume_quality import calculate_resume_quality
from nlp.skill_extractor import extract_skills
from utils.validation import validate_resume_text
from utils.pdf_reader import extract_text_from_pdf
from models.predictor import predict_placement
from config.roles import role_difficulty, role_descriptions
from data.demo_profiles import (
    STRONG_RESUME,
    AVERAGE_RESUME,
    WEAK_RESUME,
    STRONG_JD,
    AVERAGE_JD,
    WEAK_JD,
)
from nlp.skill_extractor import extract_keywords

with open("assets/style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="PlacementIQ",
    layout="wide"
)

# ---------------- LOAD MODEL SAFELY ----------------
try:
    model = joblib.load("models/placement_model.pkl")
except:
    st.error("Model file not found. Please run train_model.py first.")
    st.stop()


from pathlib import Path

logo_path = Path(__file__).resolve().parent / "assets/logo_.jpg"

if logo_path.is_file():
    col1, col2, col3 = st.sidebar.columns([1,2,1])
    with col2:
        st.image(str(logo_path), width=180)
else:
    st.sidebar.error(f"Logo not found at {logo_path}")


# --- Header Card ---
st.sidebar.markdown("""
<div class="sidebar-card" style="text-align:center;">
    <div class="sidebar-title">PlacementIQ</div>
    <div class="sidebar-sub">AI Placement Readiness Analyzer</div>
    <div style="font-size:12px;margin-top:6px;">HackWave 2026 Project</div>
</div>
""", unsafe_allow_html=True)


# --- Features ---
st.sidebar.markdown("""
<div class="sidebar-card">
<div class="sidebar-section">Features</div>
<div class="sidebar-text">
• Resume Parsing <br>
• Skill Matching <br>
• ML Prediction <br>
• Personalized Roadmap
</div>
</div>
""", unsafe_allow_html=True)

# --- About ---
st.sidebar.markdown("""
<div class="sidebar-card">
<div class="sidebar-section">About This Tool</div>
<div class="sidebar-text">
Analyzes resume-job alignment and predicts placement readiness
using machine learning.
</div>
</div>
""", unsafe_allow_html=True)

# --- Tips ---
st.sidebar.markdown("""
<div class="sidebar-card">
<div class="sidebar-section">Tips</div>
<div class="sidebar-text">
• Paste full resume text <br>
• Upload resume PDF <br>
• Use real job descriptions <br>
• Include projects & internships
</div>
</div>
""", unsafe_allow_html=True)

# --- Why Tool ---
st.sidebar.markdown("""
<div class="sidebar-card">
<div class="sidebar-section">Why PlacementIQ?</div>
<div class="sidebar-text">
Helps students measure placement readiness early
and improve targeted skills before interviews.
</div>
</div>
""", unsafe_allow_html=True)

# --- Model Info ---
st.sidebar.markdown("""
<div class="sidebar-card">
<div class="sidebar-section">Model Info</div>
<div class="sidebar-text">
Algorithm: Logistic Regression <br>
Inputs: CGPA, Internship, Projects, Skills, Communication <br>
Dataset: Campus Placement Dataset <br>
Model Accuracy: 79.2%
</div>
</div>
""", unsafe_allow_html=True)



# --- Team ---
st.sidebar.markdown("""
<div class="sidebar-card" style="text-align:center;">
<div class="sidebar-section">Team</div>
<div class="sidebar-text">PlacementIQ Team ❤️</div>
</div>
""", unsafe_allow_html=True)



st.markdown("""
<div class="hero">
    <div class="hero-title">PlacementIQ</div>
    <div class="hero-sub">AI Placement Readiness Analyzer</div>
    <div class="hero-desc">
        Enterprise-grade placement intelligence using machine learning
    </div>
</div>
""", unsafe_allow_html=True)




st.divider()

# ---------------- INPUT SECTION ----------------
left, right = st.columns([2,1])

with left:
    st.markdown(f"""
    <div class="card-hover neutral-card">
        <h3 style="margin-bottom:6px;">Student Profile Input</h3>
        <p style="font-size:14px; color:#555;">
        Provide academic details and resume information for analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown(f"""
    <div class="card-hover neutral-card">
        <h3 style="margin-bottom:6px;">Analysis Overview</h3>
        <p style="font-size:14px; color:#555;">
        Our AI analyzes resume-job match, skills, and placement readiness.
        </p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")



demo1, demo2, demo3 = st.columns(3)




# ---------- MODIFY DEMO BUTTONS ----------
with demo1:
    if st.button(" Strong Candidate"):
        st.session_state.cgpa = 8.8
        st.session_state.internship = 1
        st.session_state.projects = 4
        st.session_state.communication = 8
        st.session_state.dsa_score = 8
        st.session_state.hackathons = 3
        st.session_state.resume_demo = STRONG_RESUME
        st.session_state.jd_demo = STRONG_JD

with demo2:
    if st.button(" Average Candidate"):
        st.session_state.cgpa = 7.2
        st.session_state.internship = 0
        st.session_state.projects = 2
        st.session_state.communication = 6
        st.session_state.dsa_score = 5
        st.session_state.hackathons = 1
        st.session_state.resume_demo = AVERAGE_RESUME
        st.session_state.jd_demo = AVERAGE_JD

with demo3:
    if st.button(" Weak Candidate"):
        st.session_state.cgpa = 6.0
        st.session_state.internship = 0
        st.session_state.projects = 0
        st.session_state.communication = 4
        st.session_state.dsa_score = 2
        st.session_state.hackathons = 0
        st.session_state.resume_demo = WEAK_RESUME
        st.session_state.jd_demo = WEAK_JD


# ---------- DEFAULT VALUES ----------
if "cgpa" not in st.session_state:
    st.session_state.cgpa = 7.5
if "internship" not in st.session_state:
    st.session_state.internship = 0
if "projects" not in st.session_state:
    st.session_state.projects = 2
if "communication" not in st.session_state:
    st.session_state.communication = 6
if "dsa_score" not in st.session_state:
    st.session_state.dsa_score = 5
if "hackathons" not in st.session_state:
    st.session_state.hackathons = 1


# ---------- INPUT SLIDERS ----------
col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider("CGPA", 5.0, 10.0, st.session_state.cgpa)
    internship = st.selectbox(
        "Internship Experience",
        [0, 1],
        index=st.session_state.internship
    )
    projects = st.slider("Number of Projects", 0, 5, st.session_state.projects)
    communication = st.slider(
        "Communication Skill (1-10)",
        1, 10,
        st.session_state.communication
    )
    dsa_score = st.slider(
        "DSA / Coding Skill (1-10)",
        1, 10,
        st.session_state.dsa_score
    )
    hackathons = st.slider(
        "Hackathons / Certifications",
        0, 5,
        st.session_state.hackathons
    )


# ---------- SAVE BACK ----------
st.session_state.cgpa = cgpa
st.session_state.internship = internship
st.session_state.projects = projects
st.session_state.communication = communication
st.session_state.dsa_score = dsa_score
st.session_state.hackathons = hackathons




with col2:
    role = st.selectbox(
    "Select Target Role",
    [
        "Custom",
        "Python Developer",
        "Data Analyst",
        "ML Engineer",
        "Backend Developer",
        "Frontend Developer",
        "Full Stack Developer",
        "Software Engineer",
        "Mechanical Engineer",
        "Cybersecurity Analyst",
        "DevOps Engineer",
        "Cloud Engineer",
        "Business Analyst",
        "QA Engineer",
        "AI Research Intern",
        "Mobile App Developer",
        "Database Engineer",
        "Product Engineer",
        "Robotics Engineer",
        "Data Scientist",
        "Amazon SDE",
        "Google SWE",
        "Infosys Graduate Engineer",
        "Startup Intern"
    ]
)
    


    # ---------- Resume Input ----------
    st.markdown("### Resume Input")

    extracted_text = ""

    uploaded_file = st.file_uploader(
        "Upload your resume PDF",
        type=["pdf"],
        key="resume_pdf_upload"
    )

    if uploaded_file is not None:
        ok, text = extract_text_from_pdf(uploaded_file)

        if ok:
            extracted_text = text
            st.success(f"Uploaded: {uploaded_file.name}")
        else:
            st.error("Could not read this PDF. Try another resume.")

    # 🔹 Lock typing if PDF uploaded
    if extracted_text != "":
        resume_text = st.text_area(
            "Resume Text (from PDF)",
            extracted_text,
            height=200,
            disabled=True
        )
    else:
        resume_text = st.text_area(
        "Paste Resume Text",
        value=st.session_state.get("resume_demo", "")
    )



    if role == "Custom":
        job_description = st.text_area(
            "Paste Job Description",
            value=st.session_state.get("jd_demo", "")
        )

    else:
        job_description = role_descriptions[role]
        st.text_area("Job Description (Auto-filled)", job_description, disabled=True)

# ---------- SKILL TAG FUNCTION ----------
def show_tags(skills, color="#e8f4ff"):
    tag_html = ""
    for s in skills:
        tag_html += f'<span class="skill-tag" style="background:{color};">{s.title()}</span>'
    st.markdown(tag_html, unsafe_allow_html=True)

def show_tags(skills, color="#e8f4ff"):
    tag_html = ""
    for s in skills:
        tag_html += f'<span style="background:{color}; padding:6px 10px; border-radius:8px; margin:4px; display:inline-block;">{s.title()}</span>'
    st.markdown(tag_html, unsafe_allow_html=True)

      
# ---------------- ANALYZE ----------------
if st.button("Analyze"):

    # ---------- VALIDATION ----------
    if not resume_text.strip():
        st.error("Please paste resume text.")
        st.stop()

    if not job_description.strip():
        st.error("Please provide job description.")
        st.stop()
    
    with st.spinner("Analyzing profile..."):

        progress = st.progress(0)

        st.write(" Extracting resume skills...")
        time.sleep(0.6)
        progress.progress(25)

        st.write(" Matching with job description...")
        time.sleep(0.6)
        progress.progress(50)

        st.write(" Running placement prediction model...")
        time.sleep(0.6)
        progress.progress(75)

        st.write(" Generating improvement roadmap...")
        time.sleep(0.6)
        progress.progress(100)

        #SKILLS EXTRACTOR
        resume_skills, job_skills = extract_skills(resume_text, job_description)
        # -------- ATS KEYWORD SCORE --------
        jd_keywords = extract_keywords(job_description)
        resume_keywords = extract_keywords(resume_text)

        matched_keywords = set(jd_keywords) & set(resume_keywords)
        missing_keywords = set(jd_keywords) - set(resume_keywords)

        if len(jd_keywords) > 0:
            ats_score = round(len(matched_keywords) / len(jd_keywords) * 100, 2)
        else:
            ats_score = 0

        # ---------- RESUME QUALITY ----------   # out of 10
        resume_quality = calculate_resume_quality(resume_text, resume_skills)
        
        # ---- Resume Validation ----

        text = resume_text.strip()

        if text == "":
            st.error("Please paste your resume.")
            st.stop()

        words = text.lower().split()

        # garbage like "asdfgh"
        if len(words) <= 1:
            st.error("Resume text seems invalid.")
            st.stop()

        # repeated spam like "python python python"
        if len(set(words)) <= 1:
            st.error("Resume text looks like repeated spam.")
            st.stop()

        # few skills → just warn
        if len(resume_skills) == 0:
            st.warning("No recognizable technical skills found.")
        elif len(resume_skills) <= 2:
            st.info("Few skills detected. Consider adding more skills.")


        # ---------- SMART RESUME VALIDATION ----------
        is_valid, msg = validate_resume_text(resume_text)

        if not is_valid:
            st.error(msg)
            st.stop()

        match_percentage = 0
        if len(job_skills) > 0:
            match_percentage = round(
                (len(set(resume_skills).intersection(job_skills)) / len(job_skills)) * 100, 2
            )

        missing_skills = list(set(job_skills) - set(resume_skills))

        # ---------- MODEL ----------
        ok, result = predict_placement(
            model,
            cgpa,
            internship,
            communication,
            match_percentage,
            projects,
            dsa_score,
            resume_quality,
            hackathons,
            role,
            role_difficulty
        )

        if not ok:
            st.error("Prediction failed. Please check inputs.")
            st.caption(result)   # optional debug
            st.stop()

        probability = result



        st.divider()

        st.markdown('<div class="results-box">Results</div>', unsafe_allow_html=True)


        colA, colB, colC = st.columns(3)

        
        with colA:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Match %</h4>
                <h2>{match_percentage}%</h2>
            </div>
            """, unsafe_allow_html=True)

        with colB:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Placement Probability</h4>
                <h2>{probability}%</h2>
            </div>
            """, unsafe_allow_html=True)

        with colC:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Matched Skills</h4>
                <h2>{len(resume_skills)}</h2>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.write(f"Resume Quality Score: {resume_quality}/10")
        st.subheader("Job-Specific ATS Keywords")
        st.write(f"ATS Score: {ats_score}%")
        
        st.caption(
            "Core Skill Gaps = fundamental skills missing from your profile. "
            "ATS Keywords = job-specific terms recruiters search in resumes."
            )

        st.write("Matched Keywords:")
        st.write(", ".join(list(matched_keywords)[:10]))

        st.write("Missing Keywords:")
        st.write(", ".join(list(missing_keywords)[:10]))

        # ---------- WHY THIS SCORE ----------
        st.subheader(" Why this score?")

        reasons = []

        if cgpa >= 8:
            reasons.append(f"Strong CGPA (+{round(cgpa*1.5,1)}%)")
        elif cgpa < 6.5:
            reasons.append("Low CGPA (-8%)")

        if internship == 1:
            reasons.append("Internship experience (+10%)")
        else:
            reasons.append("No internship (-10%)")

        if match_percentage >= 70:
            reasons.append(f"Good skill match (+{round(match_percentage/5,1)}%)")
        elif match_percentage < 40:
            reasons.append("Low skill match (-10%)")

        if communication >= 7:
            reasons.append("Good communication (+6%)")
        elif communication <= 4:
            reasons.append("Weak communication (-6%)")

        if dsa_score >= 7:
            reasons.append("Strong DSA skills (+8%)")
        elif dsa_score <= 3:
            reasons.append("Weak DSA skills (-8%)")

        if hackathons >= 2:
            reasons.append("Certifications/Hackathons (+5%)")

        if len(reasons) == 0:
            st.write("Balanced profile with no major strengths or weaknesses.")
        else:
            for r in reasons:
                st.write("•", r)


        st.caption("Prediction confidence derived from academic scores, internship exposure, skill alignment, coding strength, and resume quality metrics.")

        # ---------- Resume Strength Meter ----------
        st.subheader("📄 Resume Strength")

        st.progress(int(resume_quality * 10))

        if resume_quality < 4:
            st.warning("Resume needs major improvement.")
        elif resume_quality < 7:
            st.info("Resume is decent but can be improved.")
        else:
            st.success("Strong resume structure.")


        st.subheader(" Why This Score?")

        reasons = []

        if cgpa >= 8:
            reasons.append("Strong academic performance boosted your score.")
        elif cgpa < 6.5:
            reasons.append("Low CGPA reduced your placement chances.")

        if internship == 1:
            reasons.append("Internship experience increased industry readiness.")
        else:
            reasons.append("No internship experience lowered real-world exposure.")

        if match_percentage >= 70:
            reasons.append("Your skills match the job requirements well.")
        elif match_percentage < 40:
            reasons.append("Low skill match with job description reduced score.")

        if dsa_score >= 7:
            reasons.append("Strong DSA/problem solving is a big advantage.")
            
        if hackathons >= 2:
            reasons.append("Hackathons/certifications improved profile strength.")

        if resume_quality >= 7:
            reasons.append("Well-structured resume improved recruiter impression.")

        for r in reasons:
            st.write("•", r)



        st.write("Placement Readiness Score")
        st.markdown(f"""
        <div class="score-circle"
            style="background:conic-gradient(#1f77b4 {probability}%, #eee {probability}%);">
            <b>{probability}%</b>
        </div>
        """, unsafe_allow_html=True)


        st.subheader("Skill Profile")
        skills_df = pd.DataFrame({
            "Category": ["CGPA", "Projects", "Skill Match", "Communication"],
            "Score": [cgpa*10, projects*20, match_percentage, communication*10]
        })
        st.bar_chart(skills_df.set_index("Category"))

        st.subheader(f"Core Skill Gaps ({len(missing_skills)})")

        if missing_skills:
            for skill in missing_skills:
                st.write(f"• {skill.title()}")
        else:
            st.success("No major skill gaps detected.")

        if probability >= 80:
            st.success("High Placement Readiness")
        elif probability >= 50:
            st.warning("Moderate Placement Readiness")
        else:
            st.error("Low Placement Readiness - Improvement Required")

        st.divider()
        st.info(
            "This score reflects academic performance, project experience, "
            "skill alignment with job role, and communication ability."
        )

        # ---------- EXPLAINABILITY PANEL ----------
        st.divider()
        st.subheader(" Why This Score?")

        strengths = []
        improvements = []

        if cgpa >= 8:
            strengths.append("Strong CGPA")
        else:
            improvements.append("Improve academic performance")

        if internship == 1:
            strengths.append("Internship experience")
        else:
            improvements.append("Get internship experience")

        if match_percentage >= 70:
            strengths.append("Good skill match with job role")
        elif match_percentage >= 40:
            improvements.append("Improve skill match with job role")
        else:
            improvements.append("Major skill gaps for target role")

        if dsa_score >= 7:
            strengths.append("Good Dsa / Coding ability")
        else:
            improvements.append("Practice Dsa problems")

        if hackathons >= 2:
            strengths.append("Active in hackathons / certifications")
        else:
            improvements.append("Do certifications or hackathons")

        if resume_quality >= 7:
            strengths.append("Strong resume quality")
        else:
            improvements.append("Improve resume structure and achievements")

        st.markdown("### Strengths")
        for s in strengths:
            st.success(f"▸ {s}")

        st.markdown("### Needs Improvement")
        for i in improvements:
            st.warning(f"▸ {i}")

        # ---------- INSIGHT PANEL ----------
        st.divider()
        st.markdown(
        "<h3 style='text-align:center;'>Placement Insights</h3>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)


        # ---------- STRENGTHS ----------
        with col1:

            strengths = []

            if cgpa >= 8:
                strengths.append("Strong academic performance")
            if internship == 1:
                strengths.append("Internship experience adds real-world exposure")
            if len(resume_skills) >= 4:
                strengths.append("Good skill coverage for the role")
            if dsa_score >= 7:
                strengths.append("Strong problem-solving ability")
            if hackathons >= 2:
                strengths.append("Active in hackathons / certifications")

            if not strengths:

                strengths.append("Keep Hustling Hard !!")

            html = "<br>".join([f"• {s}" for s in strengths])

            st.markdown(
                f"""
                <div class="box-hover info-box">
                    <h3> Strengths</h3>
                    {html}
                </div>
                """,
                unsafe_allow_html=True
            )


        # ---------- WEAK AREAS ----------
        with col2:

            weak = []

            if cgpa < 7:
                weak.append("Low CGPA may affect shortlist chances")
            if internship == 0:
                weak.append("No internship experience")
            if len(resume_skills) <= 2:
                weak.append("Limited technical skills detected")
            if match_percentage < 50:
                weak.append("Resume not aligned with job role")
            if dsa_score <= 5:
                weak.append("Improve coding/problem-solving skills")

            if not weak:
                weak.append("No major weak areas — keep growing 🔥")

            html = "<br>".join([f"• {w}" for w in weak])

            st.markdown(
                f"""
                <div class="box-hover info-box" style="text-align:center;">
                    <h3> Weak Areas</h3>
                    {html}
                </div>
                """,
                unsafe_allow_html=True
            )


        
        st.subheader(" Exact Next Steps")

        if match_percentage < 60:
            st.write("• Add missing skills from job description")

        if internship == 0:
            st.write("• Try 1–2 internships or open-source projects")

        if dsa_score < 7:
            st.write("• Solve 150+ DSA problems on LeetCode")

        if resume_quality < 6:
            st.write("• Add quantified achievements (numbers, impact)")

        if hackathons < 2:
            st.write("• Participate in hackathons or certifications")

        if cgpa < 7.5:
            st.write("• Focus on academics next semester")

        st.write("• Do mock interviews weekly")
        st.write("• Build 1 real-world project")
        

        st.subheader("Learning Roadmap")

        ROADMAP_LIBRARY = {
            "python": "Complete Python OOP and build 2 mini projects.",
            "sql": "Practice SQL joins and queries.",
            "machine learning": "Build one ML project.",
            "data analysis": "Learn Pandas + visualization.",
            "git": "Learn Git branching and maintain repo.",
            "communication": "Practice mock interviews.",
            "data structures": "Solve 100+ DSA problems."
        }

        if missing_skills:
            for skill in missing_skills:
                if skill in ROADMAP_LIBRARY:
                    st.write("•", ROADMAP_LIBRARY[skill])
        else:
            st.write("No major skill gaps. Focus on advanced projects.")
        
        # ---------- Recruiter Recommendation ----------
        st.subheader(" Recruiter Recommendation")

        if probability >= 85:
            st.success("Ready to apply to top product companies.")
        elif probability >= 65:
            st.info("Ready for mid-tier companies. Improve 1-2 skills for top companies.")
        else:
            st.warning("Focus on internships, DSA, and skill matching before applying.")

        # ---------- Company Readiness ----------
        st.subheader(" Company Readiness")

        company_targets = {
            "Amazon SDE": 1.2,
            "Google SWE": 1.25,
            "Infosys Graduate Engineer": 0.9,
            "Startup Intern": 0.8
        }

        for company, difficulty in company_targets.items():
            company_score = probability / difficulty
            company_score = round(max(5, min(company_score, 95)), 1)

            if company_score >= 85:
                st.success(f"{company} → {company_score}% Ready")
            elif company_score >= 65:
                st.info(f"{company} → {company_score}% Almost Ready")
            else:
                st.warning(f"{company} → {company_score}% Needs Improvement")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:14px;'>"
    "PlacementIQ • HackWave 2026 • Built with ❤️"
    "</p>",
    unsafe_allow_html=True
)
