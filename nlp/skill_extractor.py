
SKILLS = {
    "python": ["python"],
    "sql": ["sql"],
    "machine learning": ["machine learning", "ml"],
    "data analysis": ["data analysis", "analysis"],
    "git": ["git", "github"],
    "communication": ["communication", "presentation", "teamwork"],
}

def extract_skills(resume_text, job_description):
    resume_clean = resume_text.lower()
    job_clean = job_description.lower()

    resume_skills = []
    job_skills = []

    for skill, keywords in SKILLS.items():
        if any(k in resume_clean for k in keywords):
            resume_skills.append(skill)

        if any(k in job_clean for k in keywords):
            job_skills.append(skill)

    return resume_skills, job_skills

def extract_keywords(text):
    words = text.lower().replace(",", " ").split()

    stopwords = {
        "the","and","with","for","a","an","to","of","in",
        "role","candidate","should","experience","looking",
        "we","are","required","skills","job","work"
    }

    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return list(set(keywords))