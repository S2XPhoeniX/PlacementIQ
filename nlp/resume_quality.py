def calculate_resume_quality(resume_text, resume_skills):
    word_count = len(resume_text.split())
    numbers_found = sum(c.isdigit() for c in resume_text)

    score = (
        min(word_count / 200, 1) * 4 +
        min(len(resume_skills) / 6, 1) * 4 +
        min(numbers_found / 10, 1) * 2
    )

    return round(score, 2)