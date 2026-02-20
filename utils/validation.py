
def validate_resume_text(resume_text):
    text = resume_text.strip()

    if text == "":
        return False, "Please paste your resume."

    words = text.lower().split()

    if len(words) <= 1:
        return False, "Resume text seems invalid."

    if len(set(words)) <= 1:
        return False, "Resume text looks like repeated spam."

    real_words = [w for w in words if any(c.isalpha() for c in w)]

    if len(real_words) < 3:
        return False, "Resume text seems invalid."

    unique_ratio = len(set(words)) / max(len(words), 1)
    if len(words) > 6 and unique_ratio < 0.3:
        return False, "Resume text looks like repeated spam."

    return True, ""