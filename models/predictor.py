# models/predictor.py
import numpy as np

def predict_placement(
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
):
    try:
        input_features = np.array([[
            cgpa,
            internship,
            communication,
            match_percentage
        ]])

        raw_prob = model.predict_proba(input_features)[0][1]

        probability = raw_prob * 60
        probability += projects * 1.2
        probability += dsa_score * 0.8
        probability += resume_quality * 0.7
        probability += hackathons * 0.6

        probability *= role_difficulty.get(role, 1.0)

        if internship == 0:
            probability *= 0.9
        if cgpa < 6.5:
            probability *= 0.85
        if match_percentage < 40:
            probability *= 0.8

        probability = max(5, min(probability, 95))
        return True, round(probability, 2)

    except Exception as e:
        return False, str(e)